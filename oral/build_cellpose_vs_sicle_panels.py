#!/usr/bin/env python3
"""Panels: Cellpose vs SICLE + per-cell winners (why better/worse)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS

CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_ROOT = RUNS / "nf_sweep_full"
METRICS = RUNS / "cellpose_vs_sicle" / "metrics_cellpose_vs_sicle.csv"
PANELS = RUNS / "cellpose_vs_sicle" / "panels"
GT_CYAN = (0, 255, 255)
GT_RED = (255, 0, 0)  # zoom slides: GT drawn on top of predictions
CP_YELLOW = (255, 255, 0)
SICLE_GREEN = (0, 220, 0)
WIN_CP = (255, 180, 0)
WIN_SICLE = (80, 255, 120)
LOSE = (255, 70, 70)
MIN_ZOOM_COL_W = 132
ZOOM_GAP = 8
ZOOM_TITLE_H = 34


def _dejavu_fonts(
    bold_size: int = 10,
    regular_size: int = 9,
) -> tuple:
    from PIL import ImageFont

    try:
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", bold_size
        )
        regular = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", regular_size
        )
        return bold, regular
    except OSError:
        fb = ImageFont.load_default()
        return fb, fb


def _text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _draw_centered_line(draw, text: str, x0: int, col_w: int, y: int, font) -> None:
    tw = _text_width(draw, text, font)
    draw.text((x0 + max(0, (col_w - tw) // 2), y), text, fill=(0, 0, 0), font=font)


def _draw_column_titles(
    draw,
    specs: list[tuple[str, str | None]],
    col_w: int,
    gap: int,
    title_h: int,
) -> None:
    """One or two centered lines per column (method name + optional BR)."""
    font_b, font_r = _dejavu_fonts(10, 9)
    x = 0
    for line1, line2 in specs:
        if line2 is None:
            _draw_centered_line(draw, line1, x, col_w, (title_h - 12) // 2, font_b)
        else:
            _draw_centered_line(draw, line1, x, col_w, 5, font_b)
            _draw_centered_line(draw, line2, x, col_w, 19, font_r)
        x += col_w + gap


def _load_sicle_mask(category: str, stem: str) -> np.ndarray:
    for root in (SICLE_ROOT, CP_ROOT):
        case = root / category / stem
        for pattern in ("nf2_n0200_raw", "nf2_n0*_raw", "sicle_raw"):
            if "*" in pattern:
                paths = list(case.glob(f"{pattern}/merged_percell_sicle_masks_int32.npy"))
            else:
                paths = [case / pattern / "merged_percell_sicle_masks_int32.npy"]
            for p in paths:
                if p.is_file() and p.stat().st_size > 0:
                    return np.load(p).astype(np.int32)
    raise FileNotFoundError(f"No SICLE mask for {category}/{stem}")


def _per_cell_br(gt: np.ndarray, pr: np.ndarray, margin: int = 8) -> dict[int, float]:
    from percell_boundary_recall import (
        bbox_of_mask,
        compute_boundary_recall,
        isolate_pred_for_gt,
    )

    out: dict[int, float] = {}
    h, w = gt.shape
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        m = gt == gid
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(h, r1 + margin), min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_iso = np.where(gt_crop == gid, gt_crop, 0)
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, gid)
        br, _, _ = compute_boundary_recall(pr_iso, gt_iso)
        out[gid] = float(br)
    return out


def _highlight_cells(base, gt, gids, color, thick=2):
    from percell_boundary_recall import draw_contours

    img = base.copy()
    for gid in gids:
        m = np.where(gt == gid, gid, 0).astype(np.int32)
        img = draw_contours(img, m, color, thickness=thick)
    return img


def _roi_panel_images(
    category: str,
    stem: str,
) -> list[tuple[str, np.ndarray]]:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    case = CP_ROOT / category / stem
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = _load_sicle_mask(category, stem)

    br_cp = br_si = None
    if METRICS.is_file():
        with METRICS.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row["category"] == category and row["roi"] == stem:
                    if row["method"] == "cellpose":
                        br_cp = float(row["br_mean_strict"])
                    else:
                        br_si = float(row["br_mean_strict"])

    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    return [
        ("Original", orig),
        ("GT (ciano)", base_gt),
        (f"Cellpose  BR={br_cp:.3f}" if br_cp else "Cellpose", draw_contours(base_gt.copy(), cp, CP_YELLOW, 1)),
        (f"SICLE Nf=2  BR={br_si:.3f}" if br_si else "SICLE Nf=2", draw_contours(base_gt.copy(), si, SICLE_GREEN, 1)),
    ]


def export_roi_tiles(category: str, stem: str, dest: Path) -> None:
    """Four PNG tiles for Beamer 2x2 layout (slide 6)."""
    from PIL import Image

    names = ("cp2_orig", "cp2_gt", "cp2_cellpose", "cp2_sicle")
    dest.mkdir(parents=True, exist_ok=True)
    for (title, img), name in zip(_roi_panel_images(category, stem), names):
        out = dest / f"{name}.png"
        Image.fromarray(img).save(out)
        print(f"  {out.name}  ({title})")


def build_roi_panel(category: str, stem: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    panels = _roi_panel_images(category, stem)
    h, w = panels[0][1].shape[:2]
    gap, th, foot = 6, 28, 40
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + th + foot, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[th : th + h, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = font_sm = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 4, 6), title, fill=(0, 0, 0), font=font)
        x += w + gap
    br_cp = br_si = None
    if METRICS.is_file():
        with METRICS.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if row["category"] == category and row["roi"] == stem:
                    if row["method"] == "cellpose":
                        br_cp = float(row["br_mean_strict"])
                    else:
                        br_si = float(row["br_mean_strict"])
    if br_cp is not None and br_si is not None:
        draw.text(
            (4, h + th + 6),
            f"{category}/{stem} — SICLE cru vs Cellpose. ΔBR={br_si - br_cp:+.3f}",
            fill=(40, 40, 40),
            font=font_sm,
        )
    out = PANELS / f"{category}_{stem}_cellpose_vs_sicle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    return out


def _percell_winner_images(
    category: str,
    stem: str,
    min_delta: float = 0.08,
    max_cells: int = 8,
) -> tuple[list[tuple[str, np.ndarray]], list[int], list[int]]:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    case = CP_ROOT / category / stem
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = _load_sicle_mask(category, stem)

    br_cp = _per_cell_br(gt, cp)
    br_si = _per_cell_br(gt, si)
    win_s, win_c = [], []
    for gid in br_cp:
        if gid not in br_si:
            continue
        d = br_si[gid] - br_cp[gid]
        if d >= min_delta:
            win_s.append((d, gid))
        elif d <= -min_delta:
            win_c.append((d, gid))
    win_s.sort(reverse=True)
    win_c.sort()
    gids_s = [g for _, g in win_s[:max_cells]]
    gids_c = [g for _, g in win_c[:max_cells]]

    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    pan_win = draw_contours(base_gt.copy(), gt, GT_CYAN, 1)
    pan_win = _highlight_cells(pan_win, gt, gids_s, WIN_SICLE, 2)
    pan_win = _highlight_cells(pan_win, gt, gids_c, WIN_CP, 2)

    panels = [
        ("GT", draw_contours(orig, gt, GT_CYAN, thickness=1)),
        ("Cellpose", draw_contours(base_gt.copy(), cp, CP_YELLOW, 1)),
        ("SICLE Nf=2", draw_contours(base_gt.copy(), si, SICLE_GREEN, 1)),
        ("Quem ganha por celula", pan_win),
    ]
    return panels, gids_s, gids_c


def export_percell_highlight(category: str, stem: str, dest: Path) -> None:
    from PIL import Image

    panels, _, _ = _percell_winner_images(category, stem)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "percell_winners_highlight.png"
    Image.fromarray(panels[-1][1]).save(out)
    print(f"  {out.name}")


def build_percell_winners(category: str, stem: str, min_delta: float = 0.08, max_cells: int = 8) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    panels, gids_s, gids_c = _percell_winner_images(category, stem, min_delta, max_cells)
    h, w = panels[0][1].shape[:2]
    gap, th, foot = 6, 26, 56
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + th + foot, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[th : th + h, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except OSError:
        font = font_sm = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 3, 5), title, fill=(0, 0, 0), font=font)
        x += w + gap

    lines = [
        f"{stem}: verde=SICLE melhor ({len(gids_s)} cel.), laranja=Cellpose melhor ({len(gids_c)} cel.)",
        "SICLE ganha: borda segue gradiente da saliencia (gradvmaxmul).",
        "Cellpose ganha: nucleo/flow ok mas borda lateral falha ou SICLE encolhe (Nf/minsc).",
    ]
    y = h + th + 4
    for line in lines:
        draw.text((4, y), line, fill=(30, 30, 30), font=font_sm)
        y += 12
    if gids_s:
        draw.text((4, y), f"Ex. SICLE melhor: celulas {gids_s[:5]}", fill=(0, 100, 0), font=font_sm)
        y += 12
    if gids_c:
        draw.text((4, y), f"Ex. Cellpose melhor: celulas {gids_c[:5]}", fill=(180, 80, 0), font=font_sm)

    out = PANELS / f"{category}_{stem}_percell_winners.png"
    pil.save(out)
    return out


def find_best_zoom_roi(
    candidates: list[tuple[str, str]] | None = None,
    min_delta: float = 0.08,
) -> tuple[str, str, int, int, float, float]:
    """ROI maximizing |BR_sicle - BR_cp| on best SICLE-win and CP-win cells."""
    if candidates is None:
        candidates = [
            (d.name, c.name)
            for d in sorted(CP_ROOT.iterdir())
            if d.is_dir()
            for c in sorted(d.iterdir())
            if (c / "gt" / "gold_standard_masks_int32.npy").is_file()
        ]
    best: tuple[float, str, str, int, int, float, float] | None = None
    for cat, stem in candidates:
        try:
            case = CP_ROOT / cat / stem
            gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
            cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
            si = _load_sicle_mask(cat, stem)
            br_cp = _per_cell_br(gt, cp)
            br_si = _per_cell_br(gt, si)
            s_gid, c_gid = _best_zoom_gids(cat, stem, min_delta, min_area=0)
            ds = br_si[s_gid] - br_cp[s_gid]
            dc = br_si[c_gid] - br_cp[c_gid]
            score = abs(ds) + abs(dc)
            if best is None or score > best[0]:
                best = (score, cat, stem, s_gid, c_gid, ds, dc)
        except (FileNotFoundError, OSError, ValueError):
            continue
    if best is None:
        raise RuntimeError("No ROI found for zoom exemplars")
    _, cat, stem, s_gid, c_gid, ds, dc = best
    return cat, stem, int(s_gid), int(c_gid), float(ds), float(dc)


def _best_zoom_gids(
    category: str,
    stem: str,
    min_delta: float = 0.08,
    min_area: int = 1800,
) -> tuple[int, int]:
    """Pick cells with large BR gap and enough area for a readable zoom."""
    case = CP_ROOT / category / stem
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = _load_sicle_mask(category, stem)
    br_cp = _per_cell_br(gt, cp)
    br_si = _per_cell_br(gt, si)
    best_s, best_c = None, None
    best_s_score, best_c_score = -1.0, -1.0
    for gid in br_cp:
        if gid not in br_si:
            continue
        area = int((gt == gid).sum())
        if area < min_area:
            continue
        d = br_si[gid] - br_cp[gid]
        if d >= min_delta and d * area > best_s_score:
            best_s_score, best_s = d * area, gid
        if d <= -min_delta and (-d) * area > best_c_score:
            best_c_score, best_c = (-d) * area, gid
    if best_s is None or best_c is None:
        return _best_zoom_gids(category, stem, min_delta, min_area=0)
    return int(best_s), int(best_c)


def build_cell_zoom_overlay(
    category: str,
    stem: str,
    gid: int,
    tag: str,
    margin: int = 40,
) -> Path:
    """Single crop: GT (cyan) + Cellpose (yellow) + SICLE (green) for easy visual compare."""
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import bbox_of_mask, draw_contours, isolate_pred_for_gt
    from percell_boundary_recall import compute_boundary_recall

    case = CP_ROOT / category / stem
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = _load_sicle_mask(category, stem)

    m = gt == gid
    r0, r1, c0, c1 = bbox_of_mask(m)
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(gt.shape[0], r1 + margin), min(gt.shape[1], c1 + margin)
    o_c, gt_c = orig[r0:r1, c0:c1], gt[r0:r1, c0:c1]
    gt_iso = np.where(gt_c == gid, gid, 0)
    p_cp, _ = isolate_pred_for_gt(cp[r0:r1, c0:c1], gt_c, gid)
    p_si, _ = isolate_pred_for_gt(si[r0:r1, c0:c1], gt_c, gid)
    br_cp, _, _ = compute_boundary_recall(p_cp, gt_iso)
    br_si, _, _ = compute_boundary_recall(p_si, gt_iso)

    img = draw_contours(o_c, gt_iso, GT_CYAN, 1)
    img = draw_contours(img, p_cp, CP_YELLOW, 3)
    img = draw_contours(img, p_si, SICLE_GREEN, 3)

    th = 28
    canvas = np.full((img.shape[0] + th, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[th:, :] = img
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font_b, _ = _dejavu_fonts(11, 9)
    title = f"Cell {gid}  |  CP BR={br_cp:.2f}  |  SICLE BR={br_si:.2f}  |  Delta={br_si - br_cp:+.2f}"
    tw = _text_width(draw, title, font_b)
    draw.text((max(0, (img.shape[1] - tw) // 2), 6), title, fill=(0, 0, 0), font=font_b)

    out = PANELS / f"{category}_{stem}_cell{gid:03d}_overlay_{tag}.png"
    pil.save(out)
    return out


def build_cell_zoom(
    category: str,
    stem: str,
    gid: int,
    tag: str,
    margin: int = 22,
) -> Path:
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import bbox_of_mask, draw_contours, isolate_pred_for_gt
    from percell_boundary_recall import compute_boundary_recall

    case = CP_ROOT / category / stem
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = _load_sicle_mask(category, stem)

    m = gt == gid
    r0, r1, c0, c1 = bbox_of_mask(m)
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(gt.shape[0], r1 + margin), min(gt.shape[1], c1 + margin)
    o_c, gt_c = orig[r0:r1, c0:c1], gt[r0:r1, c0:c1]
    gt_iso = np.where(gt_c == gid, gid, 0)
    p_cp, _ = isolate_pred_for_gt(cp[r0:r1, c0:c1], gt_c, gid)
    p_si, _ = isolate_pred_for_gt(si[r0:r1, c0:c1], gt_c, gid)
    br_cp, _, _ = compute_boundary_recall(p_cp, gt_iso)
    br_si, _, _ = compute_boundary_recall(p_si, gt_iso)

    # Prediction first; GT red on top (easier to see under-segmentation vs. expert rim)
    img_gt = draw_contours(o_c, gt_iso, GT_RED, 2)
    img_cp = draw_contours(o_c, p_cp, CP_YELLOW, 2)
    img_cp = draw_contours(img_cp, gt_iso, GT_RED, 2)
    img_si = draw_contours(o_c, p_si, SICLE_GREEN, 2)
    img_si = draw_contours(img_si, gt_iso, GT_RED, 2)

    images = [img_gt, img_cp, img_si]
    title_specs = [
        ("GT", None),
        ("Cellpose + GT", f"BR = {br_cp:.2f}"),
        ("SICLE + GT", f"BR = {br_si:.2f}"),
    ]

    h, w = o_c.shape[:2]
    col_w = max(w, MIN_ZOOM_COL_W)
    gap = ZOOM_GAP
    th = ZOOM_TITLE_H
    canvas_w = 3 * col_w + 2 * gap
    canvas = np.full((h + th, canvas_w, 3), 255, dtype=np.uint8)
    x = 0
    for img in images:
        x_off = max(0, (col_w - w) // 2)
        canvas[th : th + h, x + x_off : x + x_off + w] = img
        x += col_w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    _draw_column_titles(draw, title_specs, col_w, gap, th)

    out = PANELS / f"{category}_{stem}_cell{gid:03d}_{tag}.png"
    pil.save(out)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="healthy")
    p.add_argument("--roi", default="healthy-18-roi2")
    p.add_argument("--export-all", action="store_true", help="demo ROI + exemplar panels for presentation")
    args = p.parse_args()

    if args.export_all:
        rois = [
            ("healthy", "healthy-18-roi2"),
            ("healthy", "healthy-24-roi1"),
            ("severe", "severe-03-roi2"),
        ]
        for cat, stem in rois:
            build_roi_panel(cat, stem)
            build_percell_winners(cat, stem)
            print(f"ROI panels: {cat}/{stem}")
        # zoom exemplars — 3-panel layout; ROI with largest |ΔBR| pair
        z_cat, z_stem, s_gid, c_gid, ds, dc = find_best_zoom_roi()
        build_cell_zoom(z_cat, z_stem, s_gid, "sicle_wins")
        build_cell_zoom(z_cat, z_stem, c_gid, "cellpose_wins")
        print(
            f"  zoom {z_cat}/{z_stem}: SICLE wins cell {s_gid} (d={ds:+.3f}), "
            f"Cellpose wins cell {c_gid} (d={dc:+.3f})"
        )
        build_cell_zoom("healthy", "healthy-24-roi1", 7, "sicle_wins_nf")
        print("Wrote panels to", PANELS)
        return 0

    build_roi_panel(args.category, args.roi)
    build_percell_winners(args.category, args.roi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
