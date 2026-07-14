#!/usr/bin/env python3
"""Per-cell BR winner panels: Cellpose seeds vs. iDISF refinement."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS

CP_ROOT = RUNS / "postprocess_ablation_full"
IDISF_ROOT = RUNS / "percell_idisf_full"
PANELS = RUNS / "cellpose_vs_idisf" / "panels"
PAPER_FIGS = REPO / "figs" / "paper"

GT_CYAN = (0, 255, 255)
CP_RED = (255, 70, 70)
IDISF_GREEN = (0, 220, 0)
WIN_IDISF = (80, 255, 120)
WIN_CP = (255, 180, 0)


def _load_idisf_mask(category: str, stem: str) -> np.ndarray:
    case = IDISF_ROOT / category / stem
    for sub in ("idisf_unconquerable", "idisf_exclude_other"):
        path = case / sub / "merged_percell_idisf_masks_int32.npy"
        if path.is_file() and path.stat().st_size > 0:
            return np.load(path).astype(np.int32)
    raise FileNotFoundError(f"No iDISF mask for {category}/{stem}")


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


def build_percell_winners(
    category: str,
    stem: str,
    *,
    min_delta: float = 0.08,
    max_cells: int = 8,
    paper_name: str | None = None,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    case = CP_ROOT / category / stem
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    idisf = _load_idisf_mask(category, stem)

    br_cp = _per_cell_br(gt, cp)
    br_id = _per_cell_br(gt, idisf)
    win_i, win_c = [], []
    for gid in br_cp:
        if gid not in br_id:
            continue
        delta = br_id[gid] - br_cp[gid]
        if delta >= min_delta:
            win_i.append((delta, gid))
        elif delta <= -min_delta:
            win_c.append((delta, gid))
    win_i.sort(reverse=True)
    win_c.sort()
    gids_i = [g for _, g in win_i[:max_cells]]
    gids_c = [g for _, g in win_c[:max_cells]]

    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    pan_win = draw_contours(base_gt.copy(), gt, GT_CYAN, 1)
    pan_win = _highlight_cells(pan_win, gt, gids_i, WIN_IDISF, 2)
    pan_win = _highlight_cells(pan_win, gt, gids_c, WIN_CP, 2)

    panels = [
        ("GT", draw_contours(orig, gt, GT_CYAN, thickness=1)),
        ("Cellpose seeds", draw_contours(base_gt.copy(), cp, CP_RED, 1)),
        ("iDISF", draw_contours(base_gt.copy(), idisf, IDISF_GREEN, 1)),
        ("Per-cell BR winners", pan_win),
    ]

    h, w = panels[0][1].shape[:2]
    gap, th, foot = 6, 26, 48
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + th + foot, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[th : th + h, x : x + w] = img
        x += w + gap

    br_cp_m = float(np.mean(list(br_cp.values()))) if br_cp else 0.0
    br_id_m = float(np.mean(list(br_id.values()))) if br_id else 0.0

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
        f"{category}/{stem}: green=iDISF wins ({len(gids_i)} cells), orange=Cellpose wins ({len(gids_c)} cells)",
        f"Patch BR macro: seeds {br_cp_m:.3f} -> iDISF {br_id_m:.3f} (Delta {br_id_m - br_cp_m:+.3f})",
    ]
    y = h + th + 4
    for line in lines:
        draw.text((4, y), line, fill=(30, 30, 30), font=font_sm)
        y += 12

    out = PANELS / f"{category}_{stem}_percell_winners.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)

    if paper_name:
        PAPER_FIGS.mkdir(parents=True, exist_ok=True)
        paper_path = PAPER_FIGS / paper_name
        pil.save(paper_path)
        print(f"Wrote {paper_path}")

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="severe")
    p.add_argument("--roi", default="severe-10-roi2")
    p.add_argument("--paper-name", default=None)
    p.add_argument("--export-paper-defaults", action="store_true")
    args = p.parse_args()

    if args.export_paper_defaults:
        build_percell_winners(
            "healthy",
            "healthy-18-roi2",
            paper_name="cp_vs_sicle_healthy18.png",
        )
        build_percell_winners(
            "severe",
            "severe-10-roi2",
            paper_name="cp_vs_idisf_severe10.png",
        )
        return 0

    out = build_percell_winners(
        args.category,
        args.roi,
        paper_name=args.paper_name,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
