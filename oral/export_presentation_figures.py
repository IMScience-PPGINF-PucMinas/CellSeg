#!/usr/bin/env python3
"""Export Oral Epithelium figures for the UVA Beamer presentation."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from _paths import (
    GT_COLORED,
    IMAGES_ORIGINAL,
    REPO,
    RUNS,
)

OUT = REPO / "presentation_uva" / "2025-imscience-model" / "presentations" / "oral-cellseg" / "images"
DEMO_ROI = ("healthy", "healthy-18-roi2")
SEVERE_ROI = ("severe", "severe-03-roi2")
BENCH = RUNS / "path_cost_benchmark"
PANELS = BENCH / "panels"
CP_VS = RUNS / "cellpose_vs_sicle" / "panels"
NF_PANELS = RUNS / "nf_sweep_full" / "panels"

# Mosaic: representative healthy + severe patches
MOSAIC = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
    ("severe", "severe-01-roi4"),
    ("severe", "severe-02-roi1"),
]

# Cyan is reserved for ground-truth contours only (Beamer slides 9--13).
GT_CYAN = (0, 255, 255)
OURS_GREEN = (0, 255, 0)
LIT_FMAX_ORANGE = (255, 140, 0)
LIT_FSUM_MAGENTA = (255, 0, 255)
CRIT_MAXSC_ORANGE = (255, 140, 0)
CRIT_SIZE_PURPLE = (180, 0, 255)
CRIT_SPREAD_GOLD = (255, 200, 0)


def _open_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save(arr: np.ndarray, path: Path, scale: float = 1.0) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if scale != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    img.save(path, optimize=True)


def _original_path(cat: str, roi: str) -> Path:
    return IMAGES_ORIGINAL / cat / f"{roi}.tif"


def _gt_colored_path(cat: str, roi: str) -> Path:
    return GT_COLORED / cat / f"{roi}.png"


def _export_roi_bundle(cat: str, roi: str, prefix: str) -> None:
    orig = _open_rgb(_original_path(cat, roi))
    gt_col = _open_rgb(_gt_colored_path(cat, roi))
    _save(orig, OUT / f"{prefix}_original.png", scale=2.0)

    _save(gt_col, OUT / f"{prefix}_gt_colored.png", scale=2.0)

    # Translucent GT on H&E
    fg = gt_col.max(axis=2) > 8
    blend = orig.astype(np.float32).copy()
    blend[fg] = 0.55 * orig[fg] + 0.45 * gt_col[fg]
    _save(np.clip(blend, 0, 255).astype(np.uint8), OUT / f"{prefix}_gt_overlay.png", scale=2.0)

    run = BENCH / cat / roi
    cp_rgb = run / "cp_flow" / "step04_masks_rgb.png"
    if cp_rgb.is_file():
        shutil.copy2(cp_rgb, OUT / f"{prefix}_cellpose.png")

    sicle_cfg = run / "gradvmaxmul_minsc_nolin"
    if not sicle_cfg.is_dir():
        sicle_cfg = run / "gradvmaxmul_minsc"
    for name in ("merged_percell_sicle_overlay.png", "merged_percell_sicle_masks_rgb.png"):
        src = sicle_cfg / name
        if src.is_file():
            shutil.copy2(src, OUT / f"{prefix}_sicle_overlay.png")
            break


def _export_mosaic() -> None:
    panels = [
        (f"{roi}\n({cat})", _open_rgb(_original_path(cat, roi)))
        for cat, roi in MOSAIC
    ]
    _grid_figure(panels, ncols=2, out_name="dataset_mosaic_tall.png", cell_scale=4.0)


def _sicle_config(case: Path) -> Path:
    for name in ("gradvmaxmul_minsc_nolin", "gradvmaxmul_minsc"):
        p = case / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No gradvmaxmul+minsc run under {case}")


def _export_best_large(cat: str, roi: str) -> None:
    """Single large figure: GT (cyan) + best SICLE (green)."""
    sys.path.insert(0, str(REPO / "pipeline"))
    from percell_boundary_recall import draw_contours

    case = BENCH / cat / roi
    orig = _open_rgb(case / f"{roi}.png")
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    pred = np.load(_sicle_config(case) / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
    img = draw_contours(orig, gt, GT_CYAN, thickness=2)
    img = draw_contours(img, pred, OURS_GREEN, thickness=2)
    _save(img, OUT / "best_result_large.png", scale=5.0)
    overlay = _sicle_config(case) / "merged_percell_sicle_overlay.png"
    if overlay.is_file():
        _save(_open_rgb(overlay), OUT / "best_result_overlay.png", scale=5.0)


def _br_summary_panel(w: int, h: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = np.full((h, w, 3), 255, dtype=np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = font_sm = ImageFont.load_default()
    lines = [
        ("Mean BR", font),
        ("0.338  ours", font_sm),
        ("0.330  fmax", font_sm),
        ("0.258  fsum", font_sm),
    ]
    y = h // 2 - 36
    for text, fnt in lines:
        draw.text((12, y), text, fill=(30, 30, 30), font=fnt)
        y += 22
    return np.asarray(pil)


def _grid_figure(
    panels: list[tuple[str, np.ndarray]],
    ncols: int,
    out_name: str,
    cell_scale: float = 3.5,
    title_h: int = 34,
    gap: int = 10,
) -> None:
    """Arrange panels in a grid (taller aspect ratio for Beamer slides)."""
    from PIL import Image, ImageDraw, ImageFont

    if not panels:
        return
    h0, w0 = panels[0][1].shape[:2]
    cell = Image.fromarray(panels[0][1]).resize(
        (max(1, int(w0 * cell_scale)), max(1, int(h0 * cell_scale))),
        Image.Resampling.LANCZOS,
    )
    cw, ch = cell.size
    nrows = (len(panels) + ncols - 1) // ncols
    canvas_w = ncols * cw + (ncols - 1) * gap
    canvas_h = nrows * (ch + title_h) + (nrows - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, ncols)
        x0 = c * (cw + gap)
        y0 = r * (ch + title_h + gap)
        cell_img = Image.fromarray(img).resize((cw, ch), Image.Resampling.LANCZOS)
        pil.paste(cell_img, (x0, y0 + title_h))
        draw.text((x0 + 6, y0 + 6), title, fill=(20, 20, 20), font=font)
    _save(np.asarray(pil), OUT / out_name, scale=1.0)


def _export_presentation_grids(cat: str, roi: str) -> None:
    """Tall grid layouts for slides (avoid ultra-wide benchmark strips)."""
    sys.path.insert(0, str(REPO / "pipeline"))
    from percell_boundary_recall import draw_contours

    case = BENCH / cat / roi
    stem = roi
    orig = _open_rgb(case / f"{stem}.png")
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)

    def _pred(cid: str) -> np.ndarray:
        for sub in (cid, f"{cid}_nolin"):
            p = case / sub / "merged_percell_sicle_masks_int32.npy"
            if p.is_file():
                return np.load(p).astype(np.int32)
        raise FileNotFoundError(f"No masks for {cid} under {case}")

    def _pred_only(pred: np.ndarray, color: tuple[int, int, int], thick: int = 2) -> np.ndarray:
        return draw_contours(orig, pred, color, thickness=thick)

    def _overlay(pred: np.ndarray, color: tuple[int, int, int], thick: int = 2) -> np.ndarray:
        img = draw_contours(orig, gt, GT_CYAN, thickness=1)
        return draw_contours(img, pred, color, thickness=thick)

    criteria_panels: list[tuple[str, np.ndarray]] = [
        ("Original", orig),
        ("GT", draw_contours(orig, gt, GT_CYAN, thickness=2)),
        ("minsc (ours)", _overlay(_pred("gradvmaxmul_minsc"), OURS_GREEN)),
        ("maxsc", _overlay(_pred("gradvmaxmul_maxsc"), CRIT_MAXSC_ORANGE)),
        ("size", _overlay(_pred("gradvmaxmul_size"), CRIT_SIZE_PURPLE)),
        ("spread", _overlay(_pred("gradvmaxmul_spread"), CRIT_SPREAD_GOLD)),
    ]
    _grid_figure(criteria_panels, ncols=3, out_name="panel_criteria_3x2.png", cell_scale=4.0)

    four_only = [
        ("minsc (ours)", criteria_panels[2][1]),
        ("maxsc", criteria_panels[3][1]),
        ("size", criteria_panels[4][1]),
        ("spread", criteria_panels[5][1]),
    ]
    _grid_figure(four_only, ncols=2, out_name="criterion_four_tall.png", cell_scale=4.5)

    crit_names = (
        "crit_orig",
        "crit_gt",
        "crit_minsc",
        "crit_maxsc",
        "crit_size",
        "crit_spread",
    )
    for out_stem, (_title, img) in zip(crit_names, criteria_panels):
        _save(img, OUT / f"{out_stem}.png", scale=4.5)

    path_panels: list[tuple[str, np.ndarray]] = [
        ("Original", orig),
        ("GT", draw_contours(orig, gt, GT_CYAN, thickness=2)),
        ("fmax + minsc", _pred_only(_pred("fmax_minsc"), LIT_FMAX_ORANGE)),
        ("fsum + maxsc", _pred_only(_pred("fsum_maxsc"), LIT_FSUM_MAGENTA)),
        ("gradvmaxmul + minsc (ours)", _pred_only(_pred("gradvmaxmul_minsc"), OURS_GREEN)),
    ]
    h0, w0 = orig.shape[:2]
    path_panels.append(("Mean BR", _br_summary_panel(w0, h0)))
    _grid_figure(path_panels, ncols=3, out_name="panel_path_costs_3x2.png", cell_scale=5.2)

    # Individual cells for LaTeX 2x3 layout (max size per box on slide)
    names = (
        "path_orig",
        "path_gt",
        "path_fmax",
        "path_fsum",
        "path_gradvmaxmul",
        "path_br_summary",
    )
    for out_stem, (_title, img) in zip(names, path_panels):
        _save(img, OUT / f"{out_stem}.png", scale=4.5)


def _export_comparison_4panel() -> None:
    """Original | CP+GT | SICLE+GT | diff for demo ROI (from benchmark run)."""
    sys.path.insert(0, str(REPO / "pipeline"))
    from percell_boundary_recall import draw_contours

    cat, roi = DEMO_ROI
    case = BENCH / cat / roi
    orig = _open_rgb(case / f"{roi}.png")
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    cfg = case / "gradvmaxmul_minsc_nolin"
    if not cfg.is_dir():
        cfg = case / "gradvmaxmul_minsc"
    si = np.load(cfg / "merged_percell_sicle_masks_int32.npy").astype(np.int32)

    CP_YELLOW = (255, 255, 0)

    p_cp = draw_contours(orig, gt, GT_CYAN, thickness=1)
    p_cp = draw_contours(p_cp, cp, CP_YELLOW, thickness=1)
    p_si = draw_contours(orig, gt, GT_CYAN, thickness=1)
    p_si = draw_contours(p_si, si, OURS_GREEN, thickness=1)

    h, w = orig.shape[:2]
    gap = 6
    row = np.full((h, 4 * w + 3 * gap, 3), 255, dtype=np.uint8)
    for i, img in enumerate((orig, p_cp, p_si)):
        row[:, i * (w + gap) : i * (w + gap) + w] = img
    # diff panel
    diff = (orig.astype(np.float32) * 0.3).astype(np.uint8)
    fg_c, fg_s = cp > 0, si > 0
    diff[fg_c & ~fg_s] = (255, 200, 0)
    diff[fg_s & ~fg_c] = (0, 220, 0)
    diff[fg_c & fg_s & (cp != si)] = (255, 0, 255)
    row[:, 3 * (w + gap) : 3 * (w + gap) + w] = diff

    _save(row, OUT / "pipeline_comparison_row.png", scale=2.0)


def _copy_panel(src: Path, dst_name: str) -> None:
    if src.is_file():
        shutil.copy2(src, OUT / dst_name)
        print(f"  {dst_name}")


def _export_cp_vs_sicle_2x2() -> None:
    """2x2 tiles for slide 6 (Act IV preview)."""
    sys.path.insert(0, str(REPO / "oral"))
    from build_cellpose_vs_sicle_panels import export_roi_tiles

    export_roi_tiles("healthy", "healthy-18-roi2", OUT)


def _export_cellpose_and_diagnosis() -> None:
    """Copy benchmark panels for Cellpose comparison + per-cell + Nf sweep slides."""
    print("  cp2_* 2x2 tiles:")
    _export_cp_vs_sicle_2x2()
    _copy_panel(CP_VS / "healthy_healthy-18-roi2_cellpose_vs_sicle.png", "cp_vs_sicle_healthy18.png")
    _copy_panel(CP_VS / "healthy_healthy-18-roi2_percell_winners.png", "percell_winners_healthy18.png")
    from build_cellpose_vs_sicle_panels import export_percell_highlight

    print("  percell highlight:")
    export_percell_highlight("healthy", "healthy-18-roi2", OUT)
    from build_cellpose_vs_sicle_panels import _best_zoom_gids, build_cell_zoom

    # Slide 17 exemplar (alternate ROI; strong |ΔBR| on both sides)
    z_cat, z_stem = "severe", "severe-10-roi2"
    s_gid, c_gid = _best_zoom_gids(z_cat, z_stem, min_area=0)
    build_cell_zoom(z_cat, z_stem, s_gid, "sicle_wins")
    build_cell_zoom(z_cat, z_stem, c_gid, "cellpose_wins")
    print(f"  zoom {z_cat}/{z_stem}: cells {s_gid} (SICLE), {c_gid} (Cellpose)")
    prefix = f"{z_cat}_{z_stem}"
    for tag, dst in (
        ("sicle_wins", "cell_zoom_sicle_wins.png"),
        ("cellpose_wins", "cell_zoom_cellpose_wins.png"),
    ):
        hits = sorted(CP_VS.glob(f"{prefix}_cell*_{tag}.png"))
        if hits:
            _copy_panel(hits[-1], dst)
    (OUT / "cell_zoom_roi.txt").write_text(f"{z_cat}/{z_stem}\n", encoding="utf-8")
    from build_nf_sweep_panel import export_nf_sweep_3x3

    print("  nf 3x3 tiles:")
    export_nf_sweep_3x3("healthy", "healthy-24-roi1", OUT)
    from build_fsum_vs_grad_panel import export_fsum_diff_only

    print("  fsum diff only:")
    export_fsum_diff_only("healthy", "healthy-18-roi2", OUT)
    # optional second ROI
    _copy_panel(CP_VS / "healthy_healthy-24-roi1_cellpose_vs_sicle.png", "cp_vs_sicle_healthy24.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _export_mosaic()
    _export_roi_bundle(*DEMO_ROI, "demo")
    _export_best_large(*DEMO_ROI)
    _export_presentation_grids(*DEMO_ROI)
    _export_comparison_4panel()
    print("Cellpose / diagnosis figures:")
    _export_cellpose_and_diagnosis()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
