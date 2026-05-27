#!/usr/bin/env python3
"""4-panel: original | SICLE CLI default+GT | SICLE best blur05+GT | default vs best diff."""
from __future__ import annotations

import sys

import numpy as np

from _paths import PIPE, SINGLE_ROI_RUN

ROI = "healthy-18-roi2"
CASE = SINGLE_ROI_RUN / ROI
OUT = SINGLE_ROI_RUN / "comparison_panel_sicle_default_vs_best.png"

GT_CYAN = (0, 255, 255)
DEFAULT_ORANGE = (255, 140, 0)
BEST_GREEN = (0, 255, 0)
GAP = 6
TITLE_H = 28


def main() -> int:
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    orig = np.asarray(Image.open(CASE / f"{ROI}.png").convert("RGB"))
    gt = np.load(CASE / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    si_def = np.load(CASE / "sicle_cli_default" / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
    si_best = np.load(CASE / "sicle_best_blur05" / "merged_percell_sicle_masks_int32.npy").astype(np.int32)

    h, w = orig.shape[:2]
    for name, arr in (("gt", gt), ("default", si_def), ("best", si_best)):
        if arr.shape != (h, w):
            raise SystemExit(f"{name} shape mismatch: {arr.shape} vs {(h, w)}")

    panel_def_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panel_def_gt = draw_contours(panel_def_gt, si_def, DEFAULT_ORANGE, thickness=1)

    panel_best_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panel_best_gt = draw_contours(panel_best_gt, si_best, BEST_GREEN, thickness=1)

    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[:] = (orig.astype(np.float32) * 0.25).astype(np.uint8)
    fg_d = si_def > 0
    fg_b = si_best > 0
    only_def = fg_d & ~fg_b
    only_best = fg_b & ~fg_d
    label_mismatch = fg_d & fg_b & (si_def != si_best)
    agree = fg_d & fg_b & (si_def == si_best)
    diff[only_def] = np.array([255, 140, 0], dtype=np.uint8)
    diff[only_best] = np.array([0, 220, 0], dtype=np.uint8)
    diff[label_mismatch] = np.array([255, 0, 255], dtype=np.uint8)
    diff[agree] = np.array([180, 180, 180], dtype=np.uint8)

    panels = [
        ("Original", orig),
        ("SICLE default + GT", panel_def_gt),
        ("SICLE best (blur05) + GT", panel_best_gt),
        ("default vs best", diff),
    ]

    total_w = 4 * w + 3 * GAP
    canvas = np.full((h + TITLE_H, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _title, img in panels:
        canvas[TITLE_H:, x : x + w] = img
        x += w + GAP

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 4, 4), title, fill=(0, 0, 0), font=font)
        x += w + GAP

    lx = 3 * (w + GAP) + 8
    ly = TITLE_H + h - 88
    for text, color in [
        ("ciano: GT", GT_CYAN),
        ("laranja: SICLE default", DEFAULT_ORANGE),
        ("verde: SICLE best", BEST_GREEN),
        ("--- diff ---", (0, 0, 0)),
        ("laranja: so default", (255, 140, 0)),
        ("verde: so best", (0, 220, 0)),
        ("magenta: ambos, ID diff", (255, 0, 255)),
        ("cinza: mesmo ID", (120, 120, 120)),
    ]:
        draw.text((lx, ly), text, fill=color, font=font)
        ly += 14

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pil.save(OUT)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
