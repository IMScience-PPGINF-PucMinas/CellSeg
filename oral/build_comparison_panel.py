#!/usr/bin/env python3
"""4-panel: original | Cellpose+GT | SICLE+GT | CP vs SICLE difference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, SINGLE_ROI_RUN

ROI = "healthy-18-roi2"
CASE = SINGLE_ROI_RUN / ROI
OUT = SINGLE_ROI_RUN / "comparison_panel_orig_cp_gt_sicle_gt_diff.png"

GT_CYAN = (0, 255, 255)
CP_YELLOW = (255, 255, 0)
SICLE_GREEN = (0, 255, 0)
GAP = 6
TITLE_H = 28


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    orig = np.asarray(Image.open(CASE / f"{ROI}.png").convert("RGB"))
    gt = np.load(CASE / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    cp = np.load(CASE / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
    si = np.load(CASE / "sicle" / "merged_percell_sicle_masks_int32.npy").astype(np.int32)

    h, w = orig.shape[:2]
    for arr in (gt, cp, si):
        if arr.shape != (h, w):
            raise SystemExit(f"shape mismatch: {arr.shape} vs {(h, w)}")

    panel_cp_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panel_cp_gt = draw_contours(panel_cp_gt, cp, CP_YELLOW, thickness=1)

    panel_si_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panel_si_gt = draw_contours(panel_si_gt, si, SICLE_GREEN, thickness=1)

    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[:] = (orig.astype(np.float32) * 0.25).astype(np.uint8)
    fg_c = cp > 0
    fg_s = si > 0
    only_cp = fg_c & ~fg_s
    only_si = fg_s & ~fg_c
    label_mismatch = fg_c & fg_s & (cp != si)
    agree = fg_c & fg_s & (cp == si)
    diff[only_cp] = np.array([255, 200, 0], dtype=np.uint8)
    diff[only_si] = np.array([0, 220, 0], dtype=np.uint8)
    diff[label_mismatch] = np.array([255, 0, 255], dtype=np.uint8)
    diff[agree] = np.array([180, 180, 180], dtype=np.uint8)

    panels = [
        ("Original", orig),
        ("Cellpose + GT", panel_cp_gt),
        ("SICLE + GT", panel_si_gt),
        ("CP vs SICLE", diff),
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
    ly = TITLE_H + h - 72
    for text, color in [
        ("amarelo: so Cellpose", (255, 200, 0)),
        ("verde: so SICLE", (0, 220, 0)),
        ("magenta: ambos, ID diff", (255, 0, 255)),
        ("cinza: mesmo ID", (120, 120, 120)),
    ]:
        draw.text((lx, ly), text, fill=color, font=font)
        ly += 16

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pil.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
