#!/usr/bin/env python3
"""5-panel: original | GT | minsc | maxsc | size | spread (gradvmaxmul fixed)."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _paths import PIPE, RUNS

OUT_ROOT = RUNS / "path_cost_benchmark"
CONN = "gradvmaxmul"
CRITS = ("minsc", "maxsc", "size", "spread")
COLORS = {
    "minsc": (0, 255, 0),
    "maxsc": (255, 140, 0),
    "size": (255, 0, 255),
    "spread": (0, 180, 255),
}
GT_CYAN = (0, 255, 255)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--roi", required=True)
    args = p.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    case = OUT_ROOT / args.category / args.roi
    stem = args.roi
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    h, w = orig.shape[:2]

    panels: list[tuple[str, np.ndarray]] = [("Original", orig)]
    panels.append(("GT", draw_contours(orig, gt, GT_CYAN, thickness=1)))

    for crit in CRITS:
        cid = f"{CONN}_{crit}"
        pr = np.load(case / cid / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
        img = draw_contours(orig, gt, GT_CYAN, thickness=1)
        img = draw_contours(img, pr, COLORS[crit], thickness=1)
        panels.append((crit, img))

    gap, title_h = 4, 26
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + title_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[title_h:, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 2, 4), title, fill=(0, 0, 0), font=font)
        x += w + gap

    out = OUT_ROOT / "panels" / f"{args.category}_{args.roi}_criteria_{CONN}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
