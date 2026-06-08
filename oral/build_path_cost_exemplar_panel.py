#!/usr/bin/env python3
"""5-panel: original | GT | fmax | fsum | gradvmaxmul for one ROI."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from _paths import PIPE, RUNS

OUT_ROOT = RUNS / "path_cost_benchmark"
CONFIGS = ("fmax_minsc", "fsum_maxsc", "gradvmaxmul_minsc")
COLORS = {
    "fmax_minsc": (255, 140, 0),
    "fsum_maxsc": (0, 180, 255),
    "gradvmaxmul_minsc": (0, 255, 0),
}
GT_CYAN = (0, 255, 255)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--roi", required=True)
    p.add_argument("-o", type=str, default="")
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
    panel_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panels.append(("GT", panel_gt))

    metrics = {}
    for cid in CONFIGS:
        pr = np.load(case / cid / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
        img = draw_contours(orig, gt, GT_CYAN, thickness=1)
        img = draw_contours(img, pr, COLORS[cid], thickness=1)
        panels.append((cid.replace("_", " "), img))

    gap, title_h = 6, 28
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + title_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[title_h:, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 4, 4), title, fill=(0, 0, 0), font=font)
        x += w + gap

    out = Path(args.o) if args.o else OUT_ROOT / "panels" / f"{args.category}_{args.roi}_path_costs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
