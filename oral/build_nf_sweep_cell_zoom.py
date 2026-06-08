#!/usr/bin/env python3
"""Zoom one GT cell: Nf=2 vs Nf=50 masks side by side."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, RUNS

NF_ROOT = RUNS / "nf_sweep_full"
BENCH = RUNS / "postprocess_ablation_full"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="healthy")
    p.add_argument("--roi", default="healthy-24-roi1")
    p.add_argument("--cell-id", type=int, default=0, help="0 = auto-pick cell with largest BR drop")
    args = p.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import (
        bbox_of_mask,
        compute_boundary_recall,
        draw_contours,
        isolate_pred_for_gt,
    )

    cat, stem = args.category, args.roi
    case_bench = BENCH / cat / stem
    case = NF_ROOT / cat / stem
    orig = np.asarray(Image.open(case_bench / f"{stem}.png").convert("RGB"))
    gt = np.load(case_bench / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)

    def load_pr(nf: int) -> np.ndarray:
        n0 = max(200, nf + 20)
        p = case / f"nf{nf}_n0{n0}_raw" / "merged_percell_sicle_masks_int32.npy"
        if not p.is_file():
            p = next(case.glob(f"nf{nf}_n0*_raw")).joinpath("merged_percell_sicle_masks_int32.npy")
        return np.load(p).astype(np.int32)

    pr2, pr50 = load_pr(2), load_pr(50)
    margin = 12
    best_gid, best_drop = 0, -1.0
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        if args.cell_id > 0 and gid != args.cell_id:
            continue
        m = gt == gid
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(gt.shape[0], r1 + margin), min(gt.shape[1], c1 + margin)
        gt_iso = np.where(gt[r0:r1, c0:c1] == gid, gid, 0)
        p2, _ = isolate_pred_for_gt(pr2[r0:r1, c0:c1], gt[r0:r1, c0:c1], gid)
        p50, _ = isolate_pred_for_gt(pr50[r0:r1, c0:c1], gt[r0:r1, c0:c1], gid)
        br2, _, _ = compute_boundary_recall(p2, gt_iso)
        br50, _, _ = compute_boundary_recall(p50, gt_iso)
        drop = br2 - br50
        if args.cell_id > 0 or drop > best_drop:
            best_drop, best_gid = drop, gid
            crop_box = (r0, r1, c0, c1)

    r0, r1, c0, c1 = crop_box
    gt_c = gt[r0:r1, c0:c1]
    o_c = orig[r0:r1, c0:c1]
    gt_iso = np.where(gt_c == best_gid, best_gid, 0)
    p2, _ = isolate_pred_for_gt(pr2[r0:r1, c0:c1], gt_c, best_gid)
    p50, _ = isolate_pred_for_gt(pr50[r0:r1, c0:c1], gt_c, best_gid)
    br2, _, _ = compute_boundary_recall(p2, gt_iso)
    br50, _, _ = compute_boundary_recall(p50, gt_iso)

    panels = [
        ("GT", draw_contours(o_c, gt_iso, (0, 255, 255), 1)),
        (f"Nf=2 BR={br2:.2f}", draw_contours(o_c, p2, (0, 220, 0), 2)),
        (f"Nf=50 BR={br50:.2f}", draw_contours(o_c, p50, (255, 80, 80), 2)),
    ]
    h, w = o_c.shape[:2]
    gap, th = 4, 22
    canvas = np.full((h + th, 3 * w + 2 * gap, 3), 255, dtype=np.uint8)
    x = 0
    for title, img in panels:
        canvas[th:, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for title, _ in panels:
        draw.text((x + 2, 4), title, fill=(0, 0, 0), font=font)
        x += w + gap

    out = NF_ROOT / "panels" / f"{cat}_{stem}_cell{best_gid:03d}_nf2_vs_nf50.png"
    pil.save(out)
    print(f"Wrote {out} (cell {best_gid}, BR2={br2:.3f}, BR50={br50:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
