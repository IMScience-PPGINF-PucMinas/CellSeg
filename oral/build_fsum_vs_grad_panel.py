#!/usr/bin/env python3
"""Visual panel: fsum+minsc vs gradvmaxmul+minsc + per-cell BR winners."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, RUNS

OUT_ROOT = RUNS / "path_cost_benchmark"
GT_CYAN = (0, 255, 255)
FSUM_COL = (0, 180, 255)
GRAD_COL = (0, 200, 0)
WIN_FSUM = (255, 80, 80)
WIN_GRAD = (80, 255, 80)


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


def _highlight_cells(
    base: np.ndarray,
    gt: np.ndarray,
    gids: list[int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    from percell_boundary_recall import draw_contours

    img = base.copy()
    for gid in gids:
        m = np.where(gt == gid, gid, 0).astype(np.int32)
        img = draw_contours(img, m, color, thickness=thickness)
    return img


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="healthy")
    p.add_argument("--roi", default="healthy-18-roi2")
    p.add_argument("--min-delta", type=float, default=0.08, help="min |BR_grad - BR_fsum| to highlight")
    args = p.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    case = OUT_ROOT / args.category / args.roi
    stem = args.roi
    orig = np.asarray(Image.open(case / f"{stem}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    pr_f = np.load(case / "fsum_minsc_nolin" / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
    pr_g = np.load(case / "gradvmaxmul_minsc_nolin" / "merged_percell_sicle_masks_int32.npy").astype(np.int32)

    br_f = _per_cell_br(gt, pr_f)
    br_g = _per_cell_br(gt, pr_g)
    mean_f = float(np.mean(list(br_f.values()))) if br_f else float("nan")
    mean_g = float(np.mean(list(br_g.values()))) if br_g else float("nan")

    win_f, win_g, tie = [], [], []
    for gid in br_f:
        if gid not in br_g:
            continue
        d = br_g[gid] - br_f[gid]
        if d >= args.min_delta:
            win_g.append(gid)
        elif d <= -args.min_delta:
            win_f.append(gid)
        else:
            tie.append(gid)

    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    pan_f = draw_contours(base_gt.copy(), pr_f, FSUM_COL, thickness=1)
    pan_g = draw_contours(base_gt.copy(), pr_g, GRAD_COL, thickness=1)
    pan_diff = draw_contours(base_gt.copy(), gt, GT_CYAN, thickness=1)
    pan_diff = _highlight_cells(pan_diff, gt, win_g[:8], WIN_GRAD, thickness=2)
    pan_diff = _highlight_cells(pan_diff, gt, win_f[:8], WIN_FSUM, thickness=2)

    h, w = orig.shape[:2]
    gap, title_h, foot_h = 6, 28, 36
    labels = [
        ("Original", orig),
        (f"GT (ciano)", base_gt),
        (f"fsum+minsc  BR={mean_f:.3f}", pan_f),
        (f"gradvmaxmul+minsc  BR={mean_g:.3f}", pan_g),
        (f"Diferença células (verde=grad melhor, vermelho=fsum)", pan_diff),
    ]
    total_w = len(labels) * w + (len(labels) - 1) * gap
    canvas = np.full((h + title_h + foot_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in labels:
        canvas[title_h : title_h + h, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = font_sm = ImageFont.load_default()

    x = 0
    for title, _ in labels:
        draw.text((x + 4, 6), title, fill=(0, 0, 0), font=font)
        x += w + gap

    foot = (
        f"Mesmo minsc, alpha=2, sem Otsu. gradv ganha em {len(win_g)} células (Δ≥{args.min_delta}), "
        f"fsum em {len(win_f)}, empate ~{len(tie)}. Ex.: grad melhor células {win_g[:5]}; fsum {win_f[:5]}"
    )
    draw.text((4, h + title_h + 8), foot[: total_w // 6], fill=(40, 40, 40), font=font_sm)

    out = OUT_ROOT / "panels" / f"{args.category}_{args.roi}_fsum_vs_gradvminsc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"Wrote {out}")

    diff_out = OUT_ROOT / "panels" / f"{args.category}_{args.roi}_fsum_vs_grad_diff.png"
    Image.fromarray(pan_diff).save(diff_out)
    print(f"Wrote {diff_out}")

    print(f"  macro BR fsum={mean_f:.4f} gradvmaxmul={mean_g:.4f} delta={mean_g - mean_f:+.4f}")
    print(f"  cells grad better: {len(win_g)}, fsum better: {len(win_f)}")
    return 0


def export_fsum_diff_only(category: str, roi: str, dest: Path) -> None:
    """Last panel only (per-cell winners highlight) for Beamer slide 19."""
    src = OUT_ROOT / "panels" / f"{category}_{roi}_fsum_vs_grad_diff.png"
    if not src.is_file():
        import subprocess

        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--category", category, "--roi", roi],
            check=True,
        )
    dest.mkdir(parents=True, exist_ok=True)
    from shutil import copy2

    copy2(src, dest / "fsum_vs_grad_diff.png")
    print(f"  fsum_vs_grad_diff.png")


if __name__ == "__main__":
    raise SystemExit(main())
