#!/usr/bin/env python3
"""Compare SICLE CLI default vs blur05 best on the single-ROI run (metrics CSV)."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, SINGLE_ROI_RUN

ROI = "healthy-18-roi2"
CASE = SINGLE_ROI_RUN / ROI


def mean_br_strict(gt: np.ndarray, pr: np.ndarray, margin: int = 8) -> float:
    from percell_boundary_recall import (
        bbox_of_mask,
        compute_boundary_recall,
        isolate_pred_for_gt,
    )

    gt = np.asarray(gt, dtype=np.int32)
    pr = np.asarray(pr, dtype=np.int32)
    h, w = gt.shape
    vals: list[float] = []
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
        vals.append(float(br))
    return float(np.mean(vals)) if vals else float("nan")


def run_sicle(out_name: str, extra_args: list[str]) -> None:
    import os

    cp_dir = CASE / "cp_flow"
    out_dir = CASE / out_name
    img = CASE / f"{ROI}.png"
    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    subprocess.run(
        [sys.executable, str(PIPE / "percell_sicle_cellprob_pipeline.py"), "--from-dir", str(cp_dir), "-o", str(out_dir), "--image", str(img), *extra_args],
        cwd=str(REPO),
        env=env,
        check=True,
    )


def main() -> int:
    sys.path.insert(0, str(PIPE))
    from evaluate_instances import METRIC_KEYS, evaluate_pair

    gt_path = CASE / "gt" / "gold_standard_masks_int32.npy"
    if not (CASE / "cp_flow").is_dir():
        raise SystemExit(f"Run single-ROI test first (missing {CASE / 'cp_flow'})")

    best_args = [
        "--sicle-conn-opt", "gradvmaxmul", "--sicle-crit-opt", "minsc",
        "--sicle-alpha", "2.0", "--sicle-nf", "2", "--sicle-n0", "200",
        "--sicle-irreg", "0", "--sicle-adhr", "1", "--sicle-max-iters", "7",
        "--saliency-threshold", "0.3", "--saliency-blur-sigma", "0.5",
        "--margin", "4", "--min-cell-area", "128",
        "--disable-and-merge", "--and-unless-round",
        "--min-fg-circularity", "0.70", "--min-fg-solidity", "0.85",
        "--fill-holes", "--keep-largest-cc", "--closing-radius", "1",
    ]

    print("=== SICLE CLI default ===")
    run_sicle("sicle_cli_default", [])
    print("=== SICLE best blur05 ===")
    run_sicle("sicle_best_blur05", best_args)

    gt_arr = np.load(gt_path).astype(np.int32)
    methods = {
        "cellpose": CASE / "cp_flow" / "step04_masks_uint16.npy",
        "sicle_cli_default": CASE / "sicle_cli_default" / "merged_percell_sicle_masks_int32.npy",
        "sicle_best_blur05": CASE / "sicle_best_blur05" / "merged_percell_sicle_masks_int32.npy",
    }

    rows = []
    br_cp = None
    for name, pr_path in methods.items():
        pr_arr = np.load(pr_path).astype(np.int32)
        r = evaluate_pair(gt_path, pr_path)
        r.update(
            method=name,
            roi=ROI,
            br_mean_strict=mean_br_strict(gt_arr, pr_arr),
            br_strict_per_cell=True,
        )
        if name == "cellpose":
            br_cp = r["br_mean_strict"]
        rows.append(r)
        print(f"{name:22s} BR={r['br_mean_strict']:.4f} Dice={r['pixel_dice']:.4f}")

    for r in rows:
        if r["method"] != "cellpose":
            d = r["br_mean_strict"] - br_cp
            r["br_diff_vs_cellpose"] = d
            r["br_winner_vs_cellpose"] = "sicle" if d > 1e-6 else ("cellpose" if d < -1e-6 else "tie")

    out = SINGLE_ROI_RUN / "metrics_single_roi_config_compare.csv"
    keys = ["method", "roi", "br_mean_strict", "br_winner_vs_cellpose", "br_diff_vs_cellpose", "br_strict_per_cell"] + list(METRIC_KEYS)
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
