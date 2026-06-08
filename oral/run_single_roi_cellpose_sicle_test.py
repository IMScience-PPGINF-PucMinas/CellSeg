#!/usr/bin/env python3
"""
Single ROI test: Cellpose vs per-cell SICLE (sigmoid, no Otsu, blur05) on Oral Epithelium DB.

Outputs: outputs/runs/single_roi/metrics_single_roi.csv
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import DATA, GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, SINGLE_ROI_RUN

ROI = "healthy-18-roi2"
CATEGORY = "healthy"

ORIG_TIF = IMAGES_ORIGINAL / CATEGORY / f"{ROI}.tif"
COL_PNG = GT_COLORED / CATEGORY / f"{ROI}.png"
OUT = SINGLE_ROI_RUN
CASE = OUT / ROI


def mean_br_strict(gt: np.ndarray, pr: np.ndarray, margin: int = 8) -> float:
    """Mean per-GT Boundary Recall (strict: best-matching pred instance)."""
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


def colored_to_labels(rgb: np.ndarray, bg_thresh: int = 8) -> np.ndarray:
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    uniq = np.unique(rgb.reshape(-1, 3), axis=0)
    lid = 1
    for c in uniq:
        if int(c.max()) <= bg_thresh:
            continue
        labels[np.all(rgb == c, axis=2)] = lid
        lid += 1
    return labels


def main() -> int:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from evaluate_instances import METRIC_KEYS, evaluate_pair

    if not ORIG_TIF.is_file() or not COL_PNG.is_file():
        raise SystemExit(f"Missing inputs: {ORIG_TIF} or {COL_PNG}")

    CASE.mkdir(parents=True, exist_ok=True)
    gt_dir = CASE / "gt"
    cp_dir = CASE / "cp_flow"
    sicle_dir = CASE / "sicle"
    gt_dir.mkdir(exist_ok=True)

    rgb_orig = np.asarray(Image.open(ORIG_TIF).convert("RGB"))
    rgb_col = np.asarray(Image.open(COL_PNG).convert("RGB"))
    height = min(rgb_orig.shape[0], rgb_col.shape[0])
    width = min(rgb_orig.shape[1], rgb_col.shape[1])
    rgb_orig, rgb_col = rgb_orig[:height, :width], rgb_col[:height, :width]

    input_png = CASE / f"{ROI}.png"
    Image.fromarray(rgb_orig).save(input_png)

    gt_labels = colored_to_labels(rgb_col)
    gt_path = gt_dir / "gold_standard_masks_int32.npy"
    np.save(gt_path, gt_labels)
    Image.fromarray(rgb_col).save(gt_dir / "gold_standard_colored.png")

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    py = sys.executable
    print("=== Cellpose ===")
    subprocess.run(
        [py, str(PIPE / "reproduce_cellpose_pipeline.py"), str(input_png), "-o", str(cp_dir), "--gpu"],
        cwd=str(REPO),
        env=env,
        check=True,
    )

    print("=== SICLE (per-cell, nolin + blur05) ===")
    subprocess.run(
        [
            py,
            str(PIPE / "percell_sicle_cellprob_pipeline.py"),
            "--from-dir",
            str(cp_dir),
            "-o",
            str(sicle_dir),
            "--no-saliency-linearize",
            "--sicle-conn-opt",
            "gradvmaxmul",
            "--sicle-crit-opt",
            "minsc",
            "--sicle-alpha",
            "2.0",
            "--sicle-nf",
            "2",
            "--sicle-n0",
            "200",
            "--sicle-irreg",
            "0",
            "--sicle-adhr",
            "1",
            "--sicle-max-iters",
            "7",
            "--saliency-threshold",
            "0.3",
            "--saliency-blur-sigma",
            "0.5",
            "--margin",
            "4",
            "--min-cell-area",
            "128",
            "--disable-and-merge",
            "--closing-radius",
            "0",
            "--image",
            str(input_png),
        ],
        cwd=str(REPO),
        env=env,
        check=True,
    )

    methods = [
        ("cellpose", cp_dir / "step04_masks_uint16.npy"),
        ("sicle_percell", sicle_dir / "merged_percell_sicle_masks_int32.npy"),
    ]

    gt_arr = np.load(gt_path).astype(np.int32)
    tie_eps = 0.02

    rows = []
    br_by_method: dict[str, float] = {}
    print("\n=== Metrics (evaluate_instances + BR estrito por célula) ===")
    for name, pr_path in methods:
        r = evaluate_pair(gt_path, pr_path)
        pr_arr = np.load(pr_path).astype(np.int32)
        if pr_arr.shape != gt_arr.shape:
            h2, w2 = min(gt_arr.shape[0], pr_arr.shape[0]), min(gt_arr.shape[1], pr_arr.shape[1])
            pr_arr = pr_arr[:h2, :w2]
        r["br_mean_strict"] = mean_br_strict(gt_arr, pr_arr)
        br_by_method[name] = r["br_mean_strict"]
        r["method"] = name
        r["roi"] = ROI
        r["br_strict_per_cell"] = True
        rows.append(r)
        if "error" in r:
            print(f"{name}: ERROR {r['error']}")
            continue
        print(
            f"{name}: BR={r['br_mean_strict']:.4f} Dice={r['pixel_dice']:.4f} AJI={r['aji']:.4f} "
            f"PQ={r['pq']:.4f} F1@0.5={r['f1_0.5']:.4f} mAP_DSB={r['map_dsb']:.4f} "
            f"n_gt={r['n_gt']} n_pr={r['n_pr']}"
        )

    br_cp = br_by_method["cellpose"]
    br_si = br_by_method["sicle_percell"]
    for row in rows:
        if row["method"] == "sicle_percell":
            diff = br_si - br_cp
            if diff > tie_eps:
                row["br_winner_vs_cellpose"] = "sicle"
            elif -diff > tie_eps:
                row["br_winner_vs_cellpose"] = "cellpose"
            else:
                row["br_winner_vs_cellpose"] = "tie"
            row["br_diff_vs_cellpose"] = round(diff, 6)

    extra_cols = ["br_mean_strict", "br_winner_vs_cellpose", "br_diff_vs_cellpose", "br_strict_per_cell"]
    csv_path = OUT / "metrics_single_roi.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = ["method", "roi"] + extra_cols + list(METRIC_KEYS)
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metric_keys_out = list(METRIC_KEYS) + extra_cols
    summary = {
        "roi": ROI,
        "category": CATEGORY,
        "gt_instances": int(gt_labels.max()),
        "image_size": [int(height), int(width)],
        "out_dir": str(OUT),
        "sicle_config": "configs/sicle_raw_nolin_blur05.args",
        "saliency": "sigmoid only (--no-saliency-linearize)",
        "br_note": "BR = mean strict per-GT (best pred instance, margin=8px, iftEvalBR)",
        "metrics": {
            rows[i]["method"]: {k: rows[i].get(k) for k in metric_keys_out} for i in range(len(rows))
        },
    }
    (OUT / "metrics_single_roi.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {csv_path}")
    print(f"Outputs under {CASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
