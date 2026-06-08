#!/usr/bin/env python3
"""
Validate BR with sigmoid saliency (no Otsu), sweeping SICLE Nf and multiscale.

Compares full post-process vs raw SICLE (before AND with Cellpose) to see if BR
is lost in merge/seed-removal stages.

Outputs: outputs/runs/nf_multiscale_validation/metrics.csv
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import colored_to_labels, mean_br_strict

OUT_ROOT = RUNS / "nf_multiscale_validation"
BENCH = RUNS / "path_cost_benchmark"

SICLE_BASE = [
    "--no-saliency-linearize",
    "--sicle-conn-opt", "gradvmaxmul",
    "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0",
    "--sicle-n0", "200",
    "--sicle-irreg", "0",
    "--sicle-adhr", "1",
    "--sicle-max-iters", "7",
    "--saliency-threshold", "0.3",
    "--saliency-blur-sigma", "0.5",
    "--margin", "4",
    "--min-cell-area", "128",
    "--fill-holes",
    "--keep-largest-cc",
    "--closing-radius", "1",
]

ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
]


def _variants() -> list[dict]:
    out: list[dict] = []
    for nf in (2, 3, 4, 5):
        out.append(
            {
                "id": f"nf{nf}_single_full",
                "nf": nf,
                "multiscale": False,
                "post": "full",
                "extra": ["--sicle-nf", str(nf), "--disable-and-merge", "--and-unless-round",
                          "--min-fg-circularity", "0.70", "--min-fg-solidity", "0.85"],
            }
        )
        out.append(
            {
                "id": f"nf{nf}_single_raw",
                "nf": nf,
                "multiscale": False,
                "post": "sicle_raw",
                "extra": ["--sicle-nf", str(nf), "--disable-and-merge"],
            }
        )
    for nf in (2, 3, 4):
        out.append(
            {
                "id": f"nf{nf}_ms_last_full",
                "nf": nf,
                "multiscale": True,
                "scale_select": "last",
                "post": "full",
                "extra": [
                    "--sicle-nf", str(nf),
                    "--sicle-multiscale",
                    "--sicle-scale-select", "last",
                    "--disable-and-merge",
                    "--and-unless-round",
                    "--min-fg-circularity", "0.70",
                    "--min-fg-solidity", "0.85",
                ],
            }
        )
        out.append(
            {
                "id": f"nf{nf}_ms_veta_full",
                "nf": nf,
                "multiscale": True,
                "scale_select": "veta_composite",
                "post": "full",
                "extra": [
                    "--sicle-nf", str(nf),
                    "--sicle-multiscale",
                    "--sicle-scale-select", "veta_composite",
                    "--sicle-scale-min-solidity", "0.85",
                    "--disable-and-merge",
                    "--and-unless-round",
                    "--min-fg-circularity", "0.70",
                    "--min-fg-solidity", "0.85",
                ],
            }
        )
    return out


def _ensure_case(category: str, stem: str) -> tuple[Path, Path, Path]:
    from PIL import Image

    case_bench = BENCH / category / stem
    case = OUT_ROOT / category / stem
    case.mkdir(parents=True, exist_ok=True)
    input_png = case_bench / f"{stem}.png"
    gt_path = case_bench / "gt" / "gold_standard_masks_int32.npy"
    if not input_png.is_file():
        orig = np.asarray(Image.open(IMAGES_ORIGINAL / category / f"{stem}.tif").convert("RGB"))
        col = np.asarray(Image.open(GT_COLORED / category / f"{stem}.png").convert("RGB"))
        h, w = min(orig.shape[0], col.shape[0]), min(orig.shape[1], col.shape[1])
        Image.fromarray(orig[:h, :w]).save(input_png)
        np.save(gt_path, colored_to_labels(col[:h, :w]))
    return case, input_png, gt_path


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    variants = _variants()
    rows: list[dict] = []

    for category, stem in ROIS:
        case, input_png, gt_path = _ensure_case(category, stem)
        cp_dir = BENCH / category / stem / "cp_flow"
        gt_arr = np.load(gt_path).astype(np.int32)
        print(f"\n=== {category}/{stem} ===")

        for var in variants:
            sid = var["id"]
            out_dir = case / sid
            pr_path = out_dir / "merged_percell_sicle_masks_int32.npy"
            if not pr_path.is_file():
                print(f"  run {sid}")
                cmd = [
                    py,
                    str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir",
                    str(cp_dir),
                    "-o",
                    str(out_dir),
                    *SICLE_BASE,
                    *var["extra"],
                    "--image",
                    str(input_png),
                ]
                subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            rows.append(
                {
                    "category": category,
                    "roi": stem,
                    "variant_id": sid,
                    "nf": var["nf"],
                    "multiscale": var["multiscale"],
                    "scale_select": var.get("scale_select", ""),
                    "post": var["post"],
                    "br_mean_strict": br,
                    "pixel_dice": r.get("pixel_dice"),
                    "aji": r.get("aji"),
                }
            )
            print(f"    {sid}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    csv_path = OUT_ROOT / "metrics_nf_multiscale.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # macro summary by nf (full post only, single scale)
    print("\n=== BR macro (post=full, single scale, no Otsu) ===")
    for nf in (2, 3, 4, 5):
        vals = [
            r["br_mean_strict"]
            for r in rows
            if r["post"] == "full" and r["nf"] == nf and not r["multiscale"]
        ]
        if vals:
            print(f"  Nf={nf}: mean BR={np.mean(vals):.4f}  (n={len(vals)} ROIs)")

    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
