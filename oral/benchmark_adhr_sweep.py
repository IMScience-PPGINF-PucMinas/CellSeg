#!/usr/bin/env python3
"""Sweep --sicle-adhr for gradvmaxmul+minsc (no Otsu) on benchmark ROIs."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import SICLE_COMMON, mean_br_strict

OUT_ROOT = RUNS / "adhr_sweep"
BENCH = RUNS / "path_cost_benchmark"
CSV_OUT = OUT_ROOT / "metrics_adhr.csv"

ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
]

ADHR_VALUES = (1, 2, 4, 8, 12, 24)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    rows: list[dict] = []

    for category, stem in ROIS:
        case_bench = BENCH / category / stem
        case = OUT_ROOT / category / stem
        input_png = case_bench / f"{stem}.png"
        gt_path = case_bench / "gt" / "gold_standard_masks_int32.npy"
        cp_dir = case_bench / "cp_flow"
        gt_arr = np.load(gt_path).astype(np.int32)
        print(f"\n=== {category}/{stem} ===")

        for adhr in ADHR_VALUES:
            sid = f"gradvmaxmul_minsc_adhr{adhr}"
            out_dir = case / sid
            pr_path = out_dir / "merged_percell_sicle_masks_int32.npy"
            if not pr_path.is_file():
                print(f"  run adhr={adhr}")
                cmd = [
                    py,
                    str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir",
                    str(cp_dir),
                    "-o",
                    str(out_dir),
                    "--image",
                    str(input_png),
                    "--sicle-conn-opt",
                    "gradvmaxmul",
                    "--sicle-crit-opt",
                    "minsc",
                    "--sicle-alpha",
                    "2.0",
                    "--sicle-adhr",
                    str(adhr),
                    *SICLE_COMMON,
                ]
                subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            rows.append(
                {
                    "category": category,
                    "roi": stem,
                    "adhr": adhr,
                    "config_id": sid,
                    "br_mean_strict": br,
                    "pixel_dice": r.get("pixel_dice"),
                    "aji": r.get("aji"),
                }
            )
            print(f"    adhr={adhr:2d}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== BR macro por adhr ===")
    best_adhr, best_mean = 1, -1.0
    for adhr in ADHR_VALUES:
        vals = [r["br_mean_strict"] for r in rows if r["adhr"] == adhr]
        m = float(np.mean(vals))
        print(f"  adhr={adhr:2d}: mean BR={m:.4f}")
        if m > best_mean:
            best_mean, best_adhr = m, adhr

    base = float(np.mean([r["br_mean_strict"] for r in rows if r["adhr"] == 1]))
    print(f"\n  Melhor macro: adhr={best_adhr} (BR={best_mean:.4f}) vs adhr=1 (BR={base:.4f}), Δ={best_mean - base:+.4f}")
    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
