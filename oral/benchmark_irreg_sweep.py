#!/usr/bin/env python3
"""Sweep --sicle-irreg for gradvmaxmul+minsc (no Otsu) on benchmark ROIs."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import SICLE_COMMON, mean_br_strict

OUT_ROOT = RUNS / "irreg_sweep"
BENCH = RUNS / "path_cost_benchmark"
CSV_OUT = OUT_ROOT / "metrics_irreg.csv"

ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
]

IRREG_VALUES = (0.0, 0.04, 0.08, 0.16, 0.24, 0.50, 1.0)


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

        for irreg in IRREG_VALUES:
            tag = str(irreg).replace(".", "p")
            sid = f"gradvmaxmul_minsc_irreg{tag}"
            out_dir = case / sid
            pr_path = out_dir / "merged_percell_sicle_masks_int32.npy"
            if not pr_path.is_file():
                print(f"  run irreg={irreg}")
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
                    "--sicle-irreg",
                    str(irreg),
                    *SICLE_COMMON,
                ]
                subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            rows.append(
                {
                    "category": category,
                    "roi": stem,
                    "irreg": irreg,
                    "config_id": sid,
                    "br_mean_strict": br,
                    "pixel_dice": r.get("pixel_dice"),
                    "aji": r.get("aji"),
                }
            )
            print(f"    irreg={irreg:4.2f}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== BR macro por irreg ===")
    best_irreg, best_mean = 0.0, -1.0
    for irreg in IRREG_VALUES:
        vals = [r["br_mean_strict"] for r in rows if r["irreg"] == irreg]
        m = float(np.mean(vals))
        print(f"  irreg={irreg:4.2f}: mean BR={m:.4f}")
        if m > best_mean:
            best_mean, best_irreg = m, irreg

    base = float(np.mean([r["br_mean_strict"] for r in rows if r["irreg"] == 0.0]))
    print(f"\n  Melhor macro: irreg={best_irreg} (BR={best_mean:.4f}) vs irreg=0 (BR={base:.4f}), Δ={best_mean - base:+.4f}")
    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
