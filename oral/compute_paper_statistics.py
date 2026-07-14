#!/usr/bin/env python3
"""Paired statistical tests for SIBGRAPI paper tables (oral primary benchmark)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from _paths import RUNS


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    t, p_t = stats.ttest_rel(a, b)
    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = float("nan")
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    boots = np.array(
        [np.mean(diff[np.random.randint(0, len(diff), len(diff))]) for _ in range(5000)]
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(diff),
        "delta": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "t": float(t),
        "p_t": float(p_t),
        "p_w": float(p_w),
        "cohens_d": d,
    }


def load_paired_csv(csv_path: Path, method_a: str, method_b: str) -> list[tuple[str, float, float]]:
    by: dict[str, dict[str, float]] = {}
    key_cols = ("category", "roi", "sample_id")
    with csv_path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            key = row.get("sample_id") or f"{row.get('category','')}/{row.get('roi','')}"
            by.setdefault(key, {})[row["method"]] = row
    pairs: list[tuple[str, float, float]] = []
    for key, methods in by.items():
        if method_a in methods and method_b in methods:
            pairs.append((key, float(methods[method_a]["br_mean_strict"]), float(methods[method_b]["br_mean_strict"])))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="br_mean_strict")
    p.add_argument("--latex", action="store_true", help="Print LaTeX table row")
    args = p.parse_args()

    cp_vs = RUNS / "cellpose_vs_sicle" / "metrics_cellpose_vs_sicle.csv"
    path_csv = RUNS / "path_cost_benchmark_full" / "metrics_by_roi.csv"

    reports: list[tuple[str, dict]] = []

    if cp_vs.is_file():
        by: dict[str, dict] = {}
        with cp_vs.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                key = row.get("sample_id") or f"{row['category']}/{row['roi']}"
                by.setdefault(key, {})[row["method"]] = row
        paired = [v for v in by.values() if "cellpose" in v and "sicle_nf2_raw" in v]
        if paired:
            a = np.array([float(v["sicle_nf2_raw"][args.metric]) for v in paired])
            b = np.array([float(v["cellpose"][args.metric]) for v in paired])
            reports.append(("Graph-refined vs seeds only (Cellpose)", paired_stats(a, b)))

    if path_csv.is_file():
        by2: dict[str, dict] = {}
        with path_csv.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                key = f"{row['category']}/{row['roi']}"
                by2.setdefault(key, {})[row["config_id"]] = row
        paired2 = [v for v in by2.values() if "gradvmaxmul_minsc" in v and "fmax_minsc" in v]
        if paired2:
            a = np.array([float(v["gradvmaxmul_minsc"][args.metric]) for v in paired2])
            b = np.array([float(v["fmax_minsc"][args.metric]) for v in paired2])
            reports.append(("Proposed vs literature irregular connectivity", paired_stats(a, b)))

    if not reports:
        print("No paired data found.")
        return 1

    print(f"# Metric: {args.metric}\n")
    for label, s in reports:
        print(
            f"{label} (n={s['n']}): "
            f"Δ={s['delta']:+.4f}, 95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], "
            f"t={s['t']:.2f}, p_t={s['p_t']:.2e}, p_w={s['p_w']:.2e}, d={s['cohens_d']:.3f}"
        )

    if args.latex:
        print("\n% LaTeX rows")
        for label, s in reports:
            print(
                f"{label} & {s['delta']:+.4f} & [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] & "
                f"{s['p_t']:.2e} & {s['cohens_d']:.2f} \\\\"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
