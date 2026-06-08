#!/usr/bin/env python3
"""Compare fsum+minsc vs gradvmaxmul+minsc (same criterion, no Otsu)."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import (
    SICLE_COMMON,
    colored_to_labels,
    mean_br_strict,
    run_one_roi,
)

OUT_ROOT = RUNS / "path_cost_benchmark"
CSV_OUT = OUT_ROOT / "metrics_fsum_vs_gradvminsc.csv"

CONFIGS = [
    {
        "id": "fsum_minsc_nolin",
        "label": "fsum + minsc (no Otsu)",
        "conn": "fsum",
        "crit": "minsc",
        "alpha": "2.0",
        "note": "Soma no caminho; saliência |sal(raiz)−sal(j)|.",
    },
    {
        "id": "gradvmaxmul_minsc_nolin",
        "label": "gradvmaxmul + minsc (no Otsu)",
        "conn": "gradvmaxmul",
        "crit": "minsc",
        "alpha": "2.0",
        "note": "Max no caminho; salto |g(j)−g(i)| na aresta.",
    },
]

ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
]


def main() -> int:
    rows: list[dict] = []
    for cat, stem in ROIS:
        rows.extend(run_one_roi(cat, stem, CONFIGS, skip_cellpose=True))

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== fsum+minsc vs gradvmaxmul+minsc (no Otsu, alpha=2) ===")
    by_roi: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        by_roi.setdefault((r["category"], r["roi"]), {})[r["config_id"]] = r["br_mean_strict"]

    for key in ROIS:
        m = by_roi.get(key, {})
        if len(m) < 2:
            continue
        bf, bg = m.get("fsum_minsc_nolin", float("nan")), m.get("gradvmaxmul_minsc_nolin", float("nan"))
        print(f"  {key[0]}/{key[1]}: fsum={bf:.4f}  gradvmaxmul={bg:.4f}  Δ={bg - bf:+.4f}")

    f_vals = [r["br_mean_strict"] for r in rows if r["config_id"] == "fsum_minsc_nolin"]
    g_vals = [r["br_mean_strict"] for r in rows if r["config_id"] == "gradvmaxmul_minsc_nolin"]
    if f_vals and g_vals:
        print(f"\n  BR macro: fsum={np.mean(f_vals):.4f}  gradvmaxmul={np.mean(g_vals):.4f}")

    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
