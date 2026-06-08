#!/usr/bin/env python3
"""Compare Cellpose (step04) vs SICLE raw Nf=2 on full Oral Epithelium DB."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import mean_br_strict

FIELDNAMES = [
    "category",
    "roi",
    "method",
    "br_mean_strict",
    "fb_mean_strict",
    "pixel_dice",
    "aji",
    "n_gt",
    "n_pr",
]
from benchmark_postprocess_ablation import discover_rois

OUT = RUNS / "cellpose_vs_sicle"
CSV_OUT = OUT / "metrics_cellpose_vs_sicle.csv"
CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_ROOT = RUNS / "nf_sweep_full"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PIPE))
    from boundary_fb_metric import mean_fb_strict
    from evaluate_instances import evaluate_pair

    rois = discover_rois()
    rows: list[dict] = []

    for category, stem in rois:
        case = CP_ROOT / category / stem
        gt_path = case / "gt" / "gold_standard_masks_int32.npy"
        cp_path = case / "cp_flow" / "step04_masks_uint16.npy"
        def _resolve_sicle() -> Path:
            for root in (SICLE_ROOT, CP_ROOT):
                p = root / category / stem / "nf2_n0200_raw" / "merged_percell_sicle_masks_int32.npy"
                if p.is_file() and p.stat().st_size > 0:
                    return p
                alts = list((root / category / stem).glob("nf2_n0*_raw/merged_percell_sicle_masks_int32.npy"))
                for p in alts:
                    if p.is_file() and p.stat().st_size > 0:
                        return p
                p = root / category / stem / "sicle_raw" / "merged_percell_sicle_masks_int32.npy"
                if p.is_file() and p.stat().st_size > 0:
                    return p
            raise FileNotFoundError(f"No SICLE mask for {category}/{stem}")

        sicle_path = _resolve_sicle()

        gt_arr = np.load(gt_path).astype(np.int32)

        for method, pr_path in (("cellpose", cp_path), ("sicle_nf2_raw", sicle_path)):
            r = evaluate_pair(gt_path, pr_path)
            pr_arr = np.load(pr_path).astype(np.int32)
            br = mean_br_strict(gt_arr, pr_arr)
            fb = mean_fb_strict(gt_arr, pr_arr)
            rows.append(
                {
                    "category": category,
                    "roi": stem,
                    "method": method,
                    "br_mean_strict": br,
                    "fb_mean_strict": fb,
                    "pixel_dice": r.get("pixel_dice"),
                    "aji": r.get("aji"),
                    "n_gt": r.get("n_gt"),
                    "n_pr": r.get("n_pr"),
                }
            )

        br_cp = rows[-2]["br_mean_strict"]
        br_si = rows[-1]["br_mean_strict"]
        print(f"{category}/{stem}: CP={br_cp:.4f}  SICLE={br_si:.4f}  Δ={br_si-br_cp:+.4f}")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_m: dict[str, list[float]] = defaultdict(list)
    wins = {"cellpose": 0, "sicle_nf2_raw": 0, "tie": 0}
    for category, stem in rois:
        cp = next(r for r in rows if r["category"] == category and r["roi"] == stem and r["method"] == "cellpose")
        si = next(r for r in rows if r["category"] == category and r["roi"] == stem and r["method"] == "sicle_nf2_raw")
        by_m["cellpose"].append(float(cp["br_mean_strict"]))
        by_m["sicle_nf2_raw"].append(float(si["br_mean_strict"]))
        d = float(si["br_mean_strict"]) - float(cp["br_mean_strict"])
        if d > 0.01:
            wins["sicle_nf2_raw"] += 1
        elif d < -0.01:
            wins["cellpose"] += 1
        else:
            wins["tie"] += 1

    from summary_metrics import write_summary_md

    write_summary_md(
        OUT / "summary.md",
        title="Cellpose vs SICLE raw (Nf=2)",
        intro_lines=["Comparação direta: Cellpose step04 vs SICLE cru Nf=2 (gradvmaxmul+minsc)."],
        csv_rel=str(CSV_OUT.relative_to(REPO)),
        rows=rows,
        methods=["cellpose", "sicle_nf2_raw"],
        reference="cellpose",
        primary="sicle_nf2_raw",
    )
    by_fb: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("fb_mean_strict") is not None:
            by_fb[r["method"]].append(float(r["fb_mean_strict"]))
    print(
        f"\nMacro CP BR={np.mean(by_m['cellpose']):.4f}  SICLE BR={np.mean(by_m['sicle_nf2_raw']):.4f}"
    )
    print(
        f"Macro CP Fb={np.mean(by_fb['cellpose']):.4f}  SICLE Fb={np.mean(by_fb['sicle_nf2_raw']):.4f}"
    )
    print(f"Wrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
