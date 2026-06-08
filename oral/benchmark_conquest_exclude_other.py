#!/usr/bin/env python3
"""
Run per-cell SICLE (main config) with iDISF-style conquest ROI on 100 ROIs and
compare to Cellpose + previous SICLE raw (without other-cell exclusion).

Outputs:
  outputs/runs/conquest_exclude_other_full/metrics_conquest_exclude_other.csv
  outputs/runs/conquest_exclude_other_full/summary.md
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import mean_br_strict
from benchmark_postprocess_ablation import _ensure_case, discover_rois

OUT_ROOT = RUNS / "conquest_exclude_other_full"
CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_LEGACY_ROOT = RUNS / "nf_sweep_full"
CSV_OUT = OUT_ROOT / "metrics_conquest_exclude_other.csv"

SICLE_RAW_BASE = [
    "--no-saliency-linearize",
    "--sicle-conn-opt", "gradvmaxmul",
    "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0",
    "--saliency-threshold", "0.3",
    "--saliency-blur-sigma", "0.5",
    "--margin", "4",
    "--min-cell-area", "128",
    "--sicle-n0", "200",
    "--sicle-nf", "2",
    "--sicle-irreg", "0",
    "--sicle-adhr", "1",
    "--sicle-max-iters", "7",
    "--disable-and-merge",
    "--closing-radius", "0",
]

VARIANT_ID = "sicle_raw_exclude_other"
OUT_SUBDIR = "sicle_raw_exclude_other"


def _mask_ready(pr_path: Path) -> bool:
    if not pr_path.is_file() or pr_path.stat().st_size == 0:
        return False
    try:
        np.load(pr_path)
        return True
    except (OSError, ValueError, EOFError):
        return False


def _resolve_legacy_sicle(category: str, stem: str) -> Path | None:
    for root in (SICLE_LEGACY_ROOT, CP_ROOT):
        for sub in ("nf2_n0200_raw", "sicle_raw"):
            p = root / category / stem / sub / "merged_percell_sicle_masks_int32.npy"
            if _mask_ready(p):
                return p
        alts = list((root / category / stem).glob("nf2_n0*_raw/merged_percell_sicle_masks_int32.npy"))
        for p in alts:
            if _mask_ready(p):
                return p
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="All 100 ROIs")
    p.add_argument("--skip-cellpose", action="store_true")
    p.add_argument("--metrics-only", action="store_true", help="Skip pipeline; only evaluate existing masks")
    args = p.parse_args()

    rois = discover_rois() if args.full else [
        ("healthy", "healthy-18-roi2"),
        ("healthy", "healthy-19-roi2"),
        ("healthy", "healthy-17-roi2"),
        ("severe", "severe-03-roi2"),
    ]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    fieldnames = [
        "category",
        "roi",
        "method",
        "br_mean_strict",
        "pixel_dice",
        "aji",
        "n_gt",
        "n_pr",
    ]
    done: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    if CSV_OUT.is_file():
        with CSV_OUT.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                done.add((row["category"], row["roi"], row["method"]))
                rows.append(row)

    for i_roi, (category, stem) in enumerate(rois, 1):
        case = OUT_ROOT / category / stem
        cp_case = CP_ROOT / category / stem
        if (cp_case / "cp_flow" / "step04_masks_uint16.npy").is_file():
            case.mkdir(parents=True, exist_ok=True)
            input_png = cp_case / f"{stem}.png"
            gt_path = cp_case / "gt" / "gold_standard_masks_int32.npy"
            cp_dir = cp_case / "cp_flow"
            if not gt_path.is_file():
                _, gt_path, _ = _ensure_case(category, stem, case, py=py, env=env, skip_cellpose=False)
        else:
            input_png, gt_path, cp_dir = _ensure_case(
                category, stem, case, py=py, env=env, skip_cellpose=args.skip_cellpose
            )

        gt_arr = np.load(gt_path).astype(np.int32)
        print(f"\n[{i_roi}/{len(rois)}] === {category}/{stem} ===")

        pr_new = case / OUT_SUBDIR / "merged_percell_sicle_masks_int32.npy"
        if not args.metrics_only and not _mask_ready(pr_new):
            if pr_new.parent.is_dir():
                shutil.rmtree(pr_new.parent, ignore_errors=True)
            print(f"  run {VARIANT_ID}")
            subprocess.run(
                [
                    py,
                    str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir",
                    str(cp_dir),
                    "-o",
                    str(case / OUT_SUBDIR),
                    "--image",
                    str(input_png),
                    *SICLE_RAW_BASE,
                ],
                cwd=str(REPO),
                env=env,
                check=True,
            )

        methods: list[tuple[str, Path]] = [
            ("cellpose", cp_dir / "step04_masks_uint16.npy"),
            (VARIANT_ID, pr_new),
        ]
        leg = _resolve_legacy_sicle(category, stem)
        if leg is not None:
            methods.append(("sicle_raw_legacy", leg))

        for method, pr_path in methods:
            key = (category, stem, method)
            if key in done:
                continue
            if not pr_path.is_file():
                print(f"    skip {method}: missing {pr_path}")
                continue
            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            row = {
                "category": category,
                "roi": stem,
                "method": method,
                "br_mean_strict": br,
                "pixel_dice": r.get("pixel_dice"),
                "aji": r.get("aji"),
                "n_gt": r.get("n_gt"),
                "n_pr": r.get("n_pr"),
            }
            rows.append(row)
            done.add(key)
            print(f"    {method:28s}: BR={br:.4f}")

    # Rewrite CSV cleanly
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_m: dict[str, list[float]] = defaultdict(list)
    by_m_dice: dict[str, list[float]] = defaultdict(list)
    wins = {VARIANT_ID: 0, "cellpose": 0, "sicle_raw_legacy": 0, "tie": 0}
    for category, stem in rois:
        cp = next((r for r in rows if r["category"] == category and r["roi"] == stem and r["method"] == "cellpose"), None)
        nw = next((r for r in rows if r["category"] == category and r["roi"] == stem and r["method"] == VARIANT_ID), None)
        if cp and nw:
            d = float(nw["br_mean_strict"]) - float(cp["br_mean_strict"])
            if d > 0.01:
                wins[VARIANT_ID] += 1
            elif d < -0.01:
                wins["cellpose"] += 1
            else:
                wins["tie"] += 1
    for r in rows:
        by_m[r["method"]].append(float(r["br_mean_strict"]))
        if r.get("pixel_dice") is not None:
            by_m_dice[r["method"]].append(float(r["pixel_dice"]))

    from summary_metrics import write_summary_md

    write_summary_md(
        OUT_ROOT / "summary.md",
        title="SICLE — exclusão de outras células na conquista",
        intro_lines=[
            f"ROIs: **{len(rois)}**. Pipeline = gradvmaxmul+minsc, Nf=2, SICLE cru.",
            "Por célula: `--mask` = fundo do crop + instância atual; saliência zerada nas outras células.",
        ],
        csv_rel=str(CSV_OUT.relative_to(REPO)),
        rows=rows,
        methods=["cellpose", "sicle_raw_legacy", VARIANT_ID],
        reference="cellpose",
        primary=VARIANT_ID,
        second_ref="sicle_raw_legacy",
        extra_metric_cols=[("pixel_dice", "Dice")],
    )
    print("\n=== BR macro ===")
    for method, vals in sorted(by_m.items()):
        print(f"  {method:28s}: {np.mean(vals):.4f}  (n={len(vals)})")
    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
