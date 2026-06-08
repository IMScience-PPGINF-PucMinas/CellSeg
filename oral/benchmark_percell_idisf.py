#!/usr/bin/env python3
"""
Run per-cell iDISF (other cells inconquerable, BG = border only) on 100 ROIs;
compare to Cellpose and SICLE raw legacy on the same metrics.
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

METRIC_FIELDS = [
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
from benchmark_postprocess_ablation import _ensure_case, discover_rois

OUT_ROOT = RUNS / "percell_idisf_full"
CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_LEGACY_ROOT = RUNS / "nf_sweep_full"
CSV_OUT = OUT_ROOT / "metrics_percell_idisf.csv"

IDISF_ARGS = [
    "--margin", "4",
    "--min-cell-area", "128",
    "--erosion-fg", "1",
    "--erosion-bg", "1",
    "--bg-margin", "2",
    "--disable-and-merge",
]

VARIANT_ID = "idisf_unconquerable"
OUT_SUBDIR = "idisf_unconquerable"


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
        for p in (root / category / stem).glob("nf2_n0*_raw/merged_percell_sicle_masks_int32.npy"):
            if _mask_ready(p):
                return p
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--skip-cellpose", action="store_true")
    p.add_argument("--metrics-only", action="store_true")
    args = p.parse_args()

    rois = discover_rois() if args.full else [
        ("healthy", "healthy-18-roi2"),
        ("severe", "severe-03-roi2"),
    ]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PIPE))
    from boundary_fb_metric import mean_fb_strict
    from evaluate_instances import evaluate_pair

    doutorado = REPO.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PIPE),
            str(REPO / "cellpose"),
            str(doutorado),
            str(doutorado / "iDISF" / "python3"),
            env.get("PYTHONPATH", ""),
        ]
    )
    py = sys.executable

    fieldnames = METRIC_FIELDS
    rows: list[dict] = []

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

        pr_idisf = case / OUT_SUBDIR / "merged_percell_idisf_masks_int32.npy"
        if not args.metrics_only and not _mask_ready(pr_idisf):
            if pr_idisf.parent.is_dir():
                shutil.rmtree(pr_idisf.parent, ignore_errors=True)
            print(f"  run {VARIANT_ID}")
            subprocess.run(
                [
                    py,
                    str(PIPE / "percell_idisf_cellpose_pipeline.py"),
                    "--from-dir",
                    str(cp_dir),
                    "-o",
                    str(case / OUT_SUBDIR),
                    "--image",
                    str(input_png),
                    *IDISF_ARGS,
                ],
                cwd=str(REPO),
                env=env,
                check=True,
            )

        methods: list[tuple[str, Path]] = [
            ("cellpose", cp_dir / "step04_masks_uint16.npy"),
            (VARIANT_ID, pr_idisf),
        ]
        leg = _resolve_legacy_sicle(category, stem)
        if leg is not None:
            methods.append(("sicle_raw_legacy", leg))

        for method, pr_path in methods:
            if not pr_path.is_file():
                print(f"    skip {method}: missing")
                continue
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
            print(
                f"    {method:24s}: BR={br:.4f} Fb={fb:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}"
            )

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_m: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(float(r["br_mean_strict"]))

    from summary_metrics import write_summary_md

    write_summary_md(
        OUT_ROOT / "summary.md",
        title="Per-cell iDISF — outras células inconquistáveis",
        intro_lines=[
            f"ROIs: **{len(rois)}**. Outras células inconquistáveis (estilo SICLE); BG só na borda do crop.",
            "Merge: SICLE cru equivalente (`--disable-and-merge`, clip ROI).",
        ],
        csv_rel=str(CSV_OUT.relative_to(REPO)),
        rows=rows,
        methods=["cellpose", "sicle_raw_legacy", VARIANT_ID],
        reference="cellpose",
        primary=VARIANT_ID,
        second_ref="sicle_raw_legacy",
        extra_metric_cols=[("pixel_dice", "Dice"), ("aji", "AJI")],
    )

    print("\n=== BR macro ===")
    for method, vals in sorted(by_m.items()):
        print(f"  {method:24s}: {np.mean(vals):.4f}  (n={len(vals)})")
    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
