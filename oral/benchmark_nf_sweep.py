#!/usr/bin/env python3
"""
Sweep SICLE Nf (2, 5, 10, …, 500) with main pipeline = sicle_raw (no post-process).

Nf must satisfy 2 <= Nf < N0; we set N0 = max(200, Nf + 20) per variant.
Reuses cp_flow from postprocess_ablation_full when present.

Outputs:
  outputs/runs/nf_sweep_full/metrics_nf_sweep.csv
  outputs/runs/nf_sweep_full/summary.md
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

from _paths import GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import colored_to_labels, mean_br_strict
from benchmark_postprocess_ablation import _ensure_case, discover_rois

OUT_ROOT = RUNS / "nf_sweep_full"
CP_ROOT = RUNS / "postprocess_ablation_full"
CSV_OUT = OUT_ROOT / "metrics_nf_sweep.csv"

NF_VALUES = (2, 5, 10, 25, 50, 100, 250, 500)

# Main pipeline: no morph, raw SICLE paste
SICLE_RAW = [
    "--no-saliency-linearize",
    "--sicle-conn-opt", "gradvmaxmul",
    "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0",
    "--saliency-threshold", "0.3",
    "--saliency-blur-sigma", "0.5",
    "--margin", "4",
    "--min-cell-area", "128",
    "--sicle-irreg", "0",
    "--sicle-adhr", "1",
    "--sicle-max-iters", "7",
    "--disable-and-merge",
    "--closing-radius", "0",
]


def n0_for_nf(nf: int) -> int:
    return max(200, int(nf) + 20)


def _mask_ready(pr_path: Path) -> bool:
    if not pr_path.is_file() or pr_path.stat().st_size == 0:
        return False
    try:
        np.load(pr_path)
        return True
    except (OSError, ValueError, EOFError):
        return False


def _write_summary(rows: list[dict], n_rois: int) -> None:
    by_nf: dict[int, list[float]] = defaultdict(list)
    by_nf_dice: dict[int, list[float]] = defaultdict(list)
    by_nf_cat: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in rows:
        nf = int(r["nf"])
        by_nf[nf].append(float(r["br_mean_strict"]))
        if r.get("pixel_dice") is not None:
            by_nf_dice[nf].append(float(r["pixel_dice"]))
        by_nf_cat[(nf, r["category"])].append(float(r["br_mean_strict"]))

    base = float(np.mean(by_nf.get(2, [float("nan")])))
    best_nf, best_br = 2, base
    for nf in NF_VALUES:
        if by_nf[nf] and float(np.mean(by_nf[nf])) > best_br:
            best_br, best_nf = float(np.mean(by_nf[nf])), nf

    lines = [
        "# Sweep Nf — pipeline principal = SICLE cru (sem pós-processo)",
        "",
        f"ROIs: **{n_rois}**. `N0 = max(200, Nf+20)` por variante.",
        "",
        f"CSV: `{CSV_OUT.relative_to(REPO)}`",
        "",
        f"**Melhor Nf (BR macro): {best_nf}** (BR={best_br:.4f}); baseline Nf=2: BR={base:.4f}",
        "",
        "## BR médio por Nf",
        "",
        "| Nf | N0 usado | BR médio | Δ vs Nf=2 | Dice médio |",
        "|----|---------|--------:|----------:|-----------:|",
    ]
    for nf in NF_VALUES:
        br_m = float(np.mean(by_nf[nf])) if by_nf[nf] else float("nan")
        dice_m = float(np.mean(by_nf_dice[nf])) if by_nf_dice[nf] else float("nan")
        d = br_m - base if nf != 2 else 0.0
        lines.append(f"| {nf} | {n0_for_nf(nf)} | {br_m:.4f} | {d:+.4f} | {dice_m:.4f} |")

    for cat in ("healthy", "severe"):
        lines.extend(["", f"### {cat}", "", "| Nf | BR médio |", "|----|--------:|"])
        for nf in NF_VALUES:
            vals = by_nf_cat.get((nf, cat), [])
            lines.append(f"| {nf} | {np.mean(vals):.4f} |" if vals else f"| {nf} | — |")

    (OUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="All 100 ROIs (default: 4 exemplar ROIs)")
    p.add_argument("--skip-cellpose", action="store_true", help="Require cp_flow under OUT_ROOT or CP_ROOT")
    args = p.parse_args()

    if args.full:
        rois = discover_rois()
    else:
        rois = [
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
        "nf",
        "n0",
        "variant_id",
        "br_mean_strict",
        "pixel_dice",
        "aji",
        "n_gt",
        "n_pr",
    ]
    done_nf: set[tuple[str, str, str]] = set()
    if CSV_OUT.is_file():
        with CSV_OUT.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                done_nf.add((row["category"], row["roi"], row["nf"]))

    rows: list[dict] = []
    if CSV_OUT.is_file():
        with CSV_OUT.open(encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))

    n_total = len(rois) * len(NF_VALUES)

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

        for nf in NF_VALUES:
            key = (category, stem, str(nf))
            if key in done_nf:
                continue

            n0 = n0_for_nf(nf)
            sid = f"nf{nf}_n0{n0}_raw"
            out_dir = case / sid
            pr_path = out_dir / "merged_percell_sicle_masks_int32.npy"

            if not _mask_ready(pr_path):
                if pr_path.parent.is_dir():
                    shutil.rmtree(pr_path.parent, ignore_errors=True)
                print(f"  run Nf={nf} N0={n0}")
                subprocess.run(
                    [
                        py,
                        str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                        "--from-dir",
                        str(cp_dir),
                        "-o",
                        str(out_dir),
                        "--image",
                        str(input_png),
                        *SICLE_RAW,
                        "--sicle-nf",
                        str(nf),
                        "--sicle-n0",
                        str(n0),
                    ],
                    cwd=str(REPO),
                    env=env,
                    check=True,
                )

            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            row = {
                "category": category,
                "roi": stem,
                "nf": nf,
                "n0": n0,
                "variant_id": sid,
                "br_mean_strict": br,
                "pixel_dice": r.get("pixel_dice"),
                "aji": r.get("aji"),
                "n_gt": r.get("n_gt"),
                "n_pr": r.get("n_pr"),
            }
            write_header = not CSV_OUT.is_file() or CSV_OUT.stat().st_size == 0
            with CSV_OUT.open("a", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            rows.append(row)
            done_nf.add(key)
            print(f"    Nf={nf:3d}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    print(f"\nProgress: {len(done_nf)}/{n_total}")
    print("\n=== BR macro por Nf ===")
    for nf in NF_VALUES:
        vals = [float(r["br_mean_strict"]) for r in rows if int(r["nf"]) == nf]
        if vals:
            print(f"  Nf={nf:3d}: mean BR={np.mean(vals):.4f}  (n={len(vals)})")

    if rows:
        _write_summary(rows, len(rois))
    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
