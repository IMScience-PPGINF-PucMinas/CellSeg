#!/usr/bin/env python3
"""
Add Fb (boundary F-measure, Arbeláez/BSDS) to existing benchmark CSVs without re-segmentation.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_postprocess_ablation import discover_rois

CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_ROOT = RUNS / "nf_sweep_full"
IDISF_ROOT = RUNS / "percell_idisf_full"
CONQUEST_ROOT = RUNS / "conquest_exclude_other_full"


def _resolve_pr_path(category: str, stem: str, method: str) -> Path | None:
    if method == "cellpose":
        p = CP_ROOT / category / stem / "cp_flow" / "step04_masks_uint16.npy"
        return p if p.is_file() else None
    if method in ("sicle_nf2_raw", "sicle_raw_legacy"):
        for root in (SICLE_ROOT, CP_ROOT):
            for sub in ("nf2_n0200_raw", "sicle_raw"):
                p = root / category / stem / sub / "merged_percell_sicle_masks_int32.npy"
                if p.is_file() and p.stat().st_size > 0:
                    return p
            for p in (root / category / stem).glob("nf2_n0*_raw/merged_percell_sicle_masks_int32.npy"):
                if p.is_file() and p.stat().st_size > 0:
                    return p
    if method in ("idisf_exclude_other", "idisf_unconquerable"):
        for sub in (method, "idisf_unconquerable", "idisf_exclude_other"):
            p = IDISF_ROOT / category / stem / sub / "merged_percell_idisf_masks_int32.npy"
            if p.is_file() and p.stat().st_size > 0:
                return p
    if method == "sicle_raw_exclude_other":
        p = CONQUEST_ROOT / category / stem / "sicle_raw_exclude_other" / "merged_percell_sicle_masks_int32.npy"
        return p if p.is_file() and p.stat().st_size > 0 else None
    # postprocess ablation variant_id
    p = CP_ROOT / category / stem / method / "merged_percell_sicle_masks_int32.npy"
    if p.is_file() and p.stat().st_size > 0:
        return p
    p = CONQUEST_ROOT / category / stem / method / "merged_percell_sicle_masks_int32.npy"
    if p.is_file() and p.stat().st_size > 0:
        return p
    return None


def _augment_csv(csv_path: Path, *, bound_th: float) -> None:
    import sys

    sys.path.insert(0, str(PIPE))
    from boundary_fb_metric import mean_fb_strict

    if not csv_path.is_file():
        print(f"skip missing {csv_path}")
        return

    with csv_path.open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    if "fb_mean_strict" not in fieldnames:
        fieldnames.append("fb_mean_strict")

    for row in rows:
        category, stem = row["category"], row["roi"]
        method = row.get("method") or row.get("variant_id") or ""
        if not method and row.get("nf"):
            n0 = row.get("n0", "200")
            method = f"nf{row['nf']}_n0{n0}_raw"
        gt_path = CP_ROOT / category / stem / "gt" / "gold_standard_masks_int32.npy"
        pr_path = _resolve_pr_path(category, stem, method)
        if not gt_path.is_file() or pr_path is None:
            row["fb_mean_strict"] = ""
            continue
        gt_arr = np.load(gt_path).astype(np.int32)
        pr_arr = np.load(pr_path).astype(np.int32)
        row["fb_mean_strict"] = f"{mean_fb_strict(gt_arr, pr_arr, bound_th=bound_th):.6f}"

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"updated {csv_path}")


def _write_fb_summary(csv_path: Path, title: str, method_key: str = "method") -> None:
    if not csv_path.is_file():
        return
    with csv_path.open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    by_m: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        k = r.get(method_key) or r.get("method", "")
        if r.get("fb_mean_strict"):
            by_m[k].append(float(r["fb_mean_strict"]))

    lines = [
        f"# {title}",
        "",
        "Fb = boundary F-measure (Arbeláez/BSDS; tolerance 0.0075 × diagonal, per-cell strict).",
        "",
        "| Método | Fb médio | BR médio |",
        "|--------|--------:|--------:|",
    ]
    for method, fb_vals in sorted(by_m.items()):
        br_vals = [
            float(r["br_mean_strict"])
            for r in rows
            if (r.get(method_key) or r.get("method")) == method and r.get("br_mean_strict")
        ]
        fb_m = float(np.mean(fb_vals)) if fb_vals else float("nan")
        br_m = float(np.mean(br_vals)) if br_vals else float("nan")
        lines.append(f"| `{method}` | {fb_m:.4f} | {br_m:.4f} |")

    summary_path = csv_path.parent / "summary_fb.md"
    existing = csv_path.parent / "summary.md"
    if existing.is_file():
        text = existing.read_text(encoding="utf-8")
        if "## Fb" not in text:
            existing.write_text(text + "\n\n" + "\n".join(lines[2:]), encoding="utf-8")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {summary_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bound-th", type=float, default=0.0075)
    p.add_argument("--csv", type=str, action="append", default=[])
    args = p.parse_args()

    default_csvs = [
        RUNS / "cellpose_vs_sicle" / "metrics_cellpose_vs_sicle.csv",
        RUNS / "percell_idisf_full" / "metrics_percell_idisf.csv",
        RUNS / "conquest_exclude_other_full" / "metrics_conquest_exclude_other.csv",
        RUNS / "postprocess_ablation_full" / "metrics_postprocess.csv",
        RUNS / "nf_sweep_full" / "metrics_nf_sweep.csv",
    ]
    csvs = [Path(c) for c in args.csv] if args.csv else default_csvs

    for csv_path in csvs:
        _augment_csv(csv_path, bound_th=args.bound_th)

    _write_fb_summary(
        RUNS / "percell_idisf_full" / "metrics_percell_idisf.csv",
        "iDISF — Fb",
    )
    _write_fb_summary(
        RUNS / "cellpose_vs_sicle" / "metrics_cellpose_vs_sicle.csv",
        "Cellpose vs SICLE — Fb",
    )
    _write_fb_summary(
        RUNS / "conquest_exclude_other_full" / "metrics_conquest_exclude_other.csv",
        "SICLE exclude other — Fb",
    )
    _write_fb_summary(
        RUNS / "postprocess_ablation_full" / "metrics_postprocess.csv",
        "Pós-processo — Fb",
        method_key="variant_id",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
