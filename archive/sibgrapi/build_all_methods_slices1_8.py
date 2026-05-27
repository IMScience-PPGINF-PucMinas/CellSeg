#!/usr/bin/env python3
"""Aggregate all tested methods on slices 1–8 only (ignore 9–12, union GT).

Outputs:
  reports/all_methods_slices1_8.csv       — one row per method (ALL, n_gt slices 1–8)
  reports/all_methods_slices1_8_by_slice.csv — per-slice breakdown
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
TIE_EPS = 0.02
BR_MARGIN = 8
GT_ROOT = _HERE / "out_sibgrapi2026_blur05"

SLICES_1_8 = [f"12121_40x_slice{i}" for i in range(1, 9)]

METHODS: list[tuple[str, str, str]] = [
    ("ref_sicle_cp_blur05", "out_sibgrapi2026_blur05", "sicle/merged_percell_sicle_masks_int32.npy"),
    ("sicle_nolin_blur05", "out_sibgrapi2026_nolin", "sicle/merged_percell_sicle_masks_int32.npy"),
    ("sicle_nolin_noblur", "out_sibgrapi2026_nolin_noblur", "sicle/merged_percell_sicle_masks_int32.npy"),
    ("cellpose", "out_sibgrapi2026_blur05", "cp_flow/step04_masks_uint16.npy"),
    ("cellvit", "out_cellvit_br", "cellvit_flow/step04_masks_uint16.npy"),
    ("sicle_cellvit", "out_cellvit_br", "sicle/merged_percell_sicle_masks_int32.npy"),
    ("br_pick_cp_vs_sicle", "out_cellvit_br", "br_merge/merged_br_pick_masks_int32.npy"),
]


def _gt_path(case: Path) -> Path | None:
    p = case / "gt" / "macro_nuclick_masks_int32.npy"
    return p if p.is_file() else None


def _br_per_gt_slice(gt: np.ndarray, pr: np.ndarray, margin: int = BR_MARGIN) -> dict[int, float]:
    from percell_boundary_recall import (
        bbox_of_mask,
        compute_boundary_recall,
        isolate_pred_for_gt,
    )

    gt = np.asarray(gt, dtype=np.int32)
    pr = np.asarray(pr, dtype=np.int32)
    if pr.shape != gt.shape:
        h, w = min(gt.shape[0], pr.shape[0]), min(gt.shape[1], pr.shape[1])
        gt, pr = gt[:h, :w], pr[:h, :w]
    h, w = gt.shape
    out: dict[int, float] = {}
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        m = gt == gid
        if not m.any():
            continue
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(h, r1 + margin), min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_iso = np.where(gt_crop == gid, gt_crop, 0)
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, gid)
        br, _, _ = compute_boundary_recall(pr_iso, gt_iso)
        out[gid] = float(br)
    return out


def _overlap_metrics(gt: Path, pr: Path) -> dict[str, float]:
    from evaluate_sibgrapi2026 import evaluate_pair

    r = evaluate_pair(gt, pr)
    if "error" in r:
        return {}
    return {
        "dice": float(r["pixel_dice"]),
        "aji": float(r["aji"]),
        "pq": float(r["pq"]),
        "f1": float(r["f1_0.5"]),
        "map": float(r["map_dsb"]),
        "n_gt": int(r["n_gt"]),
        "n_pr": int(r["n_pr"]),
    }


def _load_slice_data() -> tuple[
    dict[tuple[str, int], float],
    dict[str, dict[str, float]],
    int,
]:
    """Returns br_tables, overlap_by_method_slice, total n_gt cells."""
    labels = [m[0] for m in METHODS]
    br_tables: dict[str, dict[tuple[str, int], float]] = {lb: {} for lb in labels}
    overlap: dict[str, dict[str, dict[str, float]]] = {
        lb: {} for lb in labels
    }
    n_gt_total = 0
    seen_gt_keys: set[tuple[str, int]] = set()

    for stem in SLICES_1_8:
        case = GT_ROOT / stem
        gt_path = _gt_path(case)
        if gt_path is None:
            print(f"[warn] missing GT for {stem}")
            continue
        gt = np.load(gt_path).astype(np.int32)
        for gid in np.unique(gt):
            if int(gid) > 0:
                seen_gt_keys.add((stem, int(gid)))

        for label, root_rel, rel in METHODS:
            pr_path = _HERE / root_rel / stem / rel
            if not pr_path.is_file():
                print(f"[warn] missing {label} @ {stem}")
                continue
            pr = np.load(pr_path).astype(np.int32)
            if pr.shape != gt.shape:
                h, w = min(gt.shape[0], pr.shape[0]), min(gt.shape[1], pr.shape[1])
                gt_c, pr_c = gt[:h, :w], pr[:h, :w]
            else:
                gt_c, pr_c = gt, pr
            for gid, br in _br_per_gt_slice(gt_c, pr_c).items():
                br_tables[label][(stem, gid)] = br
            om = _overlap_metrics(gt_path, pr_path)
            if om:
                overlap[label][stem] = om

    n_gt_total = len(seen_gt_keys)
    return br_tables, overlap, n_gt_total


def _aggregate_br(
    br_tables: dict[str, dict[tuple[str, int], float]],
    slices: list[str],
) -> tuple[dict[str, dict[str, float | int]], int]:
    labels = list(br_tables.keys())
    keys = set()
    for t in br_tables.values():
        for k in t:
            if k[0] in slices:
                keys.add(k)

    cp = br_tables.get("cellpose", {})
    out: dict[str, dict[str, float | int]] = {}
    wins_multi = {lb: 0 for lb in labels}
    ties_multi = 0

    for key in keys:
        scores = {lb: br_tables[lb].get(key, 0.0) for lb in labels}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if ranked[0][1] - second > TIE_EPS:
            wins_multi[ranked[0][0]] += 1
        else:
            ties_multi += 1

    for label in labels:
        w = l = t = 0
        br_sum = 0.0
        n = 0
        for key in keys:
            if key not in br_tables[label]:
                continue
            br_m = br_tables[label][key]
            br_sum += br_m
            n += 1
            if label == "cellpose":
                continue
            br_c = cp.get(key, 0.0)
            diff = br_m - br_c
            if diff > TIE_EPS:
                w += 1
            elif -diff > TIE_EPS:
                l += 1
            else:
                t += 1
        row: dict[str, float | int] = {
            "n_cells": n,
            "br_mean": br_sum / n if n else float("nan"),
            "br_wins_multi": wins_multi.get(label, 0),
        }
        if label != "cellpose":
            row["br_wins_vs_cellpose"] = w
            row["br_losses_vs_cellpose"] = l
            row["br_ties_vs_cellpose"] = t
            row["pct_br_wins_vs_cellpose"] = 100.0 * w / n if n else 0.0
        else:
            w_cp = l_cp = t_cp = 0
            for key in keys:
                if key not in cp:
                    continue
                br_c = cp[key]
                diff = br_c - br_tables.get("ref_sicle_cp_blur05", {}).get(key, 0.0)
                if diff > TIE_EPS:
                    w_cp += 1
                elif -diff > TIE_EPS:
                    l_cp += 1
                else:
                    t_cp += 1
            row["br_wins_vs_cellpose"] = w_cp
            row["br_losses_vs_cellpose"] = l_cp
            row["br_ties_vs_cellpose"] = t_cp
            row["pct_br_wins_vs_cellpose"] = 100.0 * w_cp / n if n else 0.0
        out[label] = row

    return out, ties_multi


def _mean_overlap(overlap: dict[str, dict[str, dict[str, float]]], label: str, slices: list[str]) -> dict[str, float]:
    rows = [overlap[label][s] for s in slices if s in overlap.get(label, {})]
    if not rows:
        return {}
    return {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("dice", "aji", "pq", "f1", "map")
    }


def main() -> None:
    br_tables, overlap, n_gt = _load_slice_data()
    br_agg, ties_multi = _aggregate_br(br_tables, SLICES_1_8)

    out_dir = _HERE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_all = out_dir / "all_methods_slices1_8.csv"
    csv_by_slice = out_dir / "all_methods_slices1_8_by_slice.csv"

    fieldnames = [
        "method",
        "slice",
        "n_gt_cells",
        "gt_layer",
        "br_mean",
        "br_wins_vs_cellpose",
        "br_losses_vs_cellpose",
        "br_ties_vs_cellpose",
        "pct_br_wins_vs_cellpose",
        "br_wins_multi_method",
        "br_ties_multi_method",
        "dice",
        "aji",
        "pq",
        "f1",
        "map",
        "slices_included",
        "tie_eps",
        "br_strict_per_cell",
    ]

    rows_all: list[dict] = []
    rows_slice: list[dict] = []

    for label, _, _ in METHODS:
        ba = br_agg[label]
        om = _mean_overlap(overlap, label, SLICES_1_8)
        ties_m = ties_multi if label == "ref_sicle_cp_blur05" else ""
        row = {
            "method": label,
            "slice": "ALL",
            "n_gt_cells": n_gt,
            "gt_layer": "macro_nuclick",
            "br_mean": round(float(ba["br_mean"]), 6),
            "br_wins_vs_cellpose": ba.get("br_wins_vs_cellpose", ""),
            "br_losses_vs_cellpose": ba.get("br_losses_vs_cellpose", ""),
            "br_ties_vs_cellpose": ba.get("br_ties_vs_cellpose", ""),
            "pct_br_wins_vs_cellpose": round(float(ba.get("pct_br_wins_vs_cellpose", 0)), 2)
            if label != "cellpose"
            else round(100.0 * int(ba.get("br_wins_vs_cellpose", 0)) / n_gt, 2),
            "br_wins_multi_method": ba.get("br_wins_multi", 0),
            "br_ties_multi_method": ties_m if label == "ref_sicle_cp_blur05" else "",
            "dice": round(om.get("dice", float("nan")), 6),
            "aji": round(om.get("aji", float("nan")), 6),
            "pq": round(om.get("pq", float("nan")), 6),
            "f1": round(om.get("f1", float("nan")), 6),
            "map": round(om.get("map", float("nan")), 6),
            "slices_included": ",".join(SLICES_1_8),
            "tie_eps": TIE_EPS,
            "br_strict_per_cell": True,
        }
        rows_all.append(row)

        for stem in SLICES_1_8:
            keys_stem = [k for k in br_tables[label] if k[0] == stem]
            if not keys_stem:
                continue
            br_vals = [br_tables[label][k] for k in keys_stem]
            cp_vals = [br_tables["cellpose"].get(k, 0.0) for k in keys_stem]
            w = sum(1 for a, b in zip(br_vals, cp_vals) if a - b > TIE_EPS)
            l = sum(1 for a, b in zip(br_vals, cp_vals) if b - a > TIE_EPS)
            t = len(keys_stem) - w - l
            om_s = overlap.get(label, {}).get(stem, {})
            rows_slice.append(
                {
                    "method": label,
                    "slice": stem,
                    "n_gt_cells": len(keys_stem),
                    "gt_layer": "macro_nuclick",
                    "br_mean": round(float(np.mean(br_vals)), 6),
                    "br_wins_vs_cellpose": w if label != "cellpose" else l,
                    "br_losses_vs_cellpose": l if label != "cellpose" else w,
                    "br_ties_vs_cellpose": t,
                    "pct_br_wins_vs_cellpose": round(100.0 * w / len(keys_stem), 2)
                    if label != "cellpose"
                    else round(100.0 * l / len(keys_stem), 2),
                    "br_wins_multi_method": "",
                    "br_ties_multi_method": "",
                    "dice": round(om_s.get("dice", float("nan")), 6),
                    "aji": round(om_s.get("aji", float("nan")), 6),
                    "pq": round(om_s.get("pq", float("nan")), 6),
                    "f1": round(om_s.get("f1", float("nan")), 6),
                    "map": round(om_s.get("map", float("nan")), 6),
                    "slices_included": stem,
                    "tie_eps": TIE_EPS,
                    "br_strict_per_cell": True,
                }
            )

    csv_full = out_dir / "all_methods_slices1_8_full.csv"
    rows_full = rows_all + rows_slice
    for path, rows in (
        (csv_all, rows_all),
        (csv_by_slice, rows_slice),
        (csv_full, rows_full),
    ):
        with path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"n_gt (slices 1-8): {n_gt}")
    print(f"Wrote {csv_all}")
    print(f"Wrote {csv_by_slice}")
    print(f"Wrote {csv_full} ({len(rows_full)} rows)")
    print()
    for r in rows_all:
        print(
            f"{r['method']:<22} BR={r['br_mean']:.4f}  "
            f"vs_CP={r.get('br_wins_vs_cellpose','—')}/{r['n_gt_cells']}  "
            f"Dice={r['dice']:.4f}"
        )


if __name__ == "__main__":
    main()
