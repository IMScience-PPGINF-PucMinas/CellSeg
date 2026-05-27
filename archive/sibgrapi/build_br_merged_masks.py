#!/usr/bin/env python3
"""
Build a merged label image by picking Cellpose vs SICLE per GT cell using Boundary Recall.

For each GT instance, BR is computed inside bbox+margin (same as percell_boundary_recall.py).
The method with higher BR (tie within ``--tie-eps`` → SICLE) supplies the instance mask
(best-overlap label in the full prediction map).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from percell_boundary_recall import (
    bbox_of_mask,
    best_pred_label,
    compute_boundary_recall,
    isolate_pred_for_gt,
)


def build_br_merge(
    gt: "np.ndarray",
    sicle: "np.ndarray",
    cellpose: "np.ndarray",
    *,
    margin: int = 8,
    tie_eps: float = 0.02,
) -> tuple["np.ndarray", list[dict]]:
    import numpy as np

    h, w = gt.shape
    merged = np.zeros((h, w), dtype=np.int32)
    rows: list[dict] = []
    new_id = 1

    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    for gid in gt_ids:
        m = gt == int(gid)
        if not m.any():
            continue
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0 = max(0, r0 - margin)
        c0 = max(0, c0 - margin)
        r1 = min(h, r1 + margin)
        c1 = min(w, c1 + margin)

        gt_crop = gt[r0:r1, c0:c1]
        sicle_crop = sicle[r0:r1, c0:c1]
        cp_crop = cellpose[r0:r1, c0:c1]
        gt_iso = np.where(gt_crop == int(gid), gt_crop, 0)
        sicle_iso, lab_s = isolate_pred_for_gt(sicle_crop, gt_crop, int(gid))
        cp_iso, lab_c = isolate_pred_for_gt(cp_crop, gt_crop, int(gid))

        br_s, _, _ = compute_boundary_recall(sicle_iso, gt_iso)
        br_c, _, _ = compute_boundary_recall(cp_iso, gt_iso)
        diff = br_s - br_c
        if diff > tie_eps:
            winner = "sicle"
            lab = lab_s
        elif -diff > tie_eps:
            winner = "cellpose"
            lab = lab_c
        else:
            winner = "sicle" if br_s >= br_c else "cellpose"
            lab = lab_s if winner == "sicle" else lab_c

        pred = sicle if winner == "sicle" else cellpose
        if lab == 0:
            lab = best_pred_label(pred, m)
        if lab == 0:
            region = m
        else:
            region = (pred == lab)

        merged[region] = new_id
        rows.append(
            {
                "gt_id": int(gid),
                "new_label": int(new_id),
                "winner": winner,
                "br_sicle": round(br_s, 6),
                "br_cellpose": round(br_c, 6),
                "br_diff": round(diff, 6),
                "pred_label": int(lab),
            }
        )
        new_id += 1

    return merged, rows


def main() -> int:
    import numpy as np

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=str, required=True, help="GT masks int32 .npy")
    p.add_argument("--sicle", type=str, required=True, help="SICLE merged masks .npy")
    p.add_argument("--cellpose", type=str, required=True, help="Cellpose step04 .npy")
    p.add_argument("-o", "--out", type=str, required=True, help="Output merged .npy")
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--tie-eps", type=float, default=0.02)
    p.add_argument("--csv", type=str, default=None, help="Optional per-cell CSV log")
    args = p.parse_args()

    gt = np.load(args.gt).astype(np.int32)
    sicle = np.load(args.sicle).astype(np.int32)
    cp = np.load(args.cellpose).astype(np.int32)
    h, w = gt.shape
    for name, arr in ("sicle", sicle), ("cellpose", cp):
        if arr.shape != (h, w):
            raise SystemExit(f"{name} shape {arr.shape} != gt {gt.shape}")

    merged, rows = build_br_merge(gt, sicle, cp, margin=args.margin, tie_eps=args.tie_eps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, merged)

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)

    print(f"Wrote {out} ({len(rows)} GT cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
