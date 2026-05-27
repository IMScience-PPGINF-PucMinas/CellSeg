#!/usr/bin/env python3
"""
Per-cell comparison of SICLE vs Cellpose predictions against NuClick GT.

For every GT instance in every slice:
  - finds the predicted instance with maximum IoU in each method (SICLE, Cellpose),
  - records IoU / Dice / areas / IDs,
  - declares a "winner" using a configurable IoU tie tolerance.

Outputs (under ``--out-root``):
  - ``percell_comparison.csv``  : one row per GT cell (across all slices)
  - ``percell_summary.csv``     : per-slice + overall win counts

The "winner" per GT cell is:
  - ``sicle``       if  IoU_sicle - IoU_cellpose >  --tie-eps
  - ``cellpose``    if  IoU_cellpose - IoU_sicle >  --tie-eps
  - ``tie``         otherwise, when at least one method has IoU > 0
  - ``both_missed`` when neither method reaches IoU >= --min-iou (default 0.0)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def best_match_per_gt(
    gt: "np.ndarray",
    pr: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """For each GT instance, return best pred id, IoU, Dice, gt_area, pr_area, inter."""
    import numpy as np

    gt = np.asarray(gt, dtype=np.int32)
    pr = np.asarray(pr, dtype=np.int32)
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pr)
    pr_ids = pr_ids[pr_ids > 0]
    n_gt = len(gt_ids)
    n_pr = len(pr_ids)
    if n_gt == 0:
        e = np.zeros(0)
        return e.astype(np.int32), e, e, e, e, e
    if n_pr == 0:
        z = np.zeros(n_gt)
        gt_areas = np.array(
            [int((gt == i).sum()) for i in gt_ids], dtype=np.float64
        )
        return (
            np.zeros(n_gt, dtype=np.int32),
            z, z, gt_areas, z, z,
        )

    g_remap = np.zeros(int(gt.max()) + 1, dtype=np.int32)
    g_remap[gt_ids] = np.arange(1, n_gt + 1)
    p_remap = np.zeros(int(pr.max()) + 1, dtype=np.int32)
    p_remap[pr_ids] = np.arange(1, n_pr + 1)
    g_r = g_remap[gt]
    p_r = p_remap[pr]

    pairs = g_r.astype(np.int64) * (n_pr + 1) + p_r.astype(np.int64)
    counts = np.bincount(pairs.ravel(), minlength=(n_gt + 1) * (n_pr + 1))
    inter = counts.reshape(n_gt + 1, n_pr + 1)[1:, 1:].astype(np.float64)

    gt_areas = np.bincount(g_r.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)
    pr_areas = np.bincount(p_r.ravel(), minlength=n_pr + 1)[1:].astype(np.float64)

    union = gt_areas[:, None] + pr_areas[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    best_col = iou.argmax(axis=1)
    best_iou = iou[np.arange(n_gt), best_col]
    best_inter = inter[np.arange(n_gt), best_col]
    best_pr_area = pr_areas[best_col]

    matched_pr_ids = pr_ids[best_col]
    # if no overlap, no match
    no_match = best_iou <= 0.0
    best_iou[no_match] = 0.0
    best_pr_area[no_match] = 0.0
    matched_pr_ids[no_match] = 0
    denom = gt_areas + best_pr_area
    dice = np.where(denom > 0, 2.0 * best_inter / denom, 0.0)

    return matched_pr_ids.astype(np.int32), best_iou, dice, gt_areas, best_pr_area, best_inter


def main() -> int:
    import numpy as np

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, default=str(here / "out_sibgrapi2026_clean"))
    p.add_argument(
        "--gt-layer",
        type=str,
        default="auto",
        help="'auto' = macro_nuclick > union > nuclick (slices 1-8 vs 9-12).",
    )
    p.add_argument(
        "--tie-eps",
        type=float,
        default=0.02,
        help="IoU absolute tie tolerance (default 0.02). |IoU_sicle - IoU_cp| < eps -> tie.",
    )
    p.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        help="Below this IoU on BOTH methods we mark winner='both_missed' (default 0.0).",
    )
    p.add_argument(
        "--cells-csv",
        type=str,
        default=None,
        help="Output per-cell CSV (default: <out-root>/percell_comparison.csv)",
    )
    p.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Output summary CSV (default: <out-root>/percell_summary.csv)",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    cells_csv = Path(args.cells_csv) if args.cells_csv else out_root / "percell_comparison.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else out_root / "percell_summary.csv"

    method_rel = {
        "sicle":    "sicle/merged_percell_sicle_masks_int32.npy",
        "cellpose": "cp_flow/step04_masks_uint16.npy",
    }

    cell_rows: list[dict] = []
    summary_rows: list[dict] = []

    overall = {
        "n_gt": 0, "sicle_wins": 0, "cellpose_wins": 0, "ties": 0, "both_missed": 0,
        "sicle_iou_sum": 0.0, "cellpose_iou_sum": 0.0,
        "sicle_dice_sum": 0.0, "cellpose_dice_sum": 0.0,
        "sicle_iou_higher_strict": 0, "cellpose_iou_higher_strict": 0,
    }

    for case_dir in sorted(out_root.iterdir()):
        if not case_dir.is_dir():
            continue
        stem = case_dir.name
        if args.gt_layer == "auto":
            gt_candidates = [
                case_dir / "gt" / "macro_nuclick_masks_int32.npy",
                case_dir / "gt" / "union_masks_int32.npy",
                case_dir / "gt" / "nuclick_masks_int32.npy",
            ]
        else:
            gt_candidates = [
                case_dir / "gt" / f"{args.gt_layer}_masks_int32.npy",
                case_dir / "gt" / "macro_nuclick_masks_int32.npy",
                case_dir / "gt" / "union_masks_int32.npy",
                case_dir / "gt" / "nuclick_masks_int32.npy",
            ]
        gt_path = next((g for g in gt_candidates if g.is_file()), gt_candidates[0])
        sicle_path = case_dir / method_rel["sicle"]
        cp_path = case_dir / method_rel["cellpose"]
        if not (gt_path.is_file() and sicle_path.is_file() and cp_path.is_file()):
            print(f"[{stem}] missing inputs; skip")
            continue

        gt = np.load(gt_path).astype(np.int32)
        sicle = np.load(sicle_path).astype(np.int32)
        cp = np.load(cp_path).astype(np.int32)

        if sicle.shape != gt.shape:
            h = min(gt.shape[0], sicle.shape[0])
            w = min(gt.shape[1], sicle.shape[1])
            gt = gt[:h, :w]
            sicle = sicle[:h, :w]
            cp = cp[:h, :w]

        gt_ids = np.unique(gt)
        gt_ids = gt_ids[gt_ids > 0]
        if gt_ids.size == 0:
            print(f"[{stem}] no GT instances; skip")
            continue

        sid, siou, sdice, garea_s, sarea, _ = best_match_per_gt(gt, sicle)
        cid, ciou, cdice, garea_c, carea, _ = best_match_per_gt(gt, cp)

        per_slice = {
            "slice": stem, "n_gt": int(gt_ids.size),
            "sicle_wins": 0, "cellpose_wins": 0, "ties": 0, "both_missed": 0,
            "sicle_iou_sum": 0.0, "cellpose_iou_sum": 0.0,
            "sicle_dice_sum": 0.0, "cellpose_dice_sum": 0.0,
        }

        for k, gid in enumerate(gt_ids):
            iou_s = float(siou[k])
            iou_c = float(ciou[k])
            d_s = float(sdice[k])
            d_c = float(cdice[k])
            diff = iou_s - iou_c

            if iou_s < args.min_iou and iou_c < args.min_iou:
                winner = "both_missed"
            elif diff > args.tie_eps:
                winner = "sicle"
            elif -diff > args.tie_eps:
                winner = "cellpose"
            else:
                winner = "tie"

            cell_rows.append(
                {
                    "slice": stem,
                    "gt_id": int(gid),
                    "gt_area": int(garea_s[k]),
                    "sicle_pred_id": int(sid[k]),
                    "sicle_pred_area": int(sarea[k]),
                    "sicle_iou": round(iou_s, 6),
                    "sicle_dice": round(d_s, 6),
                    "cellpose_pred_id": int(cid[k]),
                    "cellpose_pred_area": int(carea[k]),
                    "cellpose_iou": round(iou_c, 6),
                    "cellpose_dice": round(d_c, 6),
                    "iou_diff_sicle_minus_cp": round(diff, 6),
                    "winner": winner,
                }
            )

            per_slice[f"{winner}_wins" if winner in {"sicle", "cellpose"} else ("ties" if winner == "tie" else "both_missed")] += 1
            per_slice["sicle_iou_sum"] += iou_s
            per_slice["cellpose_iou_sum"] += iou_c
            per_slice["sicle_dice_sum"] += d_s
            per_slice["cellpose_dice_sum"] += d_c

        n = per_slice["n_gt"]
        per_slice["sicle_iou_mean"] = round(per_slice["sicle_iou_sum"] / n, 4)
        per_slice["cellpose_iou_mean"] = round(per_slice["cellpose_iou_sum"] / n, 4)
        per_slice["sicle_dice_mean"] = round(per_slice["sicle_dice_sum"] / n, 4)
        per_slice["cellpose_dice_mean"] = round(per_slice["cellpose_dice_sum"] / n, 4)
        per_slice["pct_sicle_wins"] = round(100.0 * per_slice["sicle_wins"] / n, 2)
        per_slice["pct_cellpose_wins"] = round(100.0 * per_slice["cellpose_wins"] / n, 2)
        per_slice["pct_ties"] = round(100.0 * per_slice["ties"] / n, 2)
        summary_rows.append(per_slice)

        overall["n_gt"] += n
        for k in ("sicle_wins", "cellpose_wins", "ties", "both_missed",
                  "sicle_iou_sum", "cellpose_iou_sum",
                  "sicle_dice_sum", "cellpose_dice_sum"):
            overall[k] += per_slice[k]

        # raw IoU dominance regardless of eps
        for k in range(n):
            if siou[k] > ciou[k]:
                overall["sicle_iou_higher_strict"] += 1
            elif ciou[k] > siou[k]:
                overall["cellpose_iou_higher_strict"] += 1

    if not cell_rows:
        print("No data.")
        return 0

    # write per-cell CSV
    field_cells = [
        "slice", "gt_id", "gt_area",
        "sicle_pred_id", "sicle_pred_area", "sicle_iou", "sicle_dice",
        "cellpose_pred_id", "cellpose_pred_area", "cellpose_iou", "cellpose_dice",
        "iou_diff_sicle_minus_cp", "winner",
    ]
    with cells_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=field_cells)
        w.writeheader()
        for r in cell_rows:
            w.writerow(r)
    print(f"Wrote {cells_csv}  ({len(cell_rows)} rows)")

    # write summary CSV
    field_sum = [
        "slice", "n_gt",
        "sicle_wins", "cellpose_wins", "ties", "both_missed",
        "pct_sicle_wins", "pct_cellpose_wins", "pct_ties",
        "sicle_iou_mean", "cellpose_iou_mean",
        "sicle_dice_mean", "cellpose_dice_mean",
    ]
    n_total = overall["n_gt"]
    overall_row = {
        "slice": "ALL",
        "n_gt": n_total,
        "sicle_wins": overall["sicle_wins"],
        "cellpose_wins": overall["cellpose_wins"],
        "ties": overall["ties"],
        "both_missed": overall["both_missed"],
        "pct_sicle_wins": round(100.0 * overall["sicle_wins"] / n_total, 2),
        "pct_cellpose_wins": round(100.0 * overall["cellpose_wins"] / n_total, 2),
        "pct_ties": round(100.0 * overall["ties"] / n_total, 2),
        "sicle_iou_mean": round(overall["sicle_iou_sum"] / n_total, 4),
        "cellpose_iou_mean": round(overall["cellpose_iou_sum"] / n_total, 4),
        "sicle_dice_mean": round(overall["sicle_dice_sum"] / n_total, 4),
        "cellpose_dice_mean": round(overall["cellpose_dice_sum"] / n_total, 4),
    }
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=field_sum)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in field_sum})
        w.writerow(overall_row)
    print(f"Wrote {summary_csv}")

    # console report
    print()
    print(f"== Per-cell wins (eps={args.tie_eps}, min_iou={args.min_iou}) ==")
    print(f"{'slice':<25} {'n_gt':>5} {'SICLE':>7} {'CP':>7} {'tie':>6} {'miss':>5}  {'%SICLE':>7} {'%CP':>6}  IoU_S/CP   Dice_S/CP")
    for r in summary_rows:
        print(
            f"{r['slice']:<25} {r['n_gt']:>5d} "
            f"{r['sicle_wins']:>7d} {r['cellpose_wins']:>7d} "
            f"{r['ties']:>6d} {r['both_missed']:>5d}  "
            f"{r['pct_sicle_wins']:>6.2f}% {r['pct_cellpose_wins']:>5.2f}%  "
            f"{r['sicle_iou_mean']:.3f}/{r['cellpose_iou_mean']:.3f}  "
            f"{r['sicle_dice_mean']:.3f}/{r['cellpose_dice_mean']:.3f}"
        )
    print(
        f"{'ALL':<25} {overall_row['n_gt']:>5d} "
        f"{overall_row['sicle_wins']:>7d} {overall_row['cellpose_wins']:>7d} "
        f"{overall_row['ties']:>6d} {overall_row['both_missed']:>5d}  "
        f"{overall_row['pct_sicle_wins']:>6.2f}% {overall_row['pct_cellpose_wins']:>5.2f}%  "
        f"{overall_row['sicle_iou_mean']:.3f}/{overall_row['cellpose_iou_mean']:.3f}  "
        f"{overall_row['sicle_dice_mean']:.3f}/{overall_row['cellpose_dice_mean']:.3f}"
    )
    print()
    print(
        f"  Strict IoU dominance (ignoring eps): "
        f"SICLE > CP in {overall['sicle_iou_higher_strict']} cells, "
        f"CP > SICLE in {overall['cellpose_iou_higher_strict']} cells, "
        f"equal in {n_total - overall['sicle_iou_higher_strict'] - overall['cellpose_iou_higher_strict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
