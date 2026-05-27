#!/usr/bin/env python3
"""
Quantitative instance-segmentation evaluation (Dice, AJI, PQ, F1, DSB mAP).

Originally built for SIBGRAPI 2026; reused for Oral Epithelium and other datasets.

Metrics (per slice + macro-averaged across slices):

Semantic (pixel-level, foreground vs background):
    - Dice                : 2|A∩B| / (|A|+|B|)
    - Pixel IoU (Jaccard) : |A∩B| / |A∪B|

Instance-level (matched by IoU between predicted and GT instances):
    - F1@0.5 / Precision@0.5 / Recall@0.5
        Standard detection scores at IoU≥0.5 (Hungarian matching).
    - AJI  (Aggregated Jaccard Index, Kumar et al. MoNuSeg 2017)
        Each GT i matched to pred j* = argmax_j IoU(i,j); sum intersections,
        union of matched pairs + unmatched preds in the denominator.
    - PQ / SQ / RQ  (Panoptic Quality, Kirillov et al. 2019; HoVer-Net, Stardist)
        PQ = SQ · RQ. SQ = mean IoU over TPs (IoU>0.5).
        RQ = TP / (TP + 0.5 FP + 0.5 FN) = F1 at IoU>0.5.
    - DSB mAP  (Kaggle Data Science Bowl 2018 mean Average Precision)
        Mean over t∈{0.50,0.55,…,0.95} of TP_t / (TP_t + FP_t + FN_t).

Each method is evaluated against the user-selected GT layer (NuClick by default).

Usage::

    python3 evaluate_sibgrapi2026.py \\
        --out-root ./out_sibgrapi2026 \\
        --gt-layer nuclick   # also try: macro_nuclick (slices 1–8)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _instance_iou_pack(gt: "np.ndarray", pr: "np.ndarray"):
    """Return (iou, inter, gt_areas, pr_areas) for label maps (int32)."""
    import numpy as np

    gt = np.asarray(gt, dtype=np.int32)
    pr = np.asarray(pr, dtype=np.int32)
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pr)
    pr_ids = pr_ids[pr_ids > 0]
    n_gt, n_pr = int(len(gt_ids)), int(len(pr_ids))
    if n_gt == 0 or n_pr == 0:
        return (
            np.zeros((n_gt, n_pr), dtype=np.float64),
            np.zeros((n_gt, n_pr), dtype=np.float64),
            np.zeros(n_gt, dtype=np.float64),
            np.zeros(n_pr, dtype=np.float64),
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
    return iou, inter, gt_areas, pr_areas


def dice_iou_binary(gt: "np.ndarray", pr: "np.ndarray") -> tuple[float, float]:
    import numpy as np

    a = gt > 0
    b = pr > 0
    inter = float(np.logical_and(a, b).sum())
    s = float(a.sum() + b.sum())
    union = float(np.logical_or(a, b).sum())
    dice = (2.0 * inter / s) if s > 0 else 1.0
    iou = (inter / union) if union > 0 else 1.0
    return dice, iou


def aji_score(iou, inter, gt_areas, pr_areas) -> float:
    """Aggregated Jaccard Index (Kumar 2017)."""
    import numpy as np

    n_gt, n_pr = iou.shape
    if n_gt == 0 and n_pr == 0:
        return 1.0
    if n_gt == 0 or n_pr == 0:
        return 0.0
    best = iou.argmax(axis=1)
    used = np.zeros(n_pr, dtype=bool)
    isum = 0.0
    usum = 0.0
    for i in range(n_gt):
        j = int(best[i])
        if iou[i, j] > 0.0:
            isum += inter[i, j]
            usum += gt_areas[i] + pr_areas[j] - inter[i, j]
            used[j] = True
        else:
            usum += gt_areas[i]
    usum += float(pr_areas[~used].sum())
    return isum / usum if usum > 0 else 0.0


def panoptic_quality(iou) -> tuple[float, float, float, int, int, int]:
    """PQ, SQ, RQ, TP, FP, FN — match IoU > 0.5 (Hungarian for ties)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_gt, n_pr = iou.shape
    if n_gt == 0 and n_pr == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0, n_pr, 0
    if n_pr == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_gt

    cost = -np.where(iou > 0.5, iou, 0.0)
    row, col = linear_sum_assignment(cost)
    tp_iou: list[float] = []
    for r, c in zip(row, col):
        if iou[r, c] > 0.5:
            tp_iou.append(float(iou[r, c]))
    tp = len(tp_iou)
    fp = n_pr - tp
    fn = n_gt - tp
    sq = float(np.mean(tp_iou)) if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = (tp / denom) if denom > 0 else 0.0
    return sq * rq, sq, rq, tp, fp, fn


def detection_f1(iou, threshold: float = 0.5) -> tuple[float, float, float, int, int, int]:
    """Greedy/Hungarian F1, Precision, Recall at IoU>=threshold."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_gt, n_pr = iou.shape
    if n_gt == 0 and n_pr == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0, n_pr, 0
    if n_pr == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_gt
    cost = -np.where(iou >= threshold, iou, 0.0)
    row, col = linear_sum_assignment(cost)
    tp = int(sum(1 for r, c in zip(row, col) if iou[r, c] >= threshold))
    fp = n_pr - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec, tp, fp, fn


def dsb_map(iou) -> float:
    """Mean over t in [0.5..0.95 step 0.05] of TP_t/(TP_t+FP_t+FN_t)."""
    import numpy as np

    thresholds = np.arange(0.5, 1.0, 0.05)
    if iou.size == 0:
        return 0.0
    n_gt, n_pr = iou.shape
    if n_gt == 0 and n_pr == 0:
        return 1.0
    vals: list[float] = []
    for t in thresholds:
        _, _, _, tp, fp, fn = detection_f1(iou, threshold=float(t))
        denom = tp + fp + fn
        vals.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(vals))


def evaluate_pair(gt_path: Path, pr_path: Path) -> dict:
    import numpy as np

    if not gt_path.is_file() or not pr_path.is_file():
        return {"error": f"missing: gt={gt_path.is_file()} pr={pr_path.is_file()}"}
    gt = np.load(gt_path).astype(np.int32)
    pr = np.load(pr_path).astype(np.int32)
    if gt.shape != pr.shape:
        h = min(gt.shape[0], pr.shape[0])
        w = min(gt.shape[1], pr.shape[1])
        gt = gt[:h, :w]
        pr = pr[:h, :w]
    dice, p_iou = dice_iou_binary(gt, pr)
    iou, inter, gt_a, pr_a = _instance_iou_pack(gt, pr)
    aji = aji_score(iou, inter, gt_a, pr_a)
    pq, sq, rq, tp, fp, fn = panoptic_quality(iou)
    f1_50, prec_50, rec_50, tp50, fp50, fn50 = detection_f1(iou, 0.5)
    f1_75, _, _, *_ = detection_f1(iou, 0.75)
    map_dsb = dsb_map(iou)
    return {
        "n_gt": int(iou.shape[0]),
        "n_pr": int(iou.shape[1]),
        "pixel_dice": dice,
        "pixel_iou": p_iou,
        "f1_0.5": f1_50,
        "precision_0.5": prec_50,
        "recall_0.5": rec_50,
        "tp_0.5": tp50,
        "fp_0.5": fp50,
        "fn_0.5": fn50,
        "f1_0.75": f1_75,
        "aji": aji,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "pq_tp": tp,
        "pq_fp": fp,
        "pq_fn": fn,
        "map_dsb": map_dsb,
    }


METRIC_KEYS = (
    "n_gt", "n_pr",
    "pixel_dice", "pixel_iou",
    "f1_0.5", "precision_0.5", "recall_0.5", "tp_0.5", "fp_0.5", "fn_0.5",
    "f1_0.75",
    "aji",
    "pq", "sq", "rq", "pq_tp", "pq_fp", "pq_fn",
    "map_dsb",
)


def fmt_row(d: dict) -> str:
    parts = []
    for k in METRIC_KEYS:
        v = d.get(k, "")
        if isinstance(v, float):
            parts.append(f"{v:.4f}")
        else:
            parts.append(str(v))
    return "\t".join(parts)


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, default=str(here / "out_sibgrapi2026"))
    p.add_argument(
        "--gt-layer",
        type=str,
        default="auto",
        help="Base layer name in gt/ (without _masks_int32.npy). "
        "'auto' = macro_nuclick > union > nuclick (handles slices 1-8 vs 9-12).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="CSV output (default: <out-root>/eval_<gt-layer>.csv)",
    )
    args = p.parse_args()

    import numpy as np

    out_root = Path(args.out_root)
    out_csv = Path(args.out_csv) if args.out_csv else out_root / f"eval_{args.gt_layer}.csv"

    methods = [
        ("sicle",    "sicle/merged_percell_sicle_masks_int32.npy"),
        ("cellpose", "cp_flow/step04_masks_uint16.npy"),
    ]

    rows: list[dict] = []
    print(f"GT layer: {args.gt_layer}")
    print(f"Methods : {[m[0] for m in methods]}")
    print()

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
        gt_path = next((g for g in gt_candidates if g.is_file()), None)
        if gt_path is None:
            print(f"[{stem}] no GT found; skipped")
            continue
        gt_arr = np.load(gt_path)
        n_gt = int(gt_arr.max())

        for mname, rel in methods:
            pr_path = case_dir / rel
            res = evaluate_pair(gt_path, pr_path)
            row = {"slice": stem, "method": mname, "gt_layer": gt_path.parent.name + "/" + gt_path.stem}
            row.update(res)
            rows.append(row)
            err = res.get("error")
            tag = f"!! {err}" if err else (
                f"Dice={res['pixel_dice']:.3f}  AJI={res['aji']:.3f}  "
                f"PQ={res['pq']:.3f} (SQ={res['sq']:.3f} RQ={res['rq']:.3f})  "
                f"F1@.5={res['f1_0.5']:.3f}  mAP_dsb={res['map_dsb']:.3f}  "
                f"n_gt={res['n_gt']} n_pr={res['n_pr']}"
            )
            print(f"[{stem}/{mname}] {tag}")

    if not rows:
        print("No rows produced.")
        return 0

    field_order = ["slice", "method", "gt_layer", *METRIC_KEYS]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in field_order})
    print(f"\nWrote {out_csv}")

    print("\n== Macro-averaged across slices ==")
    print(f"  (slices with n_gt very small indicate incomplete NuClick annotation)")
    for mname, _ in methods:
        method_rows = [r for r in rows if r["method"] == mname and "error" not in r]
        if not method_rows:
            continue

        def avg(key: str) -> float:
            return float(np.mean([r[key] for r in method_rows]))

        print(
            f"  {mname:9s}  "
            f"Dice={avg('pixel_dice'):.4f}  "
            f"AJI={avg('aji'):.4f}  "
            f"PQ={avg('pq'):.4f} (SQ={avg('sq'):.4f} RQ={avg('rq'):.4f})  "
            f"F1@.5={avg('f1_0.5'):.4f}  "
            f"mAP_DSB={avg('map_dsb'):.4f}"
        )

    print("\n== Dense-GT subset (slices with n_gt >= 50, NuClick well-annotated) ==")
    for mname, _ in methods:
        dense_rows = [
            r for r in rows
            if r["method"] == mname and "error" not in r and r["n_gt"] >= 50
        ]
        if not dense_rows:
            continue

        def avg(key: str) -> float:
            return float(np.mean([r[key] for r in dense_rows]))

        n = len(dense_rows)
        print(
            f"  {mname:9s} (n={n})  "
            f"Dice={avg('pixel_dice'):.4f}  "
            f"AJI={avg('aji'):.4f}  "
            f"PQ={avg('pq'):.4f} (SQ={avg('sq'):.4f} RQ={avg('rq'):.4f})  "
            f"F1@.5={avg('f1_0.5'):.4f}  "
            f"mAP_DSB={avg('map_dsb'):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
