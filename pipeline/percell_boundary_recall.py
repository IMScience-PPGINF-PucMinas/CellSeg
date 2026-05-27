#!/usr/bin/env python3
"""
Per-cell Boundary Recall (BR) comparison: SICLE vs Cellpose against NuClick GT.

Uses the same BR definition as SICLE's ``iftEvalBR`` (Stutz et al., 2018):
  - For each pixel p, check if any pixel within a (2r+1)² window has a
    different label.  If so, p is on the *label border*.  Same for GT.
  - Tolerance radius r = ceil(0.0025 * sqrt(H² + W²)).
  - TP = pixels that are GT-border AND label-border (within tolerance).
  - FN = GT-border pixels that are NOT label-border.
  - BR = TP / (TP + FN).

By default (**strict per-cell**), each method contributes only the predicted
instance with largest overlap on that GT cell (not all neighbors in the crop).
Use ``--legacy-full-crop`` for the old behavior (any pred border in the crop).

For every GT instance, BR is computed inside its bounding box (+margin) for
each method, the winner is recorded, and a composite PNG with the three
contours (GT cyan, SICLE green, Cellpose yellow) is saved under

  <out_root>/br_analysis/<slice>/{sicle_wins,cellpose_wins,ties}/cell_XXXXX.png

Outputs:
  <out_root>/br_analysis/per_cell_br_all.csv       (one row per GT cell)
  <out_root>/br_analysis/per_cell_br_summary.csv   (per slice + ALL)
  <out_root>/br_analysis/<slice>/per_cell_br.csv   (one CSV per image)
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def compute_boundary_recall(
    label: "np.ndarray", gt: "np.ndarray", tolerance_r: int | None = None
) -> tuple[float, int, int]:
    """Returns (BR, TP, FN) computed inside the given crop."""
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    lab = np.asarray(label, dtype=np.int64)
    gt_ = np.asarray(gt, dtype=np.int64)
    if lab.shape != gt_.shape:
        raise ValueError(f"label shape {lab.shape} != gt shape {gt_.shape}")
    h, w = lab.shape
    if tolerance_r is None:
        diag = float(np.sqrt(h * h + w * w))
        tolerance_r = int(np.ceil(0.0025 * diag))
        tolerance_r = max(1, tolerance_r)
    ws = 2 * tolerance_r + 1

    mx_lab = maximum_filter(lab, size=ws, mode="constant", cval=-(2**62))
    mn_lab = minimum_filter(lab, size=ws, mode="constant", cval=(2**62) - 1)
    is_lab_border = (mx_lab != lab) | (mn_lab != lab)

    mx_gt = maximum_filter(gt_, size=ws, mode="constant", cval=-(2**62))
    mn_gt = minimum_filter(gt_, size=ws, mode="constant", cval=(2**62) - 1)
    is_gt_border = (mx_gt != gt_) | (mn_gt != gt_)

    tp = int(np.logical_and(is_gt_border, is_lab_border).sum())
    fn = int(np.logical_and(is_gt_border, np.logical_not(is_lab_border)).sum())
    denom = tp + fn
    br = float(tp / denom) if denom > 0 else 1.0
    return br, tp, fn


def bbox_of_mask(mask: "np.ndarray") -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def best_pred_label(pred: "np.ndarray", gt_bool: "np.ndarray") -> int:
    """Label id in ``pred`` with largest overlap on ``gt_bool`` (0 if none)."""
    import numpy as np

    sub = pred[gt_bool]
    labs = np.unique(sub)
    labs = labs[labs > 0]
    if labs.size == 0:
        return 0
    best, best_n = 0, 0
    for lab in labs:
        n = int((sub == lab).sum())
        if n > best_n:
            best_n, best = n, int(lab)
    return best


def isolate_pred_for_gt(
    pred_crop: "np.ndarray", gt_crop: "np.ndarray", gt_id: int
) -> tuple["np.ndarray", int]:
    """Keep only the predicted instance that best overlaps this GT cell (strict per-cell BR)."""
    import numpy as np

    m = gt_crop == int(gt_id)
    lab = best_pred_label(pred_crop, m)
    if lab == 0:
        return np.zeros(pred_crop.shape, dtype=np.int32), 0
    iso = np.where(pred_crop == lab, np.int32(1), np.int32(0))
    return iso, lab


def draw_contours(
    rgb: "np.ndarray",
    masks: "np.ndarray",
    color: tuple[int, int, int],
    thickness: int = 1,
) -> "np.ndarray":
    import cv2
    import numpy as np

    L = np.asarray(masks, dtype=np.int32)
    Lpad = np.pad(L, 1, mode="constant", constant_values=-1)
    border = (L > 0) & (
        (Lpad[1:-1, :-2] != L)
        | (Lpad[1:-1, 2:] != L)
        | (Lpad[:-2, 1:-1] != L)
        | (Lpad[2:, 1:-1] != L)
    )
    if thickness > 1:
        k = max(3, 2 * int(thickness) - 1)
        border = cv2.dilate(border.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)
    out = np.asarray(rgb[..., :3], dtype=np.uint8).copy()
    r, g, b = color
    out[border, 0] = r
    out[border, 1] = g
    out[border, 2] = b
    return out


def main() -> int:
    import numpy as np
    from PIL import Image

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=str,
        default=str(here / "out_sibgrapi2026_sweep_v2" / "01_irreg_0_50"),
        help="Folder with <slice>/sicle/, <slice>/cp_flow/, <slice>/gt/",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(here / "data_sibgrapi2026" / "data_sibgrapi2026"),
        help="Folder with <slice>.png originals (for the composites).",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Pixels of context added around each GT bbox when computing BR.",
    )
    p.add_argument(
        "--tie-eps",
        type=float,
        default=0.02,
        help="|BR_sicle - BR_cellpose| <= eps → tie (default 0.02).",
    )
    p.add_argument(
        "--clean-existing",
        action="store_true",
        help="Wipe the br_analysis/ folder before running.",
    )
    p.add_argument(
        "--legacy-full-crop",
        action="store_true",
        help=(
            "Legacy BR: use all prediction borders in bbox+margin (neighbors can affect BR). "
            "Default is strict per-cell: only the best-matching pred instance per GT."
        ),
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    data_dir = Path(args.data_dir)
    br_dir = out_root / "br_analysis"
    if args.clean_existing and br_dir.exists():
        shutil.rmtree(br_dir)
    br_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    overall = {"n_gt": 0, "s_wins": 0, "c_wins": 0, "ties": 0,
               "br_s_sum": 0.0, "br_c_sum": 0.0}

    for case_dir in sorted(out_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name == "br_analysis":
            continue
        stem = case_dir.name
        gt_candidates = [
            case_dir / "gt" / "macro_nuclick_masks_int32.npy",
            case_dir / "gt" / "union_masks_int32.npy",
            case_dir / "gt" / "nuclick_masks_int32.npy",
        ]
        gt_path = next((g for g in gt_candidates if g.is_file()), None)
        sicle_path = case_dir / "sicle" / "merged_percell_sicle_masks_int32.npy"
        cp_path = case_dir / "cp_flow" / "step04_masks_uint16.npy"
        png_path = data_dir / f"{stem}.png"
        if not (gt_path and sicle_path.is_file() and cp_path.is_file() and png_path.is_file()):
            print(f"[{stem}] missing inputs; skip")
            continue

        gt = np.load(gt_path).astype(np.int32)
        sicle = np.load(sicle_path).astype(np.int32)
        cp = np.load(cp_path).astype(np.int32)
        rgb = np.asarray(Image.open(png_path).convert("RGB"))
        H, W = gt.shape
        if sicle.shape != gt.shape:
            h = min(gt.shape[0], sicle.shape[0])
            w = min(gt.shape[1], sicle.shape[1])
            gt, sicle, cp = gt[:h, :w], sicle[:h, :w], cp[:h, :w]
            rgb = rgb[:h, :w]
            H, W = h, w

        slice_dir = br_dir / stem
        for sub in ("sicle_wins", "cellpose_wins", "ties"):
            (slice_dir / sub).mkdir(parents=True, exist_ok=True)
        slice_csv = slice_dir / "per_cell_br.csv"
        slice_rows: list[dict] = []

        gt_ids = np.unique(gt)
        gt_ids = gt_ids[gt_ids > 0]
        per_slice = {"slice": stem, "n_gt": int(gt_ids.size),
                     "sicle_wins": 0, "cellpose_wins": 0, "ties": 0,
                     "br_sicle_mean": 0.0, "br_cellpose_mean": 0.0}
        br_s_acc = br_c_acc = 0.0

        for gid in gt_ids:
            m = gt == int(gid)
            if not m.any():
                continue
            r0, r1, c0, c1 = bbox_of_mask(m)
            r0 = max(0, r0 - args.margin)
            c0 = max(0, c0 - args.margin)
            r1 = min(H, r1 + args.margin)
            c1 = min(W, c1 + args.margin)

            gt_crop = gt[r0:r1, c0:c1]
            sicle_crop = sicle[r0:r1, c0:c1]
            cp_crop = cp[r0:r1, c0:c1]
            rgb_crop = rgb[r0:r1, c0:c1]
            gt_isolated = np.where(gt_crop == int(gid), gt_crop, 0)

            if args.legacy_full_crop:
                sicle_br_in = sicle_crop
                cp_br_in = cp_crop
                lab_s, lab_c = -1, -1
            else:
                sicle_br_in, lab_s = isolate_pred_for_gt(sicle_crop, gt_crop, int(gid))
                cp_br_in, lab_c = isolate_pred_for_gt(cp_crop, gt_crop, int(gid))

            br_s, tp_s, fn_s = compute_boundary_recall(sicle_br_in, gt_isolated)
            br_c, tp_c, fn_c = compute_boundary_recall(cp_br_in, gt_isolated)
            diff = br_s - br_c
            if diff > args.tie_eps:
                winner = "sicle"
            elif -diff > args.tie_eps:
                winner = "cellpose"
            else:
                winner = "tie"

            row = {
                "slice": stem,
                "gt_id": int(gid),
                "gt_area": int(m.sum()),
                "bbox": f"({r0},{r1},{c0},{c1})",
                "pred_label_sicle": lab_s,
                "pred_label_cellpose": lab_c,
                "br_sicle": round(br_s, 6),
                "br_cellpose": round(br_c, 6),
                "br_diff_sicle_minus_cp": round(diff, 6),
                "tp_sicle": tp_s,
                "fn_sicle": fn_s,
                "tp_cellpose": tp_c,
                "fn_cellpose": fn_c,
                "winner": winner,
            }
            slice_rows.append(row)
            all_rows.append(row)

            comp = draw_contours(rgb_crop, gt_isolated, (0, 255, 255), thickness=1)
            comp = draw_contours(comp, sicle_br_in, (0, 255, 0), thickness=1)
            comp = draw_contours(comp, cp_br_in, (255, 255, 0), thickness=1)
            sub = {"sicle": "sicle_wins", "cellpose": "cellpose_wins", "tie": "ties"}[winner]
            fname = (
                f"cell_{int(gid):05d}_brS{br_s:.3f}_brC{br_c:.3f}.png"
            )
            Image.fromarray(comp).save(slice_dir / sub / fname)

            if winner == "sicle":
                per_slice["sicle_wins"] += 1
            elif winner == "cellpose":
                per_slice["cellpose_wins"] += 1
            else:
                per_slice["ties"] += 1
            br_s_acc += br_s
            br_c_acc += br_c

        n = per_slice["n_gt"]
        per_slice["br_sicle_mean"] = round(br_s_acc / n, 4) if n else 0.0
        per_slice["br_cellpose_mean"] = round(br_c_acc / n, 4) if n else 0.0
        per_slice["pct_sicle_wins"] = round(100.0 * per_slice["sicle_wins"] / n, 2) if n else 0.0
        per_slice["pct_cellpose_wins"] = round(100.0 * per_slice["cellpose_wins"] / n, 2) if n else 0.0
        per_slice["pct_ties"] = round(100.0 * per_slice["ties"] / n, 2) if n else 0.0
        summary_rows.append(per_slice)

        with slice_csv.open("w", newline="", encoding="utf-8") as fp:
            field_per = [
                "slice", "gt_id", "gt_area", "bbox",
                "pred_label_sicle", "pred_label_cellpose",
                "br_sicle", "br_cellpose", "br_diff_sicle_minus_cp",
                "tp_sicle", "fn_sicle", "tp_cellpose", "fn_cellpose", "winner",
            ]
            w = csv.DictWriter(fp, fieldnames=field_per)
            w.writeheader()
            for r in slice_rows:
                w.writerow(r)

        print(
            f"[{stem}] n_gt={n}  SICLE={per_slice['sicle_wins']} "
            f"CP={per_slice['cellpose_wins']} tie={per_slice['ties']}  "
            f"BR_S={per_slice['br_sicle_mean']:.4f}  "
            f"BR_C={per_slice['br_cellpose_mean']:.4f}"
        )

        overall["n_gt"] += n
        overall["s_wins"] += per_slice["sicle_wins"]
        overall["c_wins"] += per_slice["cellpose_wins"]
        overall["ties"] += per_slice["ties"]
        overall["br_s_sum"] += br_s_acc
        overall["br_c_sum"] += br_c_acc

    field_all = [
        "slice", "gt_id", "gt_area", "bbox",
        "pred_label_sicle", "pred_label_cellpose",
        "br_sicle", "br_cellpose", "br_diff_sicle_minus_cp",
        "tp_sicle", "fn_sicle", "tp_cellpose", "fn_cellpose", "winner",
    ]
    with (br_dir / "per_cell_br_all.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=field_all)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    overall_row = {
        "slice": "ALL",
        "n_gt": overall["n_gt"],
        "sicle_wins": overall["s_wins"],
        "cellpose_wins": overall["c_wins"],
        "ties": overall["ties"],
        "pct_sicle_wins": round(100.0 * overall["s_wins"] / overall["n_gt"], 2) if overall["n_gt"] else 0.0,
        "pct_cellpose_wins": round(100.0 * overall["c_wins"] / overall["n_gt"], 2) if overall["n_gt"] else 0.0,
        "pct_ties": round(100.0 * overall["ties"] / overall["n_gt"], 2) if overall["n_gt"] else 0.0,
        "br_sicle_mean": round(overall["br_s_sum"] / overall["n_gt"], 4) if overall["n_gt"] else 0.0,
        "br_cellpose_mean": round(overall["br_c_sum"] / overall["n_gt"], 4) if overall["n_gt"] else 0.0,
    }
    field_sum = [
        "slice", "n_gt", "sicle_wins", "cellpose_wins", "ties",
        "pct_sicle_wins", "pct_cellpose_wins", "pct_ties",
        "br_sicle_mean", "br_cellpose_mean",
    ]
    with (br_dir / "per_cell_br_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=field_sum)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in field_sum})
        w.writerow(overall_row)

    print()
    mode = "legacy full-crop" if args.legacy_full_crop else "strict per-cell (best pred instance)"
    print(f"== Per-cell Boundary Recall summary (eps={args.tie_eps}, {mode}) ==")
    print(f"{'slice':<25} {'n_gt':>5} {'SICLE':>7} {'CP':>7} {'tie':>6}  {'%SICLE':>7} {'%CP':>6}  BR_S/CP")
    for r in summary_rows:
        print(
            f"{r['slice']:<25} {r['n_gt']:>5d} "
            f"{r['sicle_wins']:>7d} {r['cellpose_wins']:>7d} {r['ties']:>6d}  "
            f"{r['pct_sicle_wins']:>6.2f}% {r['pct_cellpose_wins']:>5.2f}%  "
            f"{r['br_sicle_mean']:.4f}/{r['br_cellpose_mean']:.4f}"
        )
    print(
        f"{'ALL':<25} {overall_row['n_gt']:>5d} "
        f"{overall_row['sicle_wins']:>7d} {overall_row['cellpose_wins']:>7d} {overall_row['ties']:>6d}  "
        f"{overall_row['pct_sicle_wins']:>6.2f}% {overall_row['pct_cellpose_wins']:>5.2f}%  "
        f"{overall_row['br_sicle_mean']:.4f}/{overall_row['br_cellpose_mean']:.4f}"
    )
    print(f"\nWrote {br_dir / 'per_cell_br_all.csv'}")
    print(f"Wrote {br_dir / 'per_cell_br_summary.csv'}")
    print(f"Crops under {br_dir}/<slice>/{{sicle_wins,cellpose_wins,ties}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
