#!/usr/bin/env python3
"""
Compare Cellpose and NuClick to MoNuSeg XML-derived GT (annotation_labels).

Uses *binary* nucleus foreground Dice (all instances merged):
  Dice = 2|GT ∩ Pred| / (|GT| + |Pred|)

If prediction shape ≠ GT shape, the prediction foreground is resized with nearest-neighbor
to match GT (whole-slide vs same resolution).

Usage:
  python monuseg_dice_eval.py
  python monuseg_dice_eval.py --run-dir monuseg_runs/exp1 --csv monuseg_runs/exp1/dice_results.csv

Each row in the CSV and the printed table is one image (foreground Dice vs GT).
Use --no-summary to print only per-image lines without mean/std at the end.

LaTeX:
  python monuseg_dice_eval.py --latex-out monuseg_runs/exp1/dice_table.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "monuseg_runs" / "exp1"


def load_label_png(path: Path) -> np.ndarray:
    """Instance or label PNG -> numpy (H,W), any positive = foreground."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def foreground_binary(mask: np.ndarray) -> np.ndarray:
    return mask > 0


def resize_fg_to(fg: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize boolean foreground mask to (H, W) with nearest neighbor."""
    h, w = shape_hw
    if fg.shape == (h, w):
        return fg
    u8 = (fg.astype(np.uint8)) * 255
    im = Image.fromarray(u8, mode="L")
    im = im.resize((w, h), Image.NEAREST)
    return np.asarray(im) > 0


def dice_binary(gt_fg: np.ndarray, pred_fg: np.ndarray) -> float:
    """Sørensen–Dice on boolean masks."""
    pred_fg = resize_fg_to(pred_fg, gt_fg.shape)
    inter = np.logical_and(gt_fg, pred_fg).sum()
    s = gt_fg.sum() + pred_fg.sum()
    if s == 0:
        return 1.0 if inter == 0 else 0.0
    return float((2.0 * inter) / s)


def find_cellpose_pred(cellpose_dir: Path, stem: str) -> Path | None:
    for name in (
        f"{stem}_cellpose_instances.png",
        f"{stem}_cellpose_labels.png",
    ):
        p = cellpose_dir / name
        if p.is_file():
            return p
    return None


def find_nuclick_pred(nuclick_dir: Path, stem: str) -> Path | None:
    p = nuclick_dir / f"{stem}_dots_instances.png"
    return p if p.is_file() else None


def latex_escape(s: str) -> str:
    """Escape image id for LaTeX text mode."""
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "_":
            out.append("\\_")
        elif ch == "%":
            out.append("\\%")
        elif ch == "&":
            out.append("\\&")
        elif ch == "#":
            out.append("\\#")
        elif ch == "$":
            out.append("\\$")
        elif ch in "{}~^":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def write_latex_document(
    path: Path,
    rows: list[dict[str, Any]],
    mean_cp: float,
    mean_nc: float,
    n_cp: int,
    n_nc: int,
) -> None:
    """Write a compilable .tex file with booktabs; \\textbf{} marks the better Dice per row."""
    lines: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=2cm,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\title{MoNuSeg foreground Dice: Cellpose vs NuClick}",
        r"\author{}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent\textbf{Note:} Binary foreground Dice vs expert XML-derived labels. ",
        r"\textbf{Bold} = higher score on that row (tie: both bold).",
        r"\par\medskip",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Per-image Dice coefficient (vs ground truth).}",
        r"\label{tab:dice-cellpose-nuclick}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Image ID} & \textbf{Cellpose} & \textbf{NuClick} \\",
        r"\midrule",
    ]

    for r in rows:
        stem = str(r["image"])
        cp_f = float(r["dice_cellpose"])
        nc_f = float(r["dice_nuclick"])
        if math.isnan(cp_f) and math.isnan(nc_f):
            c_cp, c_nc = "---", "---"
        elif math.isnan(cp_f):
            c_cp, c_nc = "---", f"\\textbf{{{nc_f:.4f}}}"
        elif math.isnan(nc_f):
            c_cp, c_nc = f"\\textbf{{{cp_f:.4f}}}", "---"
        elif abs(cp_f - nc_f) < 1e-6:
            c_cp = f"\\textbf{{{cp_f:.4f}}}"
            c_nc = f"\\textbf{{{nc_f:.4f}}}"
        elif cp_f > nc_f:
            c_cp, c_nc = f"\\textbf{{{cp_f:.4f}}}", f"{nc_f:.4f}"
        else:
            c_cp, c_nc = f"{cp_f:.4f}", f"\\textbf{{{nc_f:.4f}}}"

        lines.append(f"{latex_escape(stem)} & {c_cp} & {c_nc} \\\\")

    lines.append(r"\midrule")
    if not math.isnan(mean_cp) and not math.isnan(mean_nc):
        if abs(mean_cp - mean_nc) < 1e-6:
            mc = f"\\textbf{{{mean_cp:.4f}}}"
            mn = f"\\textbf{{{mean_nc:.4f}}}"
        elif mean_cp > mean_nc:
            mc = f"\\textbf{{{mean_cp:.4f}}}"
            mn = f"{mean_nc:.4f}"
        else:
            mc = f"{mean_cp:.4f}"
            mn = f"\\textbf{{{mean_nc:.4f}}}"
    else:
        mc = f"{mean_cp:.4f}" if not math.isnan(mean_cp) else "---"
        mn = f"{mean_nc:.4f}" if not math.isnan(mean_nc) else "---"

    lines.append(
        f"\\textit{{Mean}} ($n$={max(n_cp, n_nc)}) & {mc} & {mn} \\\\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\end{document}",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dice vs MoNuSeg GT for Cellpose / NuClick")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Experiment folder (contains annotation_labels, cellpose, nuclick_instances)",
    )
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Override GT folder (default: <run-dir>/annotation_labels)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write per-image CSV (default: <run-dir>/dice_cellpose_nuclick_vs_gt.csv)",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print mean/std over all images (only per-image table + paths)",
    )
    ap.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Write a compilable LaTeX file (default: <run-dir>/dice_cellpose_nuclick_vs_gt.tex)",
    )
    ap.add_argument(
        "--no-latex",
        action="store_true",
        help="Do not write the .tex file (default is to write when --latex-out is default path)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    gt_dir = (args.gt_dir if args.gt_dir is not None else run_dir / "annotation_labels").resolve()
    cellpose_dir = run_dir / "cellpose"
    nuclick_dir = run_dir / "nuclick_instances"
    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else run_dir / "dice_cellpose_nuclick_vs_gt.csv"
    )

    if not gt_dir.is_dir():
        print(f"ERROR: GT folder not found: {gt_dir}", file=sys.stderr)
        return 1

    gt_files = sorted(gt_dir.glob("*_labels.png"))
    if not gt_files:
        print(f"No *_labels.png in {gt_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, str | float]] = []
    cp_vals: list[float] = []
    nc_vals: list[float] = []

    for gt_path in gt_files:
        stem = gt_path.name[: -len("_labels.png")]
        gt = load_label_png(gt_path)
        gt_fg = foreground_binary(gt)

        row: dict[str, str | float] = {"image": stem}

        cp_path = find_cellpose_pred(cellpose_dir, stem)
        if cp_path is None:
            row["dice_cellpose"] = float("nan")
        else:
            pred = load_label_png(cp_path)
            d = dice_binary(gt_fg, foreground_binary(pred))
            row["dice_cellpose"] = d
            cp_vals.append(d)

        nc_path = find_nuclick_pred(nuclick_dir, stem)
        if nc_path is None:
            row["dice_nuclick"] = float("nan")
        else:
            pred = load_label_png(nc_path)
            d = dice_binary(gt_fg, foreground_binary(pred))
            row["dice_nuclick"] = d
            nc_vals.append(d)

        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "dice_cellpose", "dice_nuclick"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "image": r["image"],
                    "dice_cellpose": r["dice_cellpose"],
                    "dice_nuclick": r["dice_nuclick"],
                }
            )

    def fmt_dice(x: float) -> str:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{float(x):.4f}"

    print(f"GT:        {gt_dir}  ({len(gt_files)} images)")
    print(f"Cellpose:  {cellpose_dir}")
    print(f"NuClick:   {nuclick_dir}")
    print(f"CSV:       {csv_path}")
    print()
    print("Per-image foreground Dice (vs GT)")
    col_w = max(len(str(r["image"])) for r in rows) if rows else 40
    col_w = max(col_w, 36)
    hdr = f"{'image':<{col_w}}  {'dice_cellpose':>14}  {'dice_nuclick':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['image']:<{col_w}}  {fmt_dice(r['dice_cellpose']):>14}  {fmt_dice(r['dice_nuclick']):>14}"
        )

    if not args.no_summary:

        def mean_std(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            a = np.array(vals, dtype=np.float64)
            return float(a.mean()), float(a.std(ddof=0))

        m_cp, s_cp = mean_std(cp_vals)
        m_nc, s_nc = mean_std(nc_vals)
        print()
        print(
            f"Summary over images — Cellpose: mean={m_cp:.4f}  std={s_cp:.4f}  (n={len(cp_vals)})"
        )
        print(
            f"                      NuClick:  mean={m_nc:.4f}  std={s_nc:.4f}  (n={len(nc_vals)})"
        )

    latex_path = None if args.no_latex else (
        args.latex_out.resolve()
        if args.latex_out is not None
        else run_dir / "dice_cellpose_nuclick_vs_gt.tex"
    )
    if latex_path is not None:
        def mean_std(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            a = np.array(vals, dtype=np.float64)
            return float(a.mean()), float(a.std(ddof=0))

        m_cp_l, _ = mean_std(cp_vals)
        m_nc_l, _ = mean_std(nc_vals)
        write_latex_document(
            latex_path,
            rows,
            mean_cp=m_cp_l,
            mean_nc=m_nc_l,
            n_cp=len(cp_vals),
            n_nc=len(nc_vals),
        )
        print()
        print(f"LaTeX:     {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
