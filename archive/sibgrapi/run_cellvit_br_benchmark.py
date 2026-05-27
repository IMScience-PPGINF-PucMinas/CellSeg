#!/usr/bin/env python3
"""
Benchmark: Cellpose alone, CellViT alone, SICLE-on-CellViT (no BR pick), BR-merge (Cellpose vs SICLE).

Writes per-slice folders under ``--out-root`` and a summary with metrics + wall-clock times.

Example::

    python3 run_cellvit_br_benchmark.py \\
        --checkpoint /path/to/CellViT-256/model_best.pth \\
        --out-root ./out_cellvit_br \\
        --reuse-cellpose-dir ./out_sibgrapi2026
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PY = sys.executable


def _run(cmd: list[str], cwd: Path | None = None) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd or _HERE), check=True)
    return time.perf_counter() - t0


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    import os

    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=True)
    except (OSError, AttributeError, shutil.Error):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _eval_method(out_root: Path, stem: str, gt_path: Path, pred_rel: Path, method: str) -> dict:
    import numpy as np

    sys.path.insert(0, str(_HERE))
    from evaluate_sibgrapi2026 import evaluate_pair

    pr = out_root / stem / pred_rel
    row = evaluate_pair(gt_path, pr)
    row["slice"] = stem
    row["method"] = method
    return row


def main() -> int:
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, default=str(_HERE / "out_cellvit_br"))
    p.add_argument("--data-dir", type=str, default=str(_HERE / "data_sibgrapi2026" / "data_sibgrapi2026"))
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("CELLVIT_CHECKPOINT", ""),
        help="CellViT model_best.pth (or set CELLVIT_CHECKPOINT)",
    )
    p.add_argument(
        "--reuse-cellpose-dir",
        type=str,
        default=str(_HERE / "out_sibgrapi2026"),
        help="Existing pipeline root with <stem>/cp_flow/ to skip Cellpose re-run",
    )
    p.add_argument("--gpu", type=int, default=1, help="1 = use GPU for Cellpose/CellViT")
    p.add_argument("--cellvit-gpu", type=int, default=0)
    p.add_argument("--skip-cellvit", action="store_true")
    p.add_argument("--skip-cellpose", action="store_true", help="Require reuse-cellpose-dir")
    p.add_argument("--max-slices", type=int, default=0, help="0 = all PNGs")
    args = p.parse_args()

    out_root = Path(args.out_root)
    data_dir = Path(args.data_dir)
    reuse_cp = Path(args.reuse_cellpose_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pngs = sorted(data_dir.glob("*.png"))
    if args.max_slices > 0:
        pngs = pngs[: args.max_slices]
    if not pngs:
        raise SystemExit(f"No PNGs in {data_dir}")

    timings: dict[str, float] = {}
    per_slice_times: dict[str, dict[str, float]] = {}

    # --- GT (once for all slices) ---
    t_gt = time.perf_counter()
    subprocess.run(
        [_PY, str(_HERE / "extract_slices_lab_gt.py"), "--out-root", str(out_root)],
        cwd=str(_HERE),
        check=True,
    )
    timings["extract_gt_total_s"] = time.perf_counter() - t_gt

    # --- Cellpose (all slices) ---
    if not args.skip_cellpose:
        t_cp = 0.0
        for png in pngs:
            stem = png.stem
            dst = out_root / stem / "cp_flow"
            src = reuse_cp / stem / "cp_flow"
            if src.is_dir() and (src / "step04_masks_uint16.npy").is_file():
                t0 = time.perf_counter()
                _link_or_copy_tree(src, dst)
                t_cp += time.perf_counter() - t0
                per_slice_times.setdefault(stem, {})["cellpose"] = time.perf_counter() - t0
                continue
            t0 = time.perf_counter()
            dst.parent.mkdir(parents=True, exist_ok=True)
            cmd = [_PY, str(_HERE / "reproduce_cellpose_pipeline.py"), str(png), "-o", str(dst)]
            if args.gpu:
                cmd.append("--gpu")
            subprocess.run(cmd, cwd=str(_HERE), check=True)
            dt = time.perf_counter() - t0
            t_cp += dt
            per_slice_times.setdefault(stem, {})["cellpose"] = dt
        timings["cellpose_total_s"] = t_cp
    else:
        for png in pngs:
            stem = png.stem
            src = reuse_cp / stem / "cp_flow"
            dst = out_root / stem / "cp_flow"
            if not src.is_dir():
                raise SystemExit(f"Missing {src}")
            _link_or_copy_tree(src, dst)

    # --- CellViT ---
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if args.skip_cellvit:
        timings["cellvit_total_s"] = 0.0
    elif ckpt is None or not ckpt.is_file():
        print("WARNING: No CellViT checkpoint — skipping CellViT + SICLE-on-CellViT + BR merge.")
        timings["cellvit_total_s"] = 0.0
        args.skip_cellvit = True
    else:
        t_cv = 0.0
        for png in pngs:
            stem = png.stem
            cv_dir = out_root / stem / "cellvit_flow"
            t0 = time.perf_counter()
            subprocess.run(
                [
                    _PY,
                    str(_HERE / "cellvit_infer_png.py"),
                    str(png),
                    "-o",
                    str(cv_dir),
                    "--checkpoint",
                    str(ckpt),
                    "--gpu",
                    str(args.cellvit_gpu),
                ],
                cwd=str(_HERE),
                check=True,
            )
            dt = time.perf_counter() - t0
            t_cv += dt
            per_slice_times.setdefault(stem, {})["cellvit"] = dt
        timings["cellvit_total_s"] = t_cv

    sicle_base = [
        "--sicle-conn-opt", "gradvmaxmul",
        "--sicle-crit-opt", "minsc",
        "--sicle-alpha", "2.0",
        "--sicle-nf", "2",
        "--sicle-n0", "200",
        "--sicle-irreg", "0.0",
        "--sicle-adhr", "1",
        "--sicle-max-iters", "7",
        "--saliency-threshold", "0.3",
        "--saliency-blur-sigma", "0.5",
        "--margin", "4",
        "--min-cell-area", "128",
        "--disable-and-merge",
        "--and-unless-round",
        "--min-fg-circularity", "0.70",
        "--min-fg-solidity", "0.85",
        "--fill-holes",
        "--keep-largest-cc",
        "--closing-radius", "1",
        "--overlay-border-source", "both",
        "--overlay-border-color", "0,255,0",
        "--overlay-cellpose-border-color", "255,255,0",
    ]

    t_sicle = 0.0
    t_br = 0.0
    if not args.skip_cellvit:
        for png in pngs:
            stem = png.stem
            case = out_root / stem
            cv_dir = case / "cellvit_flow"
            cp_dir = case / "cp_flow"
            sicle_dir = case / "sicle"
            t0 = time.perf_counter()
            subprocess.run(
                [
                    _PY,
                    str(_HERE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir",
                    str(cv_dir),
                    "--cellprob-from-dir",
                    str(cp_dir),
                    "-o",
                    str(sicle_dir),
                    "--image",
                    str(png),
                    *sicle_base,
                ],
                cwd=str(_HERE),
                check=True,
            )
            dt = time.perf_counter() - t0
            t_sicle += dt
            per_slice_times.setdefault(stem, {})["sicle_cellvit"] = dt

            gt_candidates = [
                case / "gt" / "macro_nuclick_masks_int32.npy",
                case / "gt" / "union_masks_int32.npy",
                case / "gt" / "nuclick_masks_int32.npy",
            ]
            gt_path = next((g for g in gt_candidates if g.is_file()), None)
            if gt_path is None:
                raise SystemExit(f"[{stem}] GT not found under {case / 'gt'}")

            br_dir = case / "br_merge"
            br_dir.mkdir(parents=True, exist_ok=True)
            t1 = time.perf_counter()
            subprocess.run(
                [
                    _PY,
                    str(_HERE / "build_br_merged_masks.py"),
                    "--gt",
                    str(gt_path),
                    "--sicle",
                    str(sicle_dir / "merged_percell_sicle_masks_int32.npy"),
                    "--cellpose",
                    str(cp_dir / "step04_masks_uint16.npy"),
                    "-o",
                    str(br_dir / "merged_br_pick_masks_int32.npy"),
                    "--csv",
                    str(br_dir / "per_cell_br_pick.csv"),
                ],
                cwd=str(_HERE),
                check=True,
            )
            dt_br = time.perf_counter() - t1
            t_br += dt_br
            per_slice_times.setdefault(stem, {})["br_merge"] = dt_br

    timings["sicle_on_cellvit_total_s"] = t_sicle
    timings["br_merge_total_s"] = t_br

    # --- Evaluation ---
    t_eval = time.perf_counter()
    metrics_rows: list[dict] = []
    for png in pngs:
        stem = png.stem
        case = out_root / stem
        gt_candidates = [
            case / "gt" / "macro_nuclick_masks_int32.npy",
            case / "gt" / "union_masks_int32.npy",
            case / "gt" / "nuclick_masks_int32.npy",
        ]
        gt_path = next((g for g in gt_candidates if g.is_file()), None)
        if gt_path is None:
            continue

        methods = [
            ("cellpose", Path("cp_flow/step04_masks_uint16.npy")),
        ]
        if not args.skip_cellvit and (case / "cellvit_flow/step04_masks_uint16.npy").is_file():
            methods.append(("cellvit", Path("cellvit_flow/step04_masks_uint16.npy")))
            methods.append(("sicle_cellvit", Path("sicle/merged_percell_sicle_masks_int32.npy")))
            methods.append(("br_pick_cp_vs_sicle", Path("br_merge/merged_br_pick_masks_int32.npy")))

        for mname, rel in methods:
            metrics_rows.append(_eval_method(out_root, stem, gt_path, rel, mname))

    timings["evaluate_total_s"] = time.perf_counter() - t_eval

    # Macro averages
    import numpy as np

    summary: dict[str, dict] = {}
    for m in sorted({r["method"] for r in metrics_rows}):
        sub = [r for r in metrics_rows if r["method"] == m and "error" not in r]
        if not sub:
            continue
        summary[m] = {
            "n_slices": len(sub),
            "dice_mean": float(np.mean([r["pixel_dice"] for r in sub])),
            "aji_mean": float(np.mean([r["aji"] for r in sub])),
            "pq_mean": float(np.mean([r["pq"] for r in sub])),
            "f1_mean": float(np.mean([r["f1_0.5"] for r in sub])),
            "map_dsb_mean": float(np.mean([r["map_dsb"] for r in sub])),
        }

    out_json = out_root / "benchmark_timings.json"
    out_txt = out_root / "benchmark_summary.txt"
    payload = {
        "timings_s": timings,
        "per_slice_s": per_slice_times,
        "metrics_per_slice": metrics_rows,
        "metrics_macro": summary,
        "checkpoint": str(ckpt) if ckpt else None,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "CellViT + BR benchmark",
        f"out_root: {out_root}",
        f"checkpoint: {ckpt}",
        "",
        "=== Wall-clock (seconds) ===",
    ]
    for k, v in sorted(timings.items()):
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")
    lines.append("=== Macro metrics (mean over slices) ===")
    lines.append(f"{'method':<22} {'Dice':>7} {'AJI':>7} {'PQ':>7} {'F1':>7} {'mAP':>7}")
    for m, s in summary.items():
        lines.append(
            f"{m:<22} {s['dice_mean']:7.4f} {s['aji_mean']:7.4f} {s['pq_mean']:7.4f} "
            f"{s['f1_mean']:7.4f} {s['map_dsb_mean']:7.4f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
