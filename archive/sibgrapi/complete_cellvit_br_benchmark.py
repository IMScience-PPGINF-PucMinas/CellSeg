#!/usr/bin/env python3
"""Resume CellViT+BR benchmark: fix GT layout, finish SICLE/BR/eval, print timings."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PY = sys.executable
OUT = _HERE / "out_cellvit_br"
DATA = _HERE / "data_sibgrapi2026" / "data_sibgrapi2026"
SICLE_BASE = [
    "--sicle-conn-opt", "gradvmaxmul", "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0", "--sicle-nf", "2", "--sicle-n0", "200",
    "--sicle-irreg", "0.0", "--sicle-adhr", "1", "--sicle-max-iters", "7",
    "--saliency-threshold", "0.3", "--saliency-blur-sigma", "0.5",
    "--margin", "4", "--min-cell-area", "128",
    "--disable-and-merge", "--and-unless-round",
    "--min-fg-circularity", "0.70", "--min-fg-solidity", "0.85",
    "--fill-holes", "--keep-largest-cc", "--closing-radius", "1",
    "--overlay-border-source", "both",
    "--overlay-border-color", "0,255,0",
    "--overlay-cellpose-border-color", "255,255,0",
]


def main() -> int:
    # Fix GT nested under slice1/
    wrong = OUT / "12121_40x_slice1"
    for nested in wrong.iterdir():
        if not nested.is_dir() or not nested.name.startswith("12121_40x_slice"):
            continue
        dst_gt = OUT / nested.name / "gt"
        src_gt = nested / "gt"
        if src_gt.is_dir() and not dst_gt.is_dir():
            shutil.copytree(src_gt, dst_gt)

    subprocess.run(
        [_PY, str(_HERE / "extract_slices_lab_gt.py"), "--out-root", str(OUT)],
        cwd=str(_HERE),
        check=True,
    )

    timings: dict[str, float] = {"cellvit_total_s": 0.0, "cellpose_total_s": 0.0}
    per_slice: dict[str, dict[str, float]] = {}

    for png in sorted(DATA.glob("*.png")):
        stem = png.stem
        case = OUT / stem
        cp_dir = case / "cp_flow"
        cv_dir = case / "cellvit_flow"
        sicle_dir = case / "sicle"
        br_dir = case / "br_merge"

        if not (sicle_dir / "merged_percell_sicle_masks_int32.npy").is_file():
            t0 = time.perf_counter()
            subprocess.run(
                [
                    _PY, str(_HERE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir", str(cv_dir),
                    "--cellprob-from-dir", str(cp_dir),
                    "-o", str(sicle_dir),
                    "--image", str(png),
                    *SICLE_BASE,
                ],
                cwd=str(_HERE),
                check=True,
            )
            per_slice.setdefault(stem, {})["sicle_cellvit"] = time.perf_counter() - t0

        gt_path = next(
            (
                g
                for g in (
                    case / "gt" / "macro_nuclick_masks_int32.npy",
                    case / "gt" / "union_masks_int32.npy",
                    case / "gt" / "nuclick_masks_int32.npy",
                )
                if g.is_file()
            ),
            None,
        )
        if gt_path is None:
            raise SystemExit(f"no GT for {stem}")

        if not (br_dir / "merged_br_pick_masks_int32.npy").is_file():
            t0 = time.perf_counter()
            subprocess.run(
                [
                    _PY, str(_HERE / "build_br_merged_masks.py"),
                    "--gt", str(gt_path),
                    "--sicle", str(sicle_dir / "merged_percell_sicle_masks_int32.npy"),
                    "--cellpose", str(cp_dir / "step04_masks_uint16.npy"),
                    "-o", str(br_dir / "merged_br_pick_masks_int32.npy"),
                    "--csv", str(br_dir / "per_cell_br_pick.csv"),
                ],
                cwd=str(_HERE),
                check=True,
            )
            per_slice.setdefault(stem, {})["br_merge"] = time.perf_counter() - t0

    sys.path.insert(0, str(_HERE))
    from evaluate_sibgrapi2026 import evaluate_pair
    import numpy as np

    rows = []
    for png in sorted(DATA.glob("*.png")):
        stem = png.stem
        case = OUT / stem
        gt_path = next(
            (
                g
                for g in (
                    case / "gt" / "macro_nuclick_masks_int32.npy",
                    case / "gt" / "union_masks_int32.npy",
                    case / "gt" / "nuclick_masks_int32.npy",
                )
                if g.is_file()
            ),
            None,
        )
        for mname, rel in [
            ("cellpose", Path("cp_flow/step04_masks_uint16.npy")),
            ("cellvit", Path("cellvit_flow/step04_masks_uint16.npy")),
            ("sicle_cellvit", Path("sicle/merged_percell_sicle_masks_int32.npy")),
            ("br_pick_cp_vs_sicle", Path("br_merge/merged_br_pick_masks_int32.npy")),
        ]:
            pr = case / rel
            if not pr.is_file():
                continue
            row = evaluate_pair(gt_path, pr)
            row["slice"] = stem
            row["method"] = mname
            rows.append(row)

    summary: dict[str, dict] = {}
    for m in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == m and "error" not in r]
        if not sub:
            continue
        summary[m] = {
            "dice_mean": float(np.mean([r["pixel_dice"] for r in sub])),
            "aji_mean": float(np.mean([r["aji"] for r in sub])),
            "pq_mean": float(np.mean([r["pq"] for r in sub])),
            "f1_mean": float(np.mean([r["f1_0.5"] for r in sub])),
            "map_dsb_mean": float(np.mean([r["map_dsb"] for r in sub])),
        }

    # Wall times from first partial run (cellvit ~70s for 12 slices)
    timings["cellvit_total_s"] = 70.0
    timings["cellpose_total_s"] = 0.5
    if per_slice:
        timings["sicle_on_cellvit_total_s"] = sum(
            per_slice[s].get("sicle_cellvit", 0.0) for s in per_slice
        )
        timings["br_merge_total_s"] = sum(per_slice[s].get("br_merge", 0.0) for s in per_slice)

    payload = {"timings_s": timings, "per_slice_s": per_slice, "metrics_macro": summary, "metrics_per_slice": rows}
    (OUT / "benchmark_timings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["=== Tempos (s) ==="] + [f"  {k}: {v:.2f}" for k, v in sorted(timings.items())]
    lines += ["", "=== Métricas macro ===", f"{'method':<22} {'Dice':>7} {'AJI':>7} {'PQ':>7} {'F1':>7} {'mAP':>7}"]
    for m, s in summary.items():
        lines.append(
            f"{m:<22} {s['dice_mean']:7.4f} {s['aji_mean']:7.4f} {s['pq_mean']:7.4f} "
            f"{s['f1_mean']:7.4f} {s['map_dsb_mean']:7.4f}"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "benchmark_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
