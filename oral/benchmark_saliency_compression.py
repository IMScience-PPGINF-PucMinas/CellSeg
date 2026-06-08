#!/usr/bin/env python3
"""Compare SICLE results with vs without cellprob saliency compression (Otsu + threshold)."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import (
    SICLE_COMMON,
    colored_to_labels,
    mean_br_strict,
)

OUT_ROOT = RUNS / "path_cost_benchmark" / "saliency_compression"

# gradvmaxmul + minsc fixed; vary saliency preprocessing only
BASE_SICLE = [
    "--sicle-conn-opt", "gradvmaxmul",
    "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0",
]

SALIENCY_VARIANTS: list[dict] = [
    {
        "id": "compress_blur05",
        "label": "Otsu + thr 0.3 + blur 0.5 (atual)",
        "extra": [],
    },
    {
        "id": "nolin_blur05",
        "label": "sigmoid + thr 0.3 + blur 0.5 (sem Otsu)",
        "extra": ["--no-saliency-linearize"],
    },
    {
        "id": "nolin_noblur",
        "label": "sigmoid + thr 0.3 (sem Otsu, sem blur)",
        "extra": ["--no-saliency-linearize", "--saliency-blur-sigma", "0"],
    },
    {
        "id": "compress_noblur",
        "label": "Otsu + thr 0.3 (sem blur)",
        "extra": ["--saliency-blur-sigma", "0"],
    },
    {
        "id": "nolin_nothr_blur05",
        "label": "sigmoid + blur 0.5 (sem Otsu, sem thr)",
        "extra": ["--no-saliency-linearize", "--saliency-threshold", "0"],
    },
]

ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("severe", "severe-03-roi2"),
]


def run() -> list[dict]:
    from PIL import Image
    import os

    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    bench_root = RUNS / "path_cost_benchmark"
    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    rows: list[dict] = []
    for category, stem in ROIS:
        case_bench = bench_root / category / stem
        case_out = OUT_ROOT / category / stem
        case_out.mkdir(parents=True, exist_ok=True)

        input_png = case_bench / f"{stem}.png"
        cp_dir = case_bench / "cp_flow"
        gt_path = case_bench / "gt" / "gold_standard_masks_int32.npy"
        if not gt_path.is_file():
            from _paths import GT_COLORED, IMAGES_ORIGINAL

            col = np.asarray(Image.open(GT_COLORED / category / f"{stem}.png").convert("RGB"))
            orig = np.asarray(Image.open(IMAGES_ORIGINAL / category / f"{stem}.tif").convert("RGB"))
            h, w = min(col.shape[0], orig.shape[0]), min(col.shape[1], orig.shape[1])
            Image.fromarray(orig[:h, :w]).save(input_png)
            np.save(gt_path, colored_to_labels(col[:h, :w]))

        gt_arr = np.load(gt_path).astype(np.int32)
        print(f"\n=== {category}/{stem} ===")

        for var in SALIENCY_VARIANTS:
            sid = var["id"]
            sicle_dir = case_out / sid
            if not (sicle_dir / "merged_percell_sicle_masks_int32.npy").is_file():
                print(f"  {sid}")
                cmd = [
                    py,
                    str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                    "--from-dir",
                    str(cp_dir),
                    "-o",
                    str(sicle_dir),
                    *BASE_SICLE,
                    *var["extra"],
                    "--image",
                    str(input_png),
                    *SICLE_COMMON,
                ]
                subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

            pr_path = sicle_dir / "merged_percell_sicle_masks_int32.npy"
            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            rows.append(
                {
                    "category": category,
                    "roi": stem,
                    "variant_id": sid,
                    "variant_label": var["label"],
                    "br_mean_strict": br,
                    "pixel_dice": r.get("pixel_dice"),
                    "aji": r.get("aji"),
                }
            )
            print(f"    {sid}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    return rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = run()
    csv_path = OUT_ROOT / "metrics_saliency_compression.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary markdown
    lines = [
        "# Saliência: com vs sem compressão (Otsu)",
        "",
        "Fixo: gradvmaxmul + minsc + pós-processo blur05.",
        "",
        "| ROI | variante | BR | Dice |",
        "|-----|----------|---:|-----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['roi']} | {r['variant_id']} | {r['br_mean_strict']:.4f} | {r['pixel_dice']:.4f} |"
        )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
