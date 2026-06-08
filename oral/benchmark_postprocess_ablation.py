#!/usr/bin/env python3
"""Ablation: full post-process vs no morph / raw merge / AND-only (gradvmaxmul+minsc, no Otsu)."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import colored_to_labels, mean_br_strict

OUT_ROOT_SAMPLE = RUNS / "postprocess_ablation"
OUT_ROOT_FULL = RUNS / "postprocess_ablation_full"

SAMPLE_ROIS = [
    ("healthy", "healthy-18-roi2"),
    ("healthy", "healthy-19-roi2"),
    ("healthy", "healthy-17-roi2"),
    ("severe", "severe-03-roi2"),
]

BASE = [
    "--no-saliency-linearize",
    "--sicle-conn-opt", "gradvmaxmul",
    "--sicle-crit-opt", "minsc",
    "--sicle-alpha", "2.0",
    "--saliency-threshold", "0.3",
    "--saliency-blur-sigma", "0.5",
    "--margin", "4",
    "--min-cell-area", "128",
    "--sicle-n0", "200",
    "--sicle-nf", "2",
    "--sicle-irreg", "0",
    "--sicle-adhr", "1",
    "--sicle-max-iters", "7",
]

VARIANTS: list[dict] = [
    {
        "id": "full",
        "label": "Completo (atual)",
        "extra": [
            "--disable-and-merge",
            "--and-unless-round",
            "--min-fg-circularity", "0.70",
            "--min-fg-solidity", "0.85",
            "--fill-holes",
            "--keep-largest-cc",
            "--closing-radius", "1",
        ],
    },
    {
        "id": "no_morph_full_merge",
        "label": "Sem morfologia; merge AUR igual",
        "extra": [
            "--disable-and-merge",
            "--and-unless-round",
            "--min-fg-circularity", "0.70",
            "--min-fg-solidity", "0.85",
            "--closing-radius", "0",
        ],
    },
    {
        "id": "sicle_raw",
        "label": "Sem morfologia; SICLE cru no bbox",
        "extra": ["--disable-and-merge", "--closing-radius", "0"],
    },
    {
        "id": "and_only",
        "label": "Sem morfologia; SICLE ∧ Cellpose",
        "extra": ["--closing-radius", "0"],
    },
    {
        "id": "morph_and_only",
        "label": "Com morfologia; SICLE ∧ Cellpose",
        "extra": ["--fill-holes", "--keep-largest-cc", "--closing-radius", "1"],
    },
]

FIELDNAMES = [
    "category",
    "roi",
    "variant_id",
    "label",
    "br_mean_strict",
    "pixel_dice",
    "aji",
    "n_gt",
    "n_pr",
]


def discover_rois() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for category in ("healthy", "severe"):
        col_dir = GT_COLORED / category
        if not col_dir.is_dir():
            continue
        for col_path in sorted(col_dir.glob("*.png")):
            stem = col_path.stem
            tif = IMAGES_ORIGINAL / category / f"{stem}.tif"
            if tif.is_file():
                out.append((category, stem))
    return out


def _ensure_case(
    category: str,
    stem: str,
    case: Path,
    *,
    py: str,
    env: dict,
    skip_cellpose: bool,
) -> tuple[Path, Path, Path]:
    from PIL import Image

    orig_tif = IMAGES_ORIGINAL / category / f"{stem}.tif"
    col_png = GT_COLORED / category / f"{stem}.png"
    case.mkdir(parents=True, exist_ok=True)

    rgb_orig = np.asarray(Image.open(orig_tif).convert("RGB"))
    rgb_col = np.asarray(Image.open(col_png).convert("RGB"))
    h = min(rgb_orig.shape[0], rgb_col.shape[0])
    w = min(rgb_orig.shape[1], rgb_col.shape[1])
    rgb_orig, rgb_col = rgb_orig[:h, :w], rgb_col[:h, :w]

    input_png = case / f"{stem}.png"
    if not input_png.is_file():
        Image.fromarray(rgb_orig).save(input_png)

    gt_path = case / "gt" / "gold_standard_masks_int32.npy"
    gt_path.parent.mkdir(exist_ok=True)
    if not gt_path.is_file():
        np.save(gt_path, colored_to_labels(rgb_col))

    cp_dir = case / "cp_flow"
    if not skip_cellpose and not (cp_dir / "step04_masks_uint16.npy").is_file():
        subprocess.run(
            [py, str(PIPE / "reproduce_cellpose_pipeline.py"), str(input_png), "-o", str(cp_dir), "--gpu"],
            cwd=str(REPO),
            env=env,
            check=True,
        )

    return input_png, gt_path, cp_dir


def _load_done_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    with csv_path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            done.add((row["category"], row["roi"], row["variant_id"]))
    return done


def _append_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _write_summary(rows: list[dict], out_root: Path, csv_path: Path, n_rois: int) -> None:
    by_var: dict[str, list[float]] = defaultdict(list)
    by_var_dice: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_var[r["variant_id"]].append(float(r["br_mean_strict"]))
        if r.get("pixel_dice") is not None:
            by_var_dice[r["variant_id"]].append(float(r["pixel_dice"]))

    lines = [
        "# Ablation pós-processamento (base completa)",
        "",
        f"ROIs: **{n_rois}** (healthy + severe). SICLE: gradvmaxmul + minsc, sem Otsu, blur σ=0.5.",
        "",
        f"CSV: `{csv_path.relative_to(REPO)}`",
        "",
        "## BR médio (strict)",
        "",
        "| Variante | BR médio | Dice médio |",
        "|----------|--------:|-----------:|",
    ]
    full_br = float(np.mean(by_var.get("full", [float("nan")])))
    for var in VARIANTS:
        vid = var["id"]
        br_m = float(np.mean(by_var[vid])) if by_var[vid] else float("nan")
        dice_m = float(np.mean(by_var_dice[vid])) if by_var_dice[vid] else float("nan")
        delta = br_m - full_br if vid != "full" and not np.isnan(br_m) else 0.0
        extra = f" (Δ vs full {delta:+.4f})" if vid != "full" else ""
        lines.append(f"| `{vid}` | {br_m:.4f}{extra} | {dice_m:.4f} |")

    lines.extend(
        [
            "",
            "## Por categoria (BR)",
            "",
        ]
    )
    for cat in ("healthy", "severe"):
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Variante | BR médio |")
        lines.append("|----------|--------:|")
        for var in VARIANTS:
            vid = var["id"]
            vals = [
                float(r["br_mean_strict"])
                for r in rows
                if r["category"] == cat and r["variant_id"] == vid
            ]
            lines.append(f"| `{vid}` | {np.mean(vals):.4f} |" if vals else f"| `{vid}` | — |")
        lines.append("")

    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    rois: list[tuple[str, str]],
    out_root: Path,
    *,
    skip_cellpose: bool,
    bench_cp_root: Path | None,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metrics_postprocess.csv"

    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    done = _load_done_keys(csv_path)
    rows: list[dict] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))

    n_total = len(rois) * len(VARIANTS)
    n_done_start = len(done)

    for i_roi, (category, stem) in enumerate(rois, 1):
        case = out_root / category / stem
        bench_case = (bench_cp_root / category / stem) if bench_cp_root else None
        if bench_case and (bench_case / "cp_flow" / "step04_masks_uint16.npy").is_file():
            case.mkdir(parents=True, exist_ok=True)
            input_png = bench_case / f"{stem}.png"
            gt_path = bench_case / "gt" / "gold_standard_masks_int32.npy"
            cp_dir = bench_case / "cp_flow"
            if not input_png.is_file() or not gt_path.is_file():
                input_png, gt_path, cp_dir = _ensure_case(
                    category, stem, case, py=py, env=env, skip_cellpose=skip_cellpose
                )
        else:
            input_png, gt_path, cp_dir = _ensure_case(
                category, stem, case, py=py, env=env, skip_cellpose=skip_cellpose
            )

        gt_arr = np.load(gt_path).astype(np.int32)
        print(f"\n[{i_roi}/{len(rois)}] === {category}/{stem} ===")

        for var in VARIANTS:
            sid = var["id"]
            key = (category, stem, sid)
            if key in done:
                continue

            out_dir = case / sid
            pr_path = out_dir / "merged_percell_sicle_masks_int32.npy"
            if not pr_path.is_file():
                print(f"  run {sid}")
                subprocess.run(
                    [
                        py,
                        str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                        "--from-dir",
                        str(cp_dir),
                        "-o",
                        str(out_dir),
                        "--image",
                        str(input_png),
                        *BASE,
                        *var["extra"],
                    ],
                    cwd=str(REPO),
                    env=env,
                    check=True,
                )

            r = evaluate_pair(gt_path, pr_path)
            br = mean_br_strict(gt_arr, np.load(pr_path).astype(np.int32))
            row = {
                "category": category,
                "roi": stem,
                "variant_id": sid,
                "label": var["label"],
                "br_mean_strict": br,
                "pixel_dice": r.get("pixel_dice"),
                "aji": r.get("aji"),
                "n_gt": r.get("n_gt"),
                "n_pr": r.get("n_pr"),
            }
            _append_row(csv_path, row)
            rows.append(row)
            done.add(key)
            print(f"    {sid:22s}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    print(f"\nProgress: {len(done)}/{n_total} (started at {n_done_start})")
    print("\n=== BR macro por variante ===")
    for var in VARIANTS:
        vals = [float(r["br_mean_strict"]) for r in rows if r["variant_id"] == var["id"]]
        if vals:
            print(f"  {var['id']:22s}: {np.mean(vals):.4f}  (n={len(vals)})")

    if rows:
        full = np.mean([float(r["br_mean_strict"]) for r in rows if r["variant_id"] == "full"])
        raw = np.mean([float(r["br_mean_strict"]) for r in rows if r["variant_id"] == "sicle_raw"])
        ando = np.mean([float(r["br_mean_strict"]) for r in rows if r["variant_id"] == "and_only"])
        print(f"\n  Δ vs full: raw={raw - full:+.4f}  and_only={ando - full:+.4f}")

    _write_summary(rows, out_root, csv_path, len(rois))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out_root / 'summary.md'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Post-process ablation for Oral Epithelium.")
    p.add_argument("--full", action="store_true", help="Run all ROIs with GT + original TIFF")
    p.add_argument("--skip-cellpose", action="store_true", help="Require existing cp_flow per case")
    p.add_argument(
        "--reuse-benchmark-cp",
        action="store_true",
        help="Use path_cost_benchmark cp_flow when available (partial overlap only)",
    )
    args = p.parse_args()

    if args.full:
        rois = discover_rois()
        out_root = OUT_ROOT_FULL
        bench = RUNS / "path_cost_benchmark" if args.reuse_benchmark_cp else None
        print(f"Full dataset: {len(rois)} ROIs → {out_root}")
    else:
        rois = SAMPLE_ROIS
        out_root = OUT_ROOT_SAMPLE
        bench = RUNS / "path_cost_benchmark"

    if not rois:
        raise SystemExit("No ROIs found.")

    return run_ablation(rois, out_root, skip_cellpose=args.skip_cellpose, bench_cp_root=bench)


if __name__ == "__main__":
    raise SystemExit(main())
