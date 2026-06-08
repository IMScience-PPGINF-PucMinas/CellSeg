#!/usr/bin/env python3
"""
Benchmark iDISF per-cell with **Cellpose + PathoSAM union seeds** vs baselines.

Seeds: all Cellpose instances + PathoSAM instances with IoU < 0.5 vs any CP cell.

Compares against existing ``idisf_percell`` (CP seeds only) and reports BR/Fb/Fa/Dice.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import DATA_IHC, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import mean_br_strict
from benchmark_postprocess_ablation import discover_rois
from method_infer import (
    ihc_mask_to_instances,
    merge_cellpose_pathosam_seeds,
    run_idisf_merged_cp_pathosam_seeds,
)

OUT_ROOT = RUNS / "all_methods_comparison"
CP_ROOT = RUNS / "postprocess_ablation_full"
METHOD = "idisf_cp_pathosam_seeds"
BASELINE = "idisf_percell"


def _pipeline_env() -> dict[str, str]:
    doutorado = REPO.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PIPE),
            str(REPO / "cellpose"),
            str(doutorado),
            str(doutorado / "iDISF" / "python3"),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def _score(gt_path: Path, pr_path: Path) -> tuple[float, float, float, float]:
    sys.path.insert(0, str(PIPE))
    from boundary_fb_metric import mean_f_area_strict, mean_fb_strict
    from evaluate_instances import evaluate_pair

    gt = np.load(gt_path).astype(np.int32)
    pr = np.load(pr_path).astype(np.int32)
    ev = evaluate_pair(gt_path, pr_path)
    dice_roi = float(ev.get("pixel_dice", float("nan")))
    return (
        mean_br_strict(gt, pr),
        mean_fb_strict(gt, pr),
        mean_f_area_strict(gt, pr),
        dice_roi,
    )


def _resolve_oral_cp_flow(category: str, stem: str) -> Path:
    return CP_ROOT / category / stem / "cp_flow"


def _mask_ready(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def benchmark_oral(
    rows: list[dict],
    *,
    max_samples: int,
    run_infer: bool,
    done: set[tuple[str, str]],
) -> None:
    rois = discover_rois()
    if max_samples > 0:
        rois = rois[:max_samples]
    env = _pipeline_env()
    out_ds = OUT_ROOT / "oral_epithelium"

    for i, (category, stem) in enumerate(rois, 1):
        if (category, stem) in done:
            continue
        case = out_ds / category / stem
        gt_path = CP_ROOT / category / stem / "gt" / "gold_standard_masks_int32.npy"
        image_path = CP_ROOT / category / stem / f"{stem}.png"
        cp_flow = _resolve_oral_cp_flow(category, stem)
        ps_path = case / "pathosam_flow" / "step04_masks_uint16.npy"
        if not gt_path.is_file() or not cp_flow.is_dir() or not _mask_ready(ps_path):
            print(f"  skip {category}/{stem}: missing GT/cp/pathosam")
            continue

        print(f"[oral {i}/{len(rois)}] {category}/{stem}")
        out_dir = case / METHOD
        pr = out_dir / "merged_percell_idisf_masks_int32.npy"
        if run_infer and not _mask_ready(pr):
            run_idisf_merged_cp_pathosam_seeds(
                image_path, cp_flow, ps_path, out_dir,
                pipe_dir=PIPE, repo_dir=REPO, env=env,
            )
        if not _mask_ready(pr):
            print("    skip: no merged-seed iDISF mask")
            continue

        cp = np.load(cp_flow / "step04_masks_uint16.npy")
        ps = np.load(ps_path)
        merged = merge_cellpose_pathosam_seeds(cp, ps)
        n_cp = len([x for x in np.unique(cp) if int(x) > 0])
        n_merged = len([x for x in np.unique(merged) if int(x) > 0])
        br, fb, f_area, dice = _score(gt_path, pr)
        rows.append({
            "dataset": "oral_epithelium",
            "sample_id": stem,
            "category": category,
            "method": METHOD,
            "n_cp_seeds": n_cp,
            "n_merged_seeds": n_merged,
            "n_ps_added": n_merged - n_cp,
            "br_mean_strict": br,
            "fb_mean_strict": fb,
            "f_area_mean_strict": f_area,
            "pixel_dice": dice,
        })
        print(f"    seeds {n_cp}→{n_merged} (+{n_merged-n_cp} PS)  BR={br:.4f} Fb={fb:.4f} Fa={f_area:.4f} Dice={dice:.4f}")


def benchmark_ihc(
    rows: list[dict],
    *,
    max_samples: int,
    run_infer: bool,
    done: set[tuple[str, str]],
) -> None:
    images = sorted((DATA_IHC / "images").glob("*.png"))
    if max_samples > 0:
        images = images[:max_samples]
    env = _pipeline_env()
    out_ds = OUT_ROOT / "ihc_tma"

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        if ("ihc", stem) in done:
            continue
        case = out_ds / stem
        gt_path = case / "gt_instances_int32.npy"
        cp_flow = case / "cp_flow"
        ps_path = case / "pathosam_flow" / "step04_masks_uint16.npy"
        mask_path = DATA_IHC / "masks" / f"{stem}.npy"
        if not mask_path.is_file() or not cp_flow.is_dir() or not _mask_ready(ps_path):
            print(f"  skip {stem}: missing data")
            continue
        if not gt_path.is_file():
            np.save(gt_path, ihc_mask_to_instances(np.load(mask_path)))

        print(f"[ihc {i}/{len(images)}] {stem}")
        out_dir = case / METHOD
        pr = out_dir / "merged_percell_idisf_masks_int32.npy"
        if run_infer and not _mask_ready(pr):
            run_idisf_merged_cp_pathosam_seeds(
                img_path, cp_flow, ps_path, out_dir,
                pipe_dir=PIPE, repo_dir=REPO, env=env,
            )
        if not _mask_ready(pr):
            print("    skip: no merged-seed iDISF mask")
            continue

        cp = np.load(cp_flow / "step04_masks_uint16.npy")
        ps = np.load(ps_path)
        merged = merge_cellpose_pathosam_seeds(cp, ps)
        n_cp = len([x for x in np.unique(cp) if int(x) > 0])
        n_merged = len([x for x in np.unique(merged) if int(x) > 0])
        br, fb, f_area, dice = _score(gt_path, pr)
        rows.append({
            "dataset": "ihc_tma",
            "sample_id": stem,
            "category": "ihc",
            "method": METHOD,
            "n_cp_seeds": n_cp,
            "n_merged_seeds": n_merged,
            "n_ps_added": n_merged - n_cp,
            "br_mean_strict": br,
            "fb_mean_strict": fb,
            "f_area_mean_strict": f_area,
            "pixel_dice": dice,
        })
        print(f"    seeds {n_cp}→{n_merged} (+{n_merged-n_cp} PS)  BR={br:.4f} Fb={fb:.4f} Fa={f_area:.4f} Dice={dice:.4f}")


def _load_baseline(csv_path: Path, dataset: str) -> dict[str, dict]:
    if not csv_path.is_file():
        return {}
    out: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["dataset"] == dataset and r["method"] == BASELINE:
                out[r["sample_id"]] = r
    return out


def _macro(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ("br_mean_strict", "fb_mean_strict", "f_area_mean_strict", "pixel_dice")
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}


def write_summary(
    merged_rows: list[dict],
    baseline_csv: Path,
    out_path: Path,
) -> None:
    lines = [
        "# iDISF com sementes Cellpose + PathoSAM",
        "",
        "Sementes = todas as instâncias Cellpose + instâncias PathoSAM com IoU < 0.5 vs qualquer célula CP.",
        "iDISF per-cell igual ao pipeline padrão (`--disable-and-merge`, exclude-other-cells).",
        "",
        f"Baseline: `{BASELINE}` (somente Cellpose como semente).",
        "",
    ]
    for ds, label in (("oral_epithelium", "Oral"), ("ihc_tma", "IHC")):
        sub = [r for r in merged_rows if r["dataset"] == ds]
        if not sub:
            continue
        m_new = _macro(sub)
        base = _load_baseline(baseline_csv, ds)
        paired = [r for r in sub if r["sample_id"] in base]
        m_base = _macro([base[r["sample_id"]] for r in paired]) if paired else {}
        avg_added = float(np.mean([float(r["n_ps_added"]) for r in sub]))
        lines.extend([
            f"## {label} (n={len(sub)})",
            "",
            f"Sementes PathoSAM adicionadas em média: **+{avg_added:.1f}** por amostra.",
            "",
            "| Método | BR | Fb | Fa | Dice |",
            "|--------|---:|---:|---:|-----:|",
        ])
        if m_base:
            lines.append(
                f"| `{BASELINE}` | {m_base['br_mean_strict']:.4f} | {m_base['fb_mean_strict']:.4f} | "
                f"{m_base['f_area_mean_strict']:.4f} | {m_base['pixel_dice']:.4f} |"
            )
        lines.append(
            f"| `{METHOD}` | {m_new['br_mean_strict']:.4f} | {m_new['fb_mean_strict']:.4f} | "
            f"{m_new['f_area_mean_strict']:.4f} | {m_new['pixel_dice']:.4f} |"
        )
        if paired:
            from scipy import stats
            for key, name in (
                ("pixel_dice", "Dice"),
                ("br_mean_strict", "BR"),
                ("fb_mean_strict", "Fb"),
                ("f_area_mean_strict", "Fa"),
            ):
                a = np.array([float(r[key]) for r in paired])
                b = np.array([float(base[r["sample_id"]][key]) for r in paired])
                t, p = stats.ttest_rel(a, b)
                delta = float((a - b).mean())
                lines.append("")
                lines.append(f"- **{name}** vs baseline: Δ={delta:+.4f}, t={t:.2f}, p={p:.2e}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("oral", "ihc", "both"), default="both")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--metrics-only", action="store_true")
    args = p.parse_args()

    out_csv = OUT_ROOT / "idisf_merged_seeds_metrics.csv"
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if out_csv.is_file():
        with out_csv.open(encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                rows.append(r)
                done.add((r["dataset"], r["sample_id"]))

    run_infer = not args.metrics_only
    if args.dataset in ("oral", "both"):
        benchmark_oral(rows, max_samples=args.max_samples, run_infer=run_infer, done=done)
    if args.dataset in ("ihc", "both"):
        benchmark_ihc(rows, max_samples=args.max_samples, run_infer=run_infer, done=done)

    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["dataset"], r["sample_id"])] = r
    rows = list(latest.values())
    rows.sort(key=lambda r: (r["dataset"], r["sample_id"]))

    fieldnames = [
        "dataset", "sample_id", "category", "method",
        "n_cp_seeds", "n_merged_seeds", "n_ps_added",
        "br_mean_strict", "fb_mean_strict", "f_area_mean_strict", "pixel_dice",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary_path = OUT_ROOT / "idisf_merged_seeds_summary.md"
    write_summary(rows, OUT_ROOT / "metrics_all_methods.csv", summary_path)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
