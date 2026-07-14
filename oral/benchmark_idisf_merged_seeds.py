#!/usr/bin/env python3
"""
Benchmark iDISF per-cell with **Cellpose + PathoSAM seeds** vs baselines.

Seed merge strategies:
  - ``union``: one iDISF run on CP+PS seed map (PS neighbors can affect CP contours).
  - ``split``: iDISF on CP only, then iDISF on PS-novel only; merge without overwriting CP cells.

Compares against ``idisf_percell`` (CP seeds only).
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
    run_idisf_merged_cp_pathosam_seeds,
)

OUT_ROOT = RUNS / "all_methods_comparison"
CP_ROOT = RUNS / "postprocess_ablation_full"
BASELINE = "idisf_percell"

STRATEGIES = {
    "union": "idisf_cp_pathosam_seeds",
    "split": "idisf_cp_ps_split",
}


def _method_dir(strategy: str) -> str:
    return STRATEGIES[strategy]


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


def _run_case(
    rows: list[dict],
    *,
    dataset: str,
    sample_id: str,
    category: str,
    image_path: Path,
    gt_path: Path,
    cp_flow: Path,
    ps_path: Path,
    strategy: str,
    run_infer: bool,
    done: set[tuple[str, str, str]],
) -> None:
    method = _method_dir(strategy)
    if (dataset, sample_id, method) in done:
        return
    case_parent = OUT_ROOT / ("oral_epithelium" if dataset == "oral_epithelium" else "ihc_tma")
    case = case_parent / (f"{category}/{sample_id}" if dataset == "oral_epithelium" else sample_id)
    out_dir = case / method
    pr = out_dir / "merged_percell_idisf_masks_int32.npy"
    env = _pipeline_env()
    if run_infer and not _mask_ready(pr):
        run_idisf_merged_cp_pathosam_seeds(
            image_path, cp_flow, ps_path, out_dir,
            pipe_dir=PIPE, repo_dir=REPO, env=env, strategy=strategy,
        )
    if not _mask_ready(pr):
        print(f"    skip {method}: no mask")
        return

    cp = np.load(cp_flow / "step04_masks_uint16.npy")
    ps = np.load(ps_path)
    from method_infer import pathosam_novel_seeds_only, pathosam_saliency_path
    sal_path = pathosam_saliency_path(ps_path.parent)
    sal = np.load(sal_path).astype(np.float32) if sal_path.is_file() else None
    novel = pathosam_novel_seeds_only(cp, ps, saliency_prob=sal)
    n_cp = len([x for x in np.unique(cp) if int(x) > 0])
    n_ps = len([x for x in np.unique(novel) if int(x) > 0])
    br, fb, f_area, dice = _score(gt_path, pr)
    rows.append({
        "dataset": dataset,
        "sample_id": sample_id,
        "category": category,
        "method": method,
        "seed_strategy": strategy,
        "n_cp_seeds": n_cp,
        "n_ps_added": n_ps,
        "br_mean_strict": br,
        "fb_mean_strict": fb,
        "f_area_mean_strict": f_area,
        "pixel_dice": dice,
    })
    done.add((dataset, sample_id, method))
    print(
        f"    {method:24s} [{strategy}] seeds {n_cp}+{n_ps}  "
        f"BR={br:.4f} Fb={fb:.4f} Fa={f_area:.4f} Dice={dice:.4f}"
    )


def benchmark_oral(
    rows: list[dict],
    *,
    max_samples: int,
    strategies: list[str],
    run_infer: bool,
    done: set[tuple[str, str, str]],
) -> None:
    rois = discover_rois()
    if max_samples > 0:
        rois = rois[:max_samples]

    for i, (category, stem) in enumerate(rois, 1):
        gt_path = CP_ROOT / category / stem / "gt" / "gold_standard_masks_int32.npy"
        image_path = CP_ROOT / category / stem / f"{stem}.png"
        cp_flow = _resolve_oral_cp_flow(category, stem)
        ps_path = OUT_ROOT / "oral_epithelium" / category / stem / "pathosam_flow" / "step04_masks_uint16.npy"
        if not gt_path.is_file() or not cp_flow.is_dir() or not _mask_ready(ps_path):
            print(f"  skip {category}/{stem}: missing GT/cp/pathosam")
            continue
        print(f"[oral {i}/{len(rois)}] {category}/{stem}")
        for strategy in strategies:
            _run_case(
                rows, dataset="oral_epithelium", sample_id=stem, category=category,
                image_path=image_path, gt_path=gt_path, cp_flow=cp_flow, ps_path=ps_path,
                strategy=strategy, run_infer=run_infer, done=done,
            )


def benchmark_ihc(
    rows: list[dict],
    *,
    max_samples: int,
    strategies: list[str],
    run_infer: bool,
    done: set[tuple[str, str, str]],
) -> None:
    images = sorted((DATA_IHC / "images").glob("*.png"))
    if max_samples > 0:
        images = images[:max_samples]

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        case = OUT_ROOT / "ihc_tma" / stem
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
        for strategy in strategies:
            _run_case(
                rows, dataset="ihc_tma", sample_id=stem, category="ihc",
                image_path=img_path, gt_path=gt_path, cp_flow=cp_flow, ps_path=ps_path,
                strategy=strategy, run_infer=run_infer, done=done,
            )


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
        "**Estratégias de merge:**",
        "- `union` — um único iDISF no mapa CP+PS (PS vizinho altera exclude-other das células CP).",
        "- `split` — iDISF só em CP; depois iDISF só em PS-novo; funde sem sobrescrever células CP.",
        "",
        f"Baseline: `{BASELINE}` (somente Cellpose como semente).",
        "",
    ]
    for ds, label in (("oral_epithelium", "Oral"), ("ihc_tma", "IHC")):
        sub = [r for r in merged_rows if r["dataset"] == ds]
        if not sub:
            continue
        base = _load_baseline(baseline_csv, ds)
        lines.extend([f"## {label} (n amostras por método)", ""])
        for strategy, method in STRATEGIES.items():
            mrows = [r for r in sub if r["method"] == method]
            if not mrows:
                continue
            m_new = _macro(mrows)
            paired = [r for r in mrows if r["sample_id"] in base]
            m_base = _macro([base[r["sample_id"]] for r in paired]) if paired else {}
            avg_added = float(np.mean([float(r["n_ps_added"]) for r in mrows]))
            lines.extend([
                f"### `{method}` ({strategy})",
                "",
                f"Sementes PS adicionadas em média: **+{avg_added:.1f}**.",
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
                f"| `{method}` | {m_new['br_mean_strict']:.4f} | {m_new['fb_mean_strict']:.4f} | "
                f"{m_new['f_area_mean_strict']:.4f} | {m_new['pixel_dice']:.4f} |"
            )
            if paired and len(paired) >= 2:
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
                    sig = "ns" if p >= 0.05 else "sig"
                    lines.append("")
                    lines.append(
                        f"- **{name}** vs baseline: Δ={delta:+.4f}, t={t:.2f}, p={p:.2e} ({sig})"
                    )
            elif paired:
                lines.append("")
                lines.append(f"- t-test: n={len(paired)} (insuficiente; precisa ≥2 amostras).")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("oral", "ihc", "both"), default="both")
    p.add_argument(
        "--strategy",
        choices=("union", "split", "both"),
        default="split",
        help="Seed merge strategy (default: split — preserves CP contours).",
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--metrics-only", action="store_true")
    args = p.parse_args()

    strategies = list(STRATEGIES.keys()) if args.strategy == "both" else [args.strategy]

    out_csv = OUT_ROOT / "idisf_merged_seeds_metrics.csv"
    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    if out_csv.is_file():
        with out_csv.open(encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                rows.append(r)
                done.add((r["dataset"], r["sample_id"], r["method"]))

    run_infer = not args.metrics_only
    if args.dataset in ("oral", "both"):
        benchmark_oral(rows, max_samples=args.max_samples, strategies=strategies,
                       run_infer=run_infer, done=done)
    if args.dataset in ("ihc", "both"):
        benchmark_ihc(rows, max_samples=args.max_samples, strategies=strategies,
                      run_infer=run_infer, done=done)

    latest: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        latest[(r["dataset"], r["sample_id"], r["method"])] = r
    rows = list(latest.values())
    rows.sort(key=lambda r: (r["dataset"], r["sample_id"], r["method"]))

    fieldnames = [
        "dataset", "sample_id", "category", "method", "seed_strategy",
        "n_cp_seeds", "n_ps_added",
        "br_mean_strict", "fb_mean_strict", "f_area_mean_strict", "pixel_dice",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fieldnames} for r in rows)

    summary_path = OUT_ROOT / "idisf_merged_seeds_summary.md"
    write_summary(rows, OUT_ROOT / "metrics_all_methods.csv", summary_path)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
