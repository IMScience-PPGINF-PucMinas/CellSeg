#!/usr/bin/env python3
"""
Compare BR and Fb across methods on both datasets in data/:

  - cellpose (alone)
  - sicle_percell (per-cell SICLE on Cellpose seeds, Nf=2 raw)
  - idisf_percell (per-cell iDISF on Cellpose seeds)
  - cellvit (alone)
  - pathosam (alone, vit_l_histopathology via micro_sam)

Datasets:
  - oral_epithelium: 100 annotated ROIs (healthy + severe)
  - ihc_tma, monuseg, dsb2018, pannuke, consep: images/*.png + masks/*.npy (int32 instances)

Outputs:
  outputs/runs/all_methods_comparison/{oral_epithelium,ihc_tma}/metrics.csv
  outputs/runs/all_methods_comparison/summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from _paths import DATA_IHC, PATCH_DATASETS, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import mean_br_strict
from benchmark_postprocess_ablation import discover_rois
from method_infer import (
    ihc_mask_to_instances,
    run_cellpose,
    run_cellvit,
    run_idisf_percell,
    run_pathosam,
    run_sicle_percell,
    run_stardist,
    run_watershed,
    stardist_model_name_for_dataset,
)

OUT_ROOT = RUNS / "all_methods_comparison"
PROGRESS_PATH = OUT_ROOT / "benchmark_progress.json"
CP_ROOT = RUNS / "postprocess_ablation_full"
SICLE_ROOT = RUNS / "nf_sweep_full"
IDISF_ROOT = RUNS / "percell_idisf_full"

METHODS = (
    "cellpose",
    "sicle_percell",
    "idisf_percell",
    "cellvit",
    "pathosam",
    "stardist",
    "watershed",
)
DEFAULT_METHODS = METHODS[:5]

# Reuse existing oral masks when available
ORAL_REUSE = {
    "cellpose": lambda cat, stem: CP_ROOT / cat / stem / "cp_flow" / "step04_masks_uint16.npy",
    "sicle_percell": lambda cat, stem: _resolve_sicle(cat, stem),
    "idisf_percell": lambda cat, stem: _resolve_idisf(cat, stem),
}


def _resolve_sicle(category: str, stem: str) -> Path | None:
    for root in (SICLE_ROOT, CP_ROOT):
        for sub in ("nf2_n0200_raw", "sicle_raw"):
            p = root / category / stem / sub / "merged_percell_sicle_masks_int32.npy"
            if p.is_file() and p.stat().st_size > 0:
                return p
        for p in (root / category / stem).glob("nf2_n0*_raw/merged_percell_sicle_masks_int32.npy"):
            if p.is_file() and p.stat().st_size > 0:
                return p
    return None


def _resolve_idisf(category: str, stem: str) -> Path | None:
    for sub in ("idisf_exclude_other", "idisf_unconquerable"):
        p = IDISF_ROOT / category / stem / sub / "merged_percell_idisf_masks_int32.npy"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _mask_ready(p: Path | None) -> bool:
    if p is None or not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        np.load(p)
        return True
    except (OSError, ValueError, EOFError):
        return False


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
    sicle_bin = doutorado / "SICLE" / "bin" / "RunSICLE"
    if sicle_bin.is_file():
        env["SICLE_BIN"] = str(sicle_bin)
    return env


def _score(gt_path: Path, pr_path: Path, *, dataset: str) -> dict[str, float]:
    sys.path.insert(0, str(PIPE))
    from boundary_fb_metric import (
        boundary_metrics_seraph,
        mean_f_area_strict,
        mean_fb_strict,
    )
    from evaluate_instances import evaluate_pair

    gt = np.load(gt_path).astype(np.int32)
    pr = np.load(pr_path).astype(np.int32)
    ev = evaluate_pair(gt_path, pr_path)
    dice_roi = float(ev.get("pixel_dice", float("nan")))
    out: dict[str, float] = {
        "br_mean_strict": mean_br_strict(gt, pr),
        "fb_mean_strict": mean_fb_strict(gt, pr),
        "f_area_mean_strict": mean_f_area_strict(gt, pr),
        "pixel_dice": dice_roi,
    }
    if dataset != "pannuke":
        br_s, bp_s, bf_s = boundary_metrics_seraph(gt, pr)
        out["br_seraph"] = br_s
        out["boundary_precision_seraph"] = bp_s
        out["boundary_f_seraph"] = bf_s
    return out


def _finite_metric(val) -> bool:
    if val in (None, ""):
        return False
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def _row_complete(r: dict) -> bool:
    if not _finite_metric(r.get("f_area_mean_strict")):
        return False
    if r.get("dataset") != "pannuke" and not _finite_metric(r.get("br_seraph")):
        return False
    return True


def _append_row(
    rows: list[dict],
    *,
    dataset: str,
    sample_id: str,
    category: str,
    method: str,
    scores: dict[str, float],
) -> None:
    row = {
        "dataset": dataset,
        "sample_id": sample_id,
        "category": category,
        "method": method,
        "br_mean_strict": scores["br_mean_strict"],
        "fb_mean_strict": scores["fb_mean_strict"],
        "f_area_mean_strict": scores["f_area_mean_strict"],
        "pixel_dice": scores["pixel_dice"],
    }
    if dataset != "pannuke":
        row["br_seraph"] = scores.get("br_seraph", float("nan"))
        row["boundary_precision_seraph"] = scores.get("boundary_precision_seraph", float("nan"))
        row["boundary_f_seraph"] = scores.get("boundary_f_seraph", float("nan"))
    rows.append(row)


CSV_FIELDS = [
    "dataset",
    "sample_id",
    "category",
    "method",
    "br_mean_strict",
    "br_seraph",
    "boundary_precision_seraph",
    "boundary_f_seraph",
    "fb_mean_strict",
    "f_area_mean_strict",
    "pixel_dice",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = CSV_FIELDS
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fmt_mean(vals: list[float]) -> str:
    finite = [float(v) for v in vals if _finite_metric(v)]
    return f"{float(np.mean(finite)):.4f}" if finite else "—"


def _macro_table(rows: list[dict], dataset: str) -> list[str]:
    by_br: dict[str, list[float]] = defaultdict(list)
    by_br_s: dict[str, list[float]] = defaultdict(list)
    by_m_fb: dict[str, list[float]] = defaultdict(list)
    by_m_fa: dict[str, list[float]] = defaultdict(list)
    by_m_dice: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["dataset"] != dataset:
            continue
        by_br[r["method"]].append(float(r["br_mean_strict"]))
        if _finite_metric(r.get("br_seraph")):
            by_br_s[r["method"]].append(float(r["br_seraph"]))
        by_m_fb[r["method"]].append(float(r["fb_mean_strict"]))
        if r.get("f_area_mean_strict") not in (None, ""):
            by_m_fa[r["method"]].append(float(r["f_area_mean_strict"]))
        if r.get("pixel_dice") not in (None, ""):
            by_m_dice[r["method"]].append(float(r["pixel_dice"]))

    lines = [f"### {dataset}", ""]
    if dataset == "pannuke":
        lines.extend([
            "| Método | BR strict | Fb | Fa | Dice | n |",
            "|--------|--------:|---:|---:|-----:|--:|",
        ])
        for m in METHODS:
            if by_br.get(m):
                lines.append(
                    f"| `{m}` | {_fmt_mean(by_br[m])} | {_fmt_mean(by_m_fb[m])} | "
                    f"{_fmt_mean(by_m_fa[m])} | {_fmt_mean(by_m_dice[m])} | {len(by_br[m])} |"
                )
    else:
        lines.extend([
            "| Método | BR macro | BR pooled | Fb | Fa | Dice | n |",
            "|--------|--------:|----------:|---:|---:|-----:|--:|",
        ])
        for m in METHODS:
            if by_br.get(m):
                lines.append(
                    f"| `{m}` | {_fmt_mean(by_br[m])} | {_fmt_mean(by_br_s[m])} | "
                    f"{_fmt_mean(by_m_fb[m])} | {_fmt_mean(by_m_fa[m])} | "
                    f"{_fmt_mean(by_m_dice[m])} | {len(by_br[m])} |"
                )
    lines.append("")
    return lines


def _print_scores(method: str, scores: dict[str, float], *, dataset: str) -> None:
    if dataset == "pannuke":
        print(
            f"    {method:16s} BR={scores['br_mean_strict']:.4f} "
            f"Fb={scores['fb_mean_strict']:.4f} Fa={scores['f_area_mean_strict']:.4f} "
            f"Dice={scores['pixel_dice']:.4f}"
        )
    else:
        print(
            f"    {method:16s} BRs={scores['br_mean_strict']:.4f} "
            f"BR_S={scores.get('br_seraph', float('nan')):.4f} "
            f"Fb={scores['fb_mean_strict']:.4f} Fa={scores['f_area_mean_strict']:.4f} "
            f"Dice={scores['pixel_dice']:.4f}"
        )


def benchmark_oral(
    rows: list[dict],
    *,
    max_samples: int,
    methods: set[str],
    run_infer: bool,
    gpu: bool,
    done: set[tuple[str, str, str]],
) -> None:
    rois = discover_rois()
    if max_samples > 0:
        rois = rois[:max_samples]
    env = _pipeline_env()
    out_ds = OUT_ROOT / "oral_epithelium"

    for i, (category, stem) in enumerate(rois, 1):
        case = out_ds / category / stem
        case.mkdir(parents=True, exist_ok=True)
        gt_path = CP_ROOT / category / stem / "gt" / "gold_standard_masks_int32.npy"
        image_path = CP_ROOT / category / stem / f"{stem}.png"
        if not gt_path.is_file():
            print(f"  skip {category}/{stem}: missing GT")
            continue
        print(f"[oral {i}/{len(rois)}] {category}/{stem}")

        cp_dir = case / "cp_flow"
        if "cellpose" in methods or "sicle_percell" in methods or "idisf_percell" in methods:
            reuse_cp = ORAL_REUSE["cellpose"](category, stem)
            if _mask_ready(reuse_cp):
                cp_dir.mkdir(parents=True, exist_ok=True)
                dst = cp_dir / "step04_masks_uint16.npy"
                if not dst.is_file():
                    dst.symlink_to(reuse_cp.resolve())
            elif run_infer:
                run_cellpose(image_path, cp_dir, gpu=gpu)

        for method in methods:
            if ("oral_epithelium", stem, method) in done:
                continue
            if method == "cellpose":
                pr = ORAL_REUSE["cellpose"](category, stem) if _mask_ready(ORAL_REUSE["cellpose"](category, stem)) else cp_dir / "step04_masks_uint16.npy"
            elif method == "sicle_percell":
                pr = ORAL_REUSE["sicle_percell"](category, stem)
                if not _mask_ready(pr) and run_infer and _mask_ready(cp_dir / "step04_masks_uint16.npy"):
                    pr = run_sicle_percell(image_path, cp_dir, case / "sicle_percell",
                                           pipe_dir=PIPE, repo_dir=REPO, env=env)
            elif method == "idisf_percell":
                pr = ORAL_REUSE["idisf_percell"](category, stem)
                if not _mask_ready(pr) and run_infer and _mask_ready(cp_dir / "step04_masks_uint16.npy"):
                    pr = run_idisf_percell(image_path, cp_dir, case / "idisf_percell",
                                           pipe_dir=PIPE, repo_dir=REPO, env=env)
            elif method == "cellvit":
                pr = case / "cellvit_flow" / "step04_masks_uint16.npy"
                if not _mask_ready(pr) and run_infer:
                    run_cellvit(image_path, case / "cellvit_flow", gpu=None if not gpu else 0)
            elif method == "pathosam":
                pr = case / "pathosam_flow" / "step04_masks_uint16.npy"
                if not _mask_ready(pr) and run_infer:
                    run_pathosam(image_path, case / "pathosam_flow", device="cuda" if gpu else "cpu")
            elif method == "stardist":
                pr = case / "stardist_flow" / "step04_masks_uint16.npy"
                if not _mask_ready(pr) and run_infer:
                    run_stardist(
                        image_path,
                        case / "stardist_flow",
                        model_name=stardist_model_name_for_dataset("oral_epithelium"),
                        gpu=False,
                    )
            elif method == "watershed":
                pr = case / "watershed_flow" / "step04_masks_uint16.npy"
                if not _mask_ready(pr) and run_infer:
                    run_watershed(image_path, case / "watershed_flow")
            else:
                continue

            if not _mask_ready(pr):
                print(f"    skip {method}: no mask")
                continue
            scores = _score(gt_path, pr, dataset="oral_epithelium")
            _append_row(rows, dataset="oral_epithelium", sample_id=stem, category=category,
                        method=method, scores=scores)
            _print_scores(method, scores, dataset="oral_epithelium")


def _write_progress(
    *,
    dataset: str,
    current: int,
    total: int,
    sample_id: str,
    started_at: float,
) -> None:
    elapsed = max(0.0, time.time() - started_at)
    pct = 100.0 * current / total if total > 0 else 0.0
    rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
    remaining = total - current
    eta_s = remaining / rate if rate > 0 else None
    payload = {
        "dataset": dataset,
        "current": current,
        "total": total,
        "percent": round(pct, 2),
        "sample_id": sample_id,
        "elapsed_s": round(elapsed, 1),
        "eta_s": round(eta_s, 1) if eta_s is not None else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = OUT_ROOT / "benchmark_progress.md"
    eta_txt = f"{eta_s/3600:.1f} h" if eta_s and eta_s >= 3600 else (
        f"{eta_s/60:.0f} min" if eta_s and eta_s >= 60 else (
            f"{eta_s:.0f} s" if eta_s else "—"
        )
    )
    md.write_text(
        "\n".join([
            "# Benchmark — progresso em tempo real",
            "",
            f"**Dataset:** `{dataset}`",
            f"**Progresso:** {current}/{total} (**{pct:.1f}%**)",
            f"**Último patch:** `{sample_id}`",
            f"**Tempo decorrido:** {elapsed/3600:.1f} h" if elapsed >= 3600 else f"**Tempo decorrido:** {elapsed/60:.0f} min",
            f"**ETA estimado:** {eta_txt}",
            "",
            f"Atualizado: {payload['updated_at']}",
            "",
            "JSON: `benchmark_progress.json`",
        ]),
        encoding="utf-8",
    )


def _gt_from_mask(mask_path: Path, gt_path: Path, dataset: str) -> None:
    if gt_path.is_file():
        return
    mask = np.load(mask_path)
    if dataset == "ihc_tma" and mask.ndim == 3:
        np.save(gt_path, ihc_mask_to_instances(mask))
    else:
        np.save(gt_path, mask.astype(np.int32))


def _method_mask_path(case: Path, method: str) -> Path:
    if method == "cellpose":
        return case / "cp_flow" / "step04_masks_uint16.npy"
    if method == "sicle_percell":
        return case / "sicle_percell" / "merged_percell_sicle_masks_int32.npy"
    if method == "idisf_percell":
        return case / "idisf_percell" / "merged_percell_idisf_masks_int32.npy"
    if method == "cellvit":
        return case / "cellvit_flow" / "step04_masks_uint16.npy"
    if method == "pathosam":
        return case / "pathosam_flow" / "step04_masks_uint16.npy"
    if method == "stardist":
        return case / "stardist_flow" / "step04_masks_uint16.npy"
    if method == "watershed":
        return case / "watershed_flow" / "step04_masks_uint16.npy"
    raise ValueError(method)


def _run_cpu_percell(
    img_path: Path,
    cp_dir: Path,
    case: Path,
    methods: set[str],
    env: dict[str, str],
) -> None:
    if "sicle_percell" in methods and not _mask_ready(_method_mask_path(case, "sicle_percell")):
        if _mask_ready(cp_dir / "step04_masks_uint16.npy"):
            run_sicle_percell(img_path, cp_dir, case / "sicle_percell",
                              pipe_dir=PIPE, repo_dir=REPO, env=env)
    if "idisf_percell" in methods and not _mask_ready(_method_mask_path(case, "idisf_percell")):
        if _mask_ready(cp_dir / "step04_masks_uint16.npy"):
            run_idisf_percell(img_path, cp_dir, case / "idisf_percell",
                              pipe_dir=PIPE, repo_dir=REPO, env=env)


def _score_methods_for_case(
    rows: list[dict],
    *,
    dataset: str,
    stem: str,
    category: str,
    gt_path: Path,
    case: Path,
    methods: set[str],
    done: set[tuple[str, str, str]],
) -> None:
    for method in methods:
        if (dataset, stem, method) in done:
            continue
        pr = _method_mask_path(case, method)
        if not _mask_ready(pr):
            print(f"    skip {method}: no mask")
            continue
        scores = _score(gt_path, pr, dataset=dataset)
        _append_row(rows, dataset=dataset, sample_id=stem, category=category,
                    method=method, scores=scores)
        done.add((dataset, stem, method))
        _print_scores(method, scores, dataset=dataset)


def benchmark_patch_dataset(
    rows: list[dict],
    *,
    dataset: str,
    data_root: Path,
    category: str,
    max_samples: int,
    methods: set[str],
    run_infer: bool,
    gpu: bool,
    done: set[tuple[str, str, str]],
    cpu_workers: int = 1,
    shard_id: int = 0,
    num_shards: int = 1,
    skip_complete: bool = True,
) -> None:
    """Benchmark patch datasets with layout images/*.png + masks/*.npy."""
    images = sorted((data_root / "images").glob("*.png"))
    if num_shards > 1:
        images = images[shard_id::num_shards]
    if max_samples > 0:
        images = images[:max_samples]
    env = _pipeline_env()
    out_ds = OUT_ROOT / dataset
    parallel_cpu = cpu_workers > 1 and run_infer
    cpu_pool = ThreadPoolExecutor(max_workers=cpu_workers) if parallel_cpu else None
    pending_cpu: list[tuple[str, Path, Path, object]] = []
    started_at = time.time()
    total = len(images)

    def _drain_one() -> None:
        stem, gt_path, case, fut = pending_cpu.pop(0)
        fut.result()
        _score_methods_for_case(
            rows, dataset=dataset, stem=stem, category=category,
            gt_path=gt_path, case=case, methods=methods, done=done,
        )

    try:
        for i, img_path in enumerate(images, 1):
            stem = img_path.stem
            mask_path = data_root / "masks" / f"{stem}.npy"
            if not mask_path.is_file():
                print(f"  skip {stem}: missing mask")
                continue
            case = out_ds / stem
            case.mkdir(parents=True, exist_ok=True)
            gt_path = case / "gt_instances_int32.npy"
            _gt_from_mask(mask_path, gt_path, dataset)

            if skip_complete and run_infer:
                needs = any(
                    not _mask_ready(_method_mask_path(case, m))
                    for m in methods
                    if (dataset, stem, m) not in done
                )
                if not needs:
                    continue

            pct = 100.0 * i / total if total else 0.0
            print(
                f"[{dataset} {i}/{total} ({pct:.1f}%)] {stem}"
                + (f" [shard {shard_id}/{num_shards}]" if num_shards > 1 else ""),
                flush=True,
            )
            _write_progress(dataset=dataset, current=i, total=total, sample_id=stem, started_at=started_at)
            cp_dir = case / "cp_flow"

            if run_infer and ("cellpose" in methods or "sicle_percell" in methods or "idisf_percell" in methods):
                if not _mask_ready(cp_dir / "step04_masks_uint16.npy"):
                    run_cellpose(img_path, cp_dir, gpu=gpu)

            cpu_future = None
            if parallel_cpu and ("sicle_percell" in methods or "idisf_percell" in methods):
                cpu_future = cpu_pool.submit(
                    _run_cpu_percell, img_path, cp_dir, case, methods, env,
                )
            elif run_infer:
                _run_cpu_percell(img_path, cp_dir, case, methods, env)

            if run_infer and "cellvit" in methods and not _mask_ready(_method_mask_path(case, "cellvit")):
                run_cellvit(img_path, case / "cellvit_flow", gpu=None if not gpu else 0)
            if run_infer and "pathosam" in methods and not _mask_ready(_method_mask_path(case, "pathosam")):
                run_pathosam(img_path, case / "pathosam_flow", device="cuda" if gpu else "cpu")
            if run_infer and "stardist" in methods and not _mask_ready(_method_mask_path(case, "stardist")):
                run_stardist(
                    img_path,
                    case / "stardist_flow",
                    model_name=stardist_model_name_for_dataset(dataset),
                    gpu=False,
                )
            if run_infer and "watershed" in methods and not _mask_ready(_method_mask_path(case, "watershed")):
                run_watershed(img_path, case / "watershed_flow")

            if parallel_cpu and cpu_future is not None:
                pending_cpu.append((stem, gt_path, case, cpu_future))
                while len(pending_cpu) >= cpu_workers:
                    _drain_one()
            else:
                _score_methods_for_case(
                    rows, dataset=dataset, stem=stem, category=category,
                    gt_path=gt_path, case=case, methods=methods, done=done,
                )

        while pending_cpu:
            _drain_one()
    finally:
        if cpu_pool is not None:
            cpu_pool.shutdown(wait=True)


def benchmark_ihc(
    rows: list[dict],
    *,
    max_samples: int,
    methods: set[str],
    run_infer: bool,
    gpu: bool,
    done: set[tuple[str, str, str]],
) -> None:
    benchmark_patch_dataset(
        rows,
        dataset="ihc_tma",
        data_root=DATA_IHC,
        category="ihc",
        max_samples=max_samples,
        methods=methods,
        run_infer=run_infer,
        gpu=gpu,
        done=done,
    )


def write_summary(all_rows: list[dict]) -> None:
    lines = [
        "# Comparação multi-método — BR, Fb, Fa e Dice",
        "",
        "Métodos: `cellpose`, `sicle_percell`, `idisf_percell`, `cellvit`, `pathosam`.",
        "",
        "**Por borda (contorno):**",
        "- **BR macro** (`br_mean_strict`) — per-GT-instance recall at $\\tau{=}2$\\,px, "
        "thick boundaries, strict instance matching; **arithmetic mean over cells**.",
        "- **BR pooled** (`br_seraph`, BR Area) — same $\\tau{=}2$\\,px and boundary map, "
        "but all GT boundary pixels counted jointly on the tile (**not** averaged per cell).",
        "- **Fb** — F-measure Arbeláez no **contorno** 1 px (tolerância 0.0075×diagonal), per-cell strict.",
        "",
        "**Por região (área da célula):**",
        "- **Fa** (`f_area_mean_strict`) — F1 pixel a pixel: TP=A∩B, FN=A\\B, FP=B\\A, por célula GT, média macro.",
        "- **Dice** (`pixel_dice`) — mesmo F1 por **área**, mas na ROI inteira (todas as células fundidas em 0/1).",
        "",
        f"CSV: `outputs/runs/all_methods_comparison/metrics_all_methods.csv`",
        "",
        "## Médias macro por dataset",
        "",
    ]
    dataset_order = [
        "oral_epithelium",
        "ihc_tma",
        "monuseg",
        "consep",
        "dsb2018",
        "pannuke",
    ]
    seen = {r["dataset"] for r in all_rows}
    for ds in dataset_order:
        if ds in seen:
            lines.extend(_macro_table(all_rows, ds))
    for ds in sorted(seen - set(dataset_order)):
        lines.extend(_macro_table(all_rows, ds))

    # Cross-dataset average (methods present in both)
    lines.extend(["## Média macro combinada (todos os samples avaliados)", ""])
    by_br: dict[str, list[float]] = defaultdict(list)
    by_br_s: dict[str, list[float]] = defaultdict(list)
    by_fb: dict[str, list[float]] = defaultdict(list)
    by_fa: dict[str, list[float]] = defaultdict(list)
    by_dice: dict[str, list[float]] = defaultdict(list)
    for r in all_rows:
        by_br[r["method"]].append(float(r["br_mean_strict"]))
        if _finite_metric(r.get("br_seraph")):
            by_br_s[r["method"]].append(float(r["br_seraph"]))
        by_fb[r["method"]].append(float(r["fb_mean_strict"]))
        if r.get("f_area_mean_strict") not in (None, ""):
            by_fa[r["method"]].append(float(r["f_area_mean_strict"]))
        if r.get("pixel_dice") not in (None, ""):
            by_dice[r["method"]].append(float(r["pixel_dice"]))
    lines.extend([
        "| Método | BR macro | BR pooled | Fb | Fa | Dice | n total |",
        "|--------|--------:|----------:|---:|---:|-----:|--------:|",
    ])
    for m in METHODS:
        if by_br.get(m):
            lines.append(
                f"| `{m}` | {_fmt_mean(by_br[m])} | {_fmt_mean(by_br_s[m])} | "
                f"{_fmt_mean(by_fb[m])} | {_fmt_mean(by_fa[m])} | "
                f"{_fmt_mean(by_dice[m])} | {len(by_br[m])} |"
            )
    lines.append("")
    (OUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    # Fix imports in _paths - need to add DATA paths
    p = argparse.ArgumentParser(description=__doc__)
    patch_choices = tuple(PATCH_DATASETS.keys())
    p.add_argument(
        "--dataset",
        choices=("oral", "ihc", "both", "new4", "all") + patch_choices,
        default="both",
        help="new4=monuseg+consep+dsb2018+pannuke; all=oral+all patch datasets",
    )
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--metrics-only", action="store_true",
                   help="Only score existing masks (oral: reuse prior runs)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Parallel threads for SICLE+iDISF (overlap with GPU methods). Try 4-8.",
    )
    p.add_argument("--shard-id", type=int, default=0, help="Shard index for multi-process split")
    p.add_argument("--num-shards", type=int, default=1, help="Total shards (e.g. 2 processes: 0/2 and 1/2)")
    p.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Rescore/revisit all patches even when metrics rows already exist",
    )
    args = p.parse_args()

    methods = set(args.methods)
    rows: list[dict] = []
    csv_path = OUT_ROOT / "metrics_all_methods.csv"

    # Resume: load existing rows for samples already done
    done: set[tuple[str, str, str]] = set()
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                rows.append(r)
                if not args.no_skip_complete and _row_complete(r):
                    done.add((r["dataset"], r["sample_id"], r["method"]))

    run_infer = not args.metrics_only

    if args.dataset in ("oral", "both"):
        benchmark_oral(rows, max_samples=args.max_samples, methods=methods,
                       run_infer=run_infer, gpu=args.gpu, done=done)

    patch_sets = {
        "ihc": ("ihc_tma",),
        "new4": ("monuseg", "consep", "dsb2018", "pannuke"),
        "all": patch_choices,
    }
    if args.dataset in patch_sets:
        selected = patch_sets[args.dataset]
    elif args.dataset in PATCH_DATASETS:
        selected = (args.dataset,)
    elif args.dataset == "both":
        selected = ("ihc_tma",)
    else:
        selected = ()

    if args.dataset == "all":
        benchmark_oral(rows, max_samples=args.max_samples, methods=methods,
                       run_infer=run_infer, gpu=args.gpu, done=done)

    for ds_key in selected:
        benchmark_patch_dataset(
            rows,
            dataset=ds_key,
            data_root=PATCH_DATASETS[ds_key],
            category=ds_key,
            max_samples=args.max_samples,
            methods=methods,
            run_infer=run_infer,
            gpu=args.gpu,
            done=done,
            cpu_workers=max(1, args.cpu_workers),
            shard_id=args.shard_id,
            num_shards=max(1, args.num_shards),
            skip_complete=not args.no_skip_complete,
        )

    # Deduplicate (dataset, sample, method) keeping latest
    latest: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        latest[(r["dataset"], r["sample_id"], r["method"])] = r
    rows = list(latest.values())
    rows.sort(key=lambda r: (r["dataset"], r["sample_id"], r["method"]))

    _write_csv(csv_path, rows)
    write_summary(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {OUT_ROOT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
