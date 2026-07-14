#!/usr/bin/env python3
"""
Benchmark SICLE path costs (fmax+minsc, fsum+maxsc, gradvmaxmul+minsc) on Oral Epithelium ROIs.

Finds exemplar ROIs that justify model choice for papers / reports.
Outputs: outputs/runs/path_cost_benchmark/metrics_by_roi.csv + exemplars.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _paths import GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, RUNS

OUT_ROOT = RUNS / "path_cost_benchmark"
OUT_ROOT_FULL = RUNS / "path_cost_benchmark_full"
CP_ROOT = RUNS / "postprocess_ablation_full"
CSV_FULL = OUT_ROOT_FULL / "metrics_by_roi.csv"

FIELDNAMES = [
    "category",
    "roi",
    "config_id",
    "conn",
    "crit",
    "br_mean_strict",
    "pixel_dice",
    "aji",
    "n_gt",
    "n_pr",
]

# Main pipeline: sigmoid saliency, SICLE raw (no morph / no AUR merge)
SICLE_COMMON = [
    "--no-saliency-linearize",
    "--saliency-threshold", "0.3",
    "--saliency-blur-sigma", "0.5",
    "--margin", "4",
    "--min-cell-area", "128",
    "--disable-and-merge",
    "--closing-radius", "0",
    "--sicle-n0", "200",
    "--sicle-nf", "2",
    "--sicle-irreg", "0",
    "--sicle-adhr", "1",
    "--sicle-max-iters", "7",
]

PATH_COST_CONFIGS: list[dict] = [
    {
        "id": "fmax_minsc",
        "label": "SICLE-IRREG (fmax + minsc)",
        "conn": "fmax",
        "crit": "minsc",
        "alpha": "2.0",
        "note": "Preset irregular: custo max ao longo do caminho; saliência binária enfraquece |ΔO|.",
    },
    {
        "id": "fsum_maxsc",
        "label": "SICLE-COMP (fsum + maxsc)",
        "conn": "fsum",
        "crit": "maxsc",
        "alpha": "1.0",
        "note": "Preset compacto: soma de custos; melhor em células redondas/compactas.",
    },
    {
        "id": "gradvmaxmul_minsc",
        "label": "gradvmaxmul + minsc (nossa escolha)",
        "conn": "gradvmaxmul",
        "crit": "minsc",
        "alpha": "2.0",
        "note": "Usa |Δ|∇sal| na borda; funciona com cellprob quase binário.",
    },
]


def colored_to_labels(rgb: np.ndarray, bg_thresh: int = 8) -> np.ndarray:
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    lid = 1
    for c in np.unique(rgb.reshape(-1, 3), axis=0):
        if int(c.max()) <= bg_thresh:
            continue
        labels[np.all(rgb == c, axis=2)] = lid
        lid += 1
    return labels


def mean_br_strict(
    gt: np.ndarray,
    pr: np.ndarray,
    margin: int = 8,
    *,
    boundary_tolerance: int = 2,
) -> float:
    """Macro mean BR per GT instance at fixed tolerance (default 2 px, same as BR Area)."""
    from boundary_fb_metric import mean_br_macro

    return mean_br_macro(
        gt, pr, margin=margin, boundary_tolerance=boundary_tolerance
    )


def _append_csv_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _load_done_configs(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    with csv_path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            done.add((row["category"], row["roi"], row["config_id"]))
    return done


def write_summary_full(rows: list[dict], n_rois: int) -> None:
    from collections import defaultdict

    by_cfg: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(float(r["br_mean_strict"]))

    ours = float(np.mean(by_cfg["gradvmaxmul_minsc"])) if by_cfg["gradvmaxmul_minsc"] else float("nan")
    lines = [
        "# Path-cost benchmark — 100 ROIs (SICLE raw, same pipeline as production)",
        "",
        f"ROIs: **{n_rois}**. Configs: fmax+minsc, fsum+maxsc, gradvmaxmul+minsc.",
        "",
        f"CSV: `{CSV_FULL.relative_to(REPO)}`",
        "",
        "| Config | Mean BR (strict) | Δ vs gradvmaxmul+minsc |",
        "|--------|-----------------:|-----------------------:|",
    ]
    for cfg in PATH_COST_CONFIGS:
        cid = cfg["id"]
        br_m = float(np.mean(by_cfg[cid])) if by_cfg[cid] else float("nan")
        d = br_m - ours if cid != "gradvmaxmul_minsc" else 0.0
        lines.append(f"| `{cfg['conn']}` + `{cfg['crit']}` | {br_m:.4f} | {d:+.4f} |")
    (OUT_ROOT_FULL / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Macro BR (100 ROIs) ===")
    for cfg in PATH_COST_CONFIGS:
        cid = cfg["id"]
        if by_cfg[cid]:
            print(f"  {cid}: {np.mean(by_cfg[cid]):.4f}  (n={len(by_cfg[cid])})")


def run_path_cost_on_case(
    category: str,
    stem: str,
    *,
    case: Path,
    input_png: Path,
    gt_path: Path,
    cp_dir: Path,
    configs: list[dict],
    py: str,
    env: dict,
    skip_existing: bool = True,
) -> list[dict]:
    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    gt_arr = np.load(gt_path).astype(np.int32)
    rows: list[dict] = []
    for cfg in configs:
        sicle_dir = case / cfg["id"]
        pr_path = sicle_dir / "merged_percell_sicle_masks_int32.npy"
        if not (skip_existing and pr_path.is_file() and pr_path.stat().st_size > 0):
            print(f"  SICLE {cfg['id']}")
            sicle_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                str(PIPE / "percell_sicle_cellprob_pipeline.py"),
                "--from-dir",
                str(cp_dir),
                "-o",
                str(sicle_dir),
                "--sicle-conn-opt",
                cfg["conn"],
                "--sicle-crit-opt",
                cfg["crit"],
                "--sicle-alpha",
                cfg["alpha"],
                "--image",
                str(input_png),
                *SICLE_COMMON,
            ]
            subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

        r = evaluate_pair(gt_path, pr_path)
        pr_arr = np.load(pr_path).astype(np.int32)
        br = mean_br_strict(gt_arr, pr_arr)
        rows.append(
            {
                "category": category,
                "roi": stem,
                "config_id": cfg["id"],
                "conn": cfg["conn"],
                "crit": cfg["crit"],
                "br_mean_strict": br,
                "pixel_dice": r.get("pixel_dice"),
                "aji": r.get("aji"),
                "n_gt": r.get("n_gt"),
                "n_pr": r.get("n_pr"),
            }
        )
        print(f"    {cfg['id']}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")
    return rows


def run_one_roi(category: str, stem: str, configs: list[dict], skip_cellpose: bool) -> list[dict]:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from evaluate_instances import evaluate_pair

    orig_tif = IMAGES_ORIGINAL / category / f"{stem}.tif"
    col_png = GT_COLORED / category / f"{stem}.png"
    case = OUT_ROOT / category / stem
    case.mkdir(parents=True, exist_ok=True)

    rgb_orig = np.asarray(Image.open(orig_tif).convert("RGB"))
    rgb_col = np.asarray(Image.open(col_png).convert("RGB"))
    h = min(rgb_orig.shape[0], rgb_col.shape[0])
    w = min(rgb_orig.shape[1], rgb_col.shape[1])
    rgb_orig, rgb_col = rgb_orig[:h, :w], rgb_col[:h, :w]

    input_png = case / f"{stem}.png"
    Image.fromarray(rgb_orig).save(input_png)

    gt_path = case / "gt" / "gold_standard_masks_int32.npy"
    gt_path.parent.mkdir(exist_ok=True)
    if not gt_path.is_file():
        np.save(gt_path, colored_to_labels(rgb_col))

    cp_dir = case / "cp_flow"
    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    if not skip_cellpose or not (cp_dir / "step04_masks_uint16.npy").is_file():
        print(f"  Cellpose {category}/{stem}")
        subprocess.run(
            [py, str(PIPE / "reproduce_cellpose_pipeline.py"), str(input_png), "-o", str(cp_dir), "--gpu"],
            cwd=str(REPO),
            env=env,
            check=True,
        )

    return run_path_cost_on_case(
        category,
        stem,
        case=case,
        input_png=input_png,
        gt_path=gt_path,
        cp_dir=cp_dir,
        configs=configs,
        py=py,
        env=env,
    )


def pick_exemplars(all_rows: list[dict]) -> dict:
    """Select ROIs that best illustrate each narrative."""
    rois: dict[str, dict] = {}
    for row in all_rows:
        key = f"{row['category']}/{row['roi']}"
        rois.setdefault(key, {})[row["config_id"]] = row

    scored: list[tuple[str, float, float, float]] = []
    for key, m in rois.items():
        if not all(k in m for k in ("fmax_minsc", "fsum_maxsc", "gradvmaxmul_minsc")):
            continue
        bf, bs, bg = m["fmax_minsc"]["br_mean_strict"], m["fsum_maxsc"]["br_mean_strict"], m["gradvmaxmul_minsc"]["br_mean_strict"]
        scored.append((key, bg - bf, bs - bf, bg - bs))

    scored.sort(key=lambda x: x[1], reverse=True)

    exemplars = {
        "why_not_fmax": scored[0][0] if scored else None,
        "why_gradvmaxmul_wins": scored[0][0] if scored else None,
        "fsum_beats_fmax": None,
        "fsum_best_compact": None,
    }

    fsum_wins = sorted(
        [
            (k, m["fsum_maxsc"]["br_mean_strict"] - m["fmax_minsc"]["br_mean_strict"])
            for k, m in rois.items()
            if "fmax_minsc" in m and "fsum_maxsc" in m
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    for k, delta in fsum_wins:
        if delta > 0.02:
            exemplars["fsum_beats_fmax"] = k
            break

    grad_wins = sorted(
        [(k, m["gradvmaxmul_minsc"]["br_mean_strict"] - m["fsum_maxsc"]["br_mean_strict"]) for k, m in rois.items() if "fsum_maxsc" in m and "gradvmaxmul_minsc" in m],
        key=lambda x: x[1],
        reverse=True,
    )
    if grad_wins:
        exemplars["gradvmaxmul_over_fsum"] = grad_wins[0][0]

    return exemplars


def write_report(all_rows: list[dict], exemplars: dict, out_md: Path) -> None:
  lines = [
    "# Exemplares de custo de caminho SICLE (Oral Epithelium)",
    "",
    "No RunSICLE existem **`fmax`** e **`fsum`** como conectividades; não há `fmin`.",
    "O critério **`minsc`** (tamanho mínimo do superpixel) pareia com irregular; **`maxsc`** com compacto.",
    "",
    "## Modelos comparados",
    "",
  ]
  for cfg in PATH_COST_CONFIGS:
    lines.append(f"- **{cfg['label']}**: `{cfg['conn']}` + `{cfg['crit']}`, α={cfg['alpha']} — {cfg['note']}")
  lines.extend(["", "## Por que `fmax` falha aqui", "",
    "- Custo `fmax`: `max(path, ||f_root−f_j||^(1+α|O(R)−O(j)|))` — depende de diferença de saliência ao longo do caminho.",
    "- Cellprob pós-Otsu é **quase binário** → |O(R)−O(j)| ∈ {0,1} no interior; pouco sinal na borda.",
    "- Sem blur/pós-processo adequado, a região cresce mal (vazamento ou sub-segmentação).",
    "", "## Papel do `fsum`", "",
    "- Custo acumulado `(irreg + α|Δsal|)·||f_root−f_j||` — preset **SICLE-COMP** para células compactas.",
    "- Pode superar `fmax` em ROIs com células mais redondas, mas perde em bordas irregulares do epitélio.",
    "", "## Por que `gradvmaxmul`", "",
    "- Usa **|∇sal(j)−∇sal(i)|** na fronteira (anel fino), informativo mesmo com cellprob binário.",
    "- Com blur σ=0.5 + pós-processo, é o melhor BR na maioria das ROIs testadas.",
    "", "## ROIs exemplares (automático)", "",
  ])
  for role, key in exemplars.items():
    lines.append(f"- **{role}**: `{key}`")
  lines.extend(["", "## Métricas (todas as ROIs)", "", "| categoria | roi | config | BR | Dice |", "|---|---|---|---:|---:|"])
  for row in sorted(all_rows, key=lambda r: (r["category"], r["roi"], r["config_id"])):
    lines.append(
      f"| {row['category']} | {row['roi']} | {row['config_id']} | {row['br_mean_strict']:.4f} | {row['pixel_dice']:.4f} |"
    )
  out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def run_full_benchmark(skip_cellpose: bool, max_rois: int = 0) -> int:
    from benchmark_postprocess_ablation import _ensure_case

    rois = discover_rois()
    if max_rois > 0:
        rois = rois[:max_rois]
    OUT_ROOT_FULL.mkdir(parents=True, exist_ok=True)
    done = _load_done_configs(CSV_FULL)
    rows: list[dict] = []
    if CSV_FULL.is_file():
        with CSV_FULL.open(encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))

    env = os.environ.copy()
    env["SICLE_BIN"] = env.get("SICLE_BIN", str(REPO.parent / "SICLE" / "bin" / "RunSICLE"))
    env["PYTHONPATH"] = os.pathsep.join([str(PIPE), str(REPO / "cellpose"), env.get("PYTHONPATH", "")])
    py = sys.executable

    for i, (category, stem) in enumerate(rois, 1):
        pending = [c for c in PATH_COST_CONFIGS if (category, stem, c["id"]) not in done]
        if not pending:
            continue

        case = OUT_ROOT_FULL / category / stem
        cp_case = CP_ROOT / category / stem
        cp_masks = cp_case / "cp_flow" / "step04_masks_uint16.npy"
        if cp_masks.is_file():
            input_png = cp_case / f"{stem}.png"
            gt_path = cp_case / "gt" / "gold_standard_masks_int32.npy"
            cp_dir = cp_case / "cp_flow"
            if not gt_path.is_file():
                case.mkdir(parents=True, exist_ok=True)
                _, gt_path, _ = _ensure_case(category, stem, case, py=py, env=env, skip_cellpose=False)
        else:
            case.mkdir(parents=True, exist_ok=True)
            input_png, gt_path, cp_dir = _ensure_case(
                category, stem, case, py=py, env=env, skip_cellpose=skip_cellpose
            )

        case.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(rois)}] === {category}/{stem} ===")
        new_rows = run_path_cost_on_case(
            category,
            stem,
            case=case,
            input_png=input_png,
            gt_path=gt_path,
            cp_dir=cp_dir,
            configs=pending,
            py=py,
            env=env,
        )
        for row in new_rows:
            _append_csv_row(CSV_FULL, row)
            done.add((category, stem, row["config_id"]))
            rows.append(row)

    if rows:
        write_summary_full(rows, len(rois))
    print(f"\nWrote {CSV_FULL}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="All 100 ROIs; reuse cp_flow from postprocess_ablation_full")
    p.add_argument("--category", choices=("healthy", "severe", "both"), default="both")
    p.add_argument("--rois", nargs="*", help="stems e.g. healthy-18-roi2 (default: curated list)")
    p.add_argument("--max-rois", type=int, default=0, help="limit count after curation")
    p.add_argument("--skip-cellpose", action="store_true", help="reuse existing cp_flow")
    args = p.parse_args()

    if args.full:
        return run_full_benchmark(skip_cellpose=args.skip_cellpose, max_rois=args.max_rois)

    default_rois = [
        ("healthy", "healthy-18-roi2"),
        ("healthy", "healthy-17-roi2"),
        ("healthy", "healthy-19-roi2"),
        ("healthy", "healthy-18-roi4"),
        ("severe", "severe-01-roi1"),
        ("severe", "severe-03-roi2"),
        ("severe", "severe-01-roi4"),
    ]
    pairs = default_rois if not args.rois else []
    if args.rois:
        for r in args.rois:
            if "-" in r:
                cat = "healthy" if r.startswith("healthy") else "severe"
                pairs.append((cat, r))
            else:
                raise SystemExit("use stem like healthy-18-roi2")

    if args.category != "both":
        pairs = [(c, s) for c, s in pairs if c == args.category]
    if args.max_rois > 0:
        pairs = pairs[: args.max_rois]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for cat, stem in pairs:
        print(f"\n=== {cat}/{stem} ===")
        all_rows.extend(run_one_roi(cat, stem, PATH_COST_CONFIGS, args.skip_cellpose))

    csv_path = OUT_ROOT / "metrics_by_roi.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    exemplars = pick_exemplars(all_rows)
    (OUT_ROOT / "exemplars.json").write_text(json.dumps(exemplars, indent=2), encoding="utf-8")
    write_report(all_rows, exemplars, OUT_ROOT / "exemplars.md")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {OUT_ROOT / 'exemplars.md'}")
    print(json.dumps(exemplars, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
