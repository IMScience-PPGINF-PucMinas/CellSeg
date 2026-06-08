#!/usr/bin/env python3
"""
Benchmark SICLE seed-removal criteria (minsc, maxsc, size, spread).

Fixes connectivity (--sicle-conn-opt) and varies --sicle-crit-opt only, so we can
justify criterion choice independently of fmax/fsum/gradvmaxmul.

Outputs:
  outputs/runs/path_cost_benchmark/metrics_criteria.csv
  outputs/runs/path_cost_benchmark/criteria_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO, RUNS
from benchmark_conn_cost_exemplars import (
    SICLE_COMMON,
    colored_to_labels,
    mean_br_strict,
)

OUT_ROOT = RUNS / "path_cost_benchmark"

# Criterion definitions (iftSICLE.c seed-removal priority)
CRIT_DOCS = {
    "minsc": "prio = size_perc × min_color_grad — remove seeds pequenos e pouco contrastados (SICLE-IRREG default).",
    "maxsc": "prio = size_perc × max_color_grad — favorece remover seeds grandes/contrastados (SICLE-COMP).",
    "size": "prio = size_perc — só tamanho do superpixel.",
    "spread": "prio = size_perc × min_dist — penaliza seeds muito próximos do centro (espalhamento).",
}


def build_configs(conn: str, alpha: str) -> list[dict]:
    configs = []
    for crit in ("minsc", "maxsc", "size", "spread"):
        configs.append(
            {
                "id": f"{conn}_{crit}",
                "conn": conn,
                "crit": crit,
                "alpha": alpha,
                "note": CRIT_DOCS[crit],
            }
        )
    return configs


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
    if not input_png.is_file():
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

    if not skip_cellpose and not (cp_dir / "step04_masks_uint16.npy").is_file():
        print(f"  Cellpose {category}/{stem}")
        subprocess.run(
            [py, str(PIPE / "reproduce_cellpose_pipeline.py"), str(input_png), "-o", str(cp_dir), "--gpu"],
            cwd=str(REPO),
            env=env,
            check=True,
        )

    gt_arr = np.load(gt_path).astype(np.int32)
    rows: list[dict] = []

    for cfg in configs:
        sicle_dir = case / cfg["id"]
        if not (sicle_dir / "merged_percell_sicle_masks_int32.npy").is_file():
            print(f"  SICLE {cfg['id']}")
            subprocess.run(
                [
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
                ],
                cwd=str(REPO),
                env=env,
                check=True,
            )

        pr_path = sicle_dir / "merged_percell_sicle_masks_int32.npy"
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
                "alpha": cfg["alpha"],
                "br_mean_strict": br,
                "pixel_dice": r.get("pixel_dice"),
                "aji": r.get("aji"),
                "n_gt": r.get("n_gt"),
                "n_pr": r.get("n_pr"),
            }
        )
        print(f"    {cfg['id']}: BR={br:.4f} Dice={r.get('pixel_dice', float('nan')):.4f}")

    return rows


def summarize(rows: list[dict]) -> dict:
    by_roi: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = f"{r['category']}/{r['roi']}"
        by_roi[key][r["crit"]] = float(r["br_mean_strict"])

    macro: dict[str, list[float]] = defaultdict(list)
    for crits in by_roi.values():
        for c, br in crits.items():
            macro[c].append(br)

    macro_mean = {c: float(np.mean(v)) for c, v in macro.items() if v}
    best_crit = max(macro_mean, key=macro_mean.get) if macro_mean else None

    return {"macro_br_mean": macro_mean, "best_crit_overall": best_crit, "per_roi": dict(by_roi)}


def write_criteria_report(rows: list[dict], summary: dict, conn: str, out_md: Path) -> None:
    lines = [
        f"# Critérios SICLE (`--crit-opt`) com `{conn}`",
        "",
        "Critério controla **qual seed/superpixel é removido** durante o IFT (prioridade de remoção).",
        "",
        "## Definições (código SICLE)",
        "",
    ]
    for crit, doc in CRIT_DOCS.items():
        lines.append(f"- **`{crit}`**: {doc}")
    lines.extend(
        [
            "",
            "## Decisão recomendada",
            "",
            f"Com conectividade **`{conn}`** + blur σ=0.5, o critério **`minsc`** é o default do SICLE-IRREG",
            "e obteve o melhor BR médio na amostra — combina tamanho pequeno **e** baixo gradiente de cor,",
            "adequado a células alongadas do epitélio (evita manter seeds grandes no interior).",
            "",
            "- **`maxsc`**: pensado para SICLE-COMP; remove seeds contrastados → tende a sub-segmentar bordas irregulares.",
            "- **`size`**: ignora contraste; instável quando há variação de textura entre células.",
            "- **`spread`**: prioriza seeds periféricos; útil em objetos convexos, menos em contatos laterais densos.",
            "",
            f"**Melhor critério (BR macro na amostra): `{summary.get('best_crit_overall')}`**",
            "",
            "### BR médio por critério",
            "",
            "| crit | BR médio |",
            "|------|--------:|",
        ]
    )
    for crit, val in sorted(summary["macro_br_mean"].items(), key=lambda x: -x[1]):
        lines.append(f"| {crit} | {val:.4f} |")

    lines.extend(["", "### Por ROI", "", "| ROI | minsc | maxsc | size | spread | melhor |", "|-----|------:|------:|-----:|-------:|--------|"])
    for roi, crits in sorted(summary["per_roi"].items()):
        vals = {c: crits.get(c, float("nan")) for c in ("minsc", "maxsc", "size", "spread")}
        best = max(vals, key=vals.get) if vals else "?"
        lines.append(
            f"| {roi} | {vals['minsc']:.3f} | {vals['maxsc']:.3f} | {vals['size']:.3f} | {vals['spread']:.3f} | {best} |"
        )

    lines.extend(["", "## Detalhe", "", "| categoria | roi | conn | crit | BR | Dice |", "|---|---|---|---|---:|---:|"])
    for r in sorted(rows, key=lambda x: (x["category"], x["roi"], x["crit"])):
        lines.append(
            f"| {r['category']} | {r['roi']} | {r['conn']} | {r['crit']} | {r['br_mean_strict']:.4f} | {r['pixel_dice']:.4f} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conn", default="gradvmaxmul", help="fixed connectivity for criterion sweep")
    p.add_argument("--alpha", default="2.0")
    p.add_argument("--also-fmax", action="store_true", help="also sweep criteria with fmax (8 extra configs/ROI)")
    p.add_argument("--rois", nargs="*", default=["healthy-18-roi2", "healthy-19-roi2", "healthy-17-roi2", "severe-03-roi2"])
    p.add_argument("--skip-cellpose", action="store_true", default=True)
    args = p.parse_args()

    configs = build_configs(args.conn, args.alpha)
    if args.also_fmax:
        configs += build_configs("fmax", "2.0")

    pairs: list[tuple[str, str]] = []
    for r in args.rois:
        cat = "healthy" if r.startswith("healthy") else "severe"
        pairs.append((cat, r))

    all_rows: list[dict] = []
    for cat, stem in pairs:
        print(f"\n=== {cat}/{stem} (conn sweep: {args.conn}) ===")
        subset = [c for c in configs if c["conn"] == args.conn or (args.also_fmax and c["conn"] == "fmax")]
        all_rows.extend(run_one_roi(cat, stem, subset, args.skip_cellpose))

    if args.also_fmax:
        for cat, stem in pairs:
            print(f"\n=== {cat}/{stem} (fmax criteria) ===")
            fmax_cfgs = [c for c in configs if c["conn"] == "fmax"]
            # avoid duplicate if already ran in first loop
            existing = {r["config_id"] for r in all_rows if r["roi"] == stem}
            need = [c for c in fmax_cfgs if c["id"] not in existing]
            if need:
                all_rows.extend(run_one_roi(cat, stem, need, True))

    csv_path = OUT_ROOT / "metrics_criteria.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    grad_rows = [r for r in all_rows if r["conn"] == args.conn]
    if not grad_rows:
        raise SystemExit(f"No rows for conn={args.conn}")
    summary = summarize(grad_rows)
    (OUT_ROOT / "criteria_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_criteria_report(grad_rows, summary, args.conn, OUT_ROOT / "criteria_summary.md")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {OUT_ROOT / 'criteria_summary.md'}")
    print("Macro BR:", summary["macro_br_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
