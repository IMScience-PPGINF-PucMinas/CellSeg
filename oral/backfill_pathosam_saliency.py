#!/usr/bin/env python3
"""Backfill PathoSAM foreground saliency maps (SERAPH-style) for existing runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _paths import DATA_IHC, PIPE, REPO, RUNS
from benchmark_postprocess_ablation import discover_rois
from method_infer import pathosam_saliency_path, run_pathosam

OUT_ROOT = RUNS / "all_methods_comparison"
CP_ROOT = RUNS / "postprocess_ablation_full"


def _oral_cases() -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    for category, stem in discover_rois():
        case = OUT_ROOT / "oral_epithelium" / category / stem
        flow = case / "pathosam_flow"
        image = CP_ROOT / category / stem / f"{stem}.png"
        if flow.is_dir() and image.is_file():
            out.append((image, flow))
    return out


def _ihc_cases() -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    for img in sorted((DATA_IHC / "images").glob("*.png")):
        flow = OUT_ROOT / "ihc_tma" / img.stem / "pathosam_flow"
        if flow.is_dir():
            out.append((img, flow))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("oral", "ihc", "both"), default="both")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute masks + saliency")
    args = p.parse_args()

    cases: list[tuple[Path, Path]] = []
    if args.dataset in ("oral", "both"):
        cases.extend(_oral_cases())
    if args.dataset in ("ihc", "both"):
        cases.extend(_ihc_cases())
    if args.max_samples > 0:
        cases = cases[: args.max_samples]

    device = "cuda" if args.gpu else "cpu"
    n_ok = 0
    for i, (image_path, flow_dir) in enumerate(cases, 1):
        sal = pathosam_saliency_path(flow_dir)
        if sal.is_file() and not args.force:
            print(f"[{i}/{len(cases)}] skip {image_path.stem}: saliency exists")
            n_ok += 1
            continue
        print(f"[{i}/{len(cases)}] {image_path.name} -> {flow_dir}")
        run_pathosam(
            image_path,
            flow_dir,
            device=device,
            tiled=True,
            save_saliency=True,
            force=args.force,
        )
        if pathosam_saliency_path(flow_dir).is_file():
            n_ok += 1
            print(f"    wrote {sal.name}")
        else:
            print("    ERROR: saliency missing")

    print(f"\nDone: {n_ok}/{len(cases)} with saliency map")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
