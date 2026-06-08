# Multi-method benchmark

Unified comparison of **BR**, **Fb**, **Fa** and **Dice** across five methods:

| Method | Description |
|--------|-------------|
| `cellpose` | Cellpose alone |
| `sicle_percell` | Per-cell SICLE on Cellpose seeds (Nf=2, raw) |
| `idisf_percell` | Per-cell iDISF on Cellpose seeds |
| `cellvit` | CellViT-256 |
| `pathosam` | PathoSAM (`vit_l_histopathology` via micro_sam) |

## Metrics

- **BR** — per-cell strict boundary recall (GT contour pixels recovered).
- **Fb** — Arbeláez F-measure on 1 px contours (tolerance 0.0075×diagonal).
- **Fa** — per-cell pixel F1 (area), macro mean.
- **Dice** — ROI-level pixel F1 on merged foreground.

Implementation: `pipeline/percell_boundary_recall.py`, `pipeline/boundary_fb_metric.py`.

## Run

```bash
cd new_pipeline
export PYTHONPATH="$(pwd):$(pwd)/oral:$(pwd)/pipeline:$(pwd)/cellpose:..:../iDISF/python3"
export SICLE_BIN="/path/to/SICLE/bin/RunSICLE"

# Oral + IHC (default)
python3 oral/benchmark_all_methods.py --dataset both --gpu

# New patch benchmarks
python3 oral/benchmark_all_methods.py --dataset new4 --gpu --cpu-workers 8

# Single dataset (resumable; skips complete patches by default)
python3 oral/benchmark_all_methods.py --dataset pannuke --gpu --cpu-workers 8
```

Outputs (gitignored):

- `outputs/runs/all_methods_comparison/metrics_all_methods.csv`
- `outputs/runs/all_methods_comparison/summary.md`
- `outputs/runs/all_methods_comparison/benchmark_progress.json`

Rescore existing masks without re-inference:

```bash
python3 oral/benchmark_all_methods.py --dataset all --metrics-only
```

## Related experiments

| Script | Purpose |
|--------|---------|
| `oral/benchmark_idisf_merged_seeds.py` | Cellpose + PathoSAM seeds → iDISF |
| `oral/backfill_pathosam_saliency.py` | PathoSAM foreground prob (SERAPH-style) |
| `oral/regenerate_summaries.py` | Rebuild summary from CSV on disk |

## Disk usage

PanNuke runs generate many debug PNGs under `outputs/runs/all_methods_comparison/<dataset>/`.
Keep `.npy` masks; delete `percell_cell_outputs/` and overlay PNGs to reclaim space.
