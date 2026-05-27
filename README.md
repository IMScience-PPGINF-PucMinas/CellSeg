# CellSeg — Oral Epithelium pipeline

Research repo for **Cellpose + per-cell SICLE** instance segmentation on the **Oral Epithelium DB**, with strict per-cell Boundary Recall (BR) evaluation.

Previous SIBGRAPI 2026 benchmark artifacts were removed; legacy scripts live under `archive/sibgrapi/`.

## Layout

```
├── data/oral_epithelium/     # dataset (annotations tracked; TIFFs optional)
├── pipeline/                 # reusable segmentation & metrics
├── oral/                     # Oral-specific runners & panels
├── outputs/                  # generated runs (gitignored)
├── configs/                  # tuned SICLE parameters
├── cellpose/                 # vendored Cellpose for PYTHONPATH
├── archive/sibgrapi/         # old benchmark scripts (reference only)
└── run_oral_single_roi.sh    # demo: one ROI end-to-end
```

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `pillow`, `opencv-python`, `cellpose`
- **SICLE**: `RunSICLE` at `../SICLE/bin/RunSICLE` (or set `SICLE_BIN`)

```bash
pip install numpy scipy pillow opencv-python cellpose imageio tifffile
```

## Quick start (demo ROI `healthy-18-roi2`)

```bash
cd new_pipeline
chmod +x run_oral_single_roi.sh
./run_oral_single_roi.sh
```

Produces:

- `outputs/runs/single_roi/metrics_single_roi.csv`
- `outputs/runs/single_roi/healthy-18-roi2/{cp_flow,sicle,gt}/`

### Comparison panels

```bash
export PYTHONPATH="$(pwd):$(pwd)/pipeline:$(pwd)/cellpose"
python3 oral/build_comparison_panel.py
python3 oral/build_comparison_panel_sicle_config.py   # needs default+best runs
python3 oral/compare_sicle_configs_single_roi.py      # runs both SICLE configs + CSV
```

### Gold-standard review overlays (all 200 ROIs)

```bash
python3 oral/generate_gold_standard_overlays.py
# → outputs/reviews/gold_standard_overlays/
```

## Best SICLE config (`configs/sicle_blur05.args`)

- `gradvmaxmul` + `minsc`, α=2.0, N0=200
- Cellprob saliency: blur σ=0.5, threshold 0.3, Otsu linearization
- Post-process: disable-and-merge, and-unless-round, fill-holes, closing r=1

## Core scripts (`pipeline/`)

| Script | Role |
|--------|------|
| `reproduce_cellpose_pipeline.py` | Cellpose step-by-step → `cp_flow/` |
| `percell_sicle_cellprob_pipeline.py` | Per-cell SICLE on cellprob saliency |
| `evaluate_instances.py` | Dice, AJI, PQ, F1, mAP |
| `percell_boundary_recall.py` | BR + contour overlays |

`evaluate_sibgrapi2026.py` at repo root is a **deprecated alias** for `evaluate_instances.py`.

## Git history size

The `.git` folder may still contain old large blobs from SIBGRAPI runs. To shrink before push:

```bash
# optional — see docs/GIT_HISTORY.md
git filter-repo --path data/oral_epithelium --force  # example; read docs first
```

Working tree without `.git` is ~200 MB (with local TIFFs) or ~5 MB (annotations only).
