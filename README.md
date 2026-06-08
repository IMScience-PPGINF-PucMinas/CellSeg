# CellSeg — nuclei instance segmentation & benchmarks

Research repo for **Cellpose + per-cell SICLE/iDISF** pipelines, strict per-cell **Boundary Recall (BR)** and **boundary F-measure (Fb)**, and unified comparison against **CellViT** and **PathoSAM**.

## Layout

```
├── data/
│   ├── oral_epithelium/          # 100 ROIs (annotations in git)
│   ├── IHC_TMA_dataset/          # README only — images local
│   ├── monuseg/ dsb2018/ consep/ pannuke/   # README only — prepare with tools/
│   └── readme.txt
├── pipeline/                     # reusable segmentation, metrics, viz
├── oral/                         # runners, benchmarks, panels
│   ├── benchmark_all_methods.py  # ★ main multi-dataset benchmark
│   ├── method_infer.py           # ★ method wrappers (CP, SICLE, iDISF, CellViT, PathoSAM)
│   └── _paths.py
├── tools/
│   └── prepare_benchmark_datasets.py
├── configs/                      # tuned SICLE .args files
├── docs/
│   ├── BENCHMARK.md              # ★ how to run & interpret benchmarks
│   ├── DATASETS.md               # dataset layout & download
│   └── oral_path_cost_exemplars.md
├── outputs/                      # generated runs (gitignored)
├── cellpose/                     # vendored Cellpose for PYTHONPATH
└── run_oral_single_roi.sh        # demo: one ROI end-to-end
```

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `pillow`, `opencv-python`, `cellpose`
- **SICLE**: `RunSICLE` at `../SICLE/bin/RunSICLE` (or `SICLE_BIN`)
- **iDISF**, **CellViT**, **micro_sam** (PathoSAM): sibling repos on `PYTHONPATH` (see `docs/BENCHMARK.md`)

```bash
pip install numpy scipy pillow opencv-python cellpose imageio tifffile
```

## Quick start

### Single ROI demo

```bash
cd new_pipeline
chmod +x run_oral_single_roi.sh
./run_oral_single_roi.sh
```

### Prepare datasets

```bash
python3 tools/prepare_benchmark_datasets.py --dataset all
```

### Multi-method benchmark

```bash
export PYTHONPATH="$(pwd):$(pwd)/oral:$(pwd)/pipeline:$(pwd)/cellpose:..:../iDISF/python3"
export SICLE_BIN="../SICLE/bin/RunSICLE"
python3 oral/benchmark_all_methods.py --dataset both --gpu
```

See **`docs/BENCHMARK.md`** for all datasets, metrics, resume/sharding, and disk tips.

## Core pipeline scripts

| Script | Role |
|--------|------|
| `reproduce_cellpose_pipeline.py` | Cellpose step-by-step → `cp_flow/` |
| `percell_sicle_cellprob_pipeline.py` | Per-cell SICLE on cellprob saliency |
| `percell_idisf_cellpose_pipeline.py` | Per-cell iDISF on Cellpose seeds |
| `percell_boundary_recall.py` | BR + contour overlays |
| `boundary_fb_metric.py` | Fb (Arbeláez) and Fa |

## Best SICLE config (`configs/sicle_blur05.args`)

- `gradvmaxmul` + `minsc`, α=2.0, N0=200
- Cellprob saliency: blur σ=0.5, threshold 0.3, Otsu linearization
- Post-process: disable-and-merge, and-unless-round, fill-holes, closing r=1

## Git & data size

- **Tracked:** code, configs, docs, oral annotations, dataset READMEs.
- **Ignored:** `outputs/`, dataset images/masks, checkpoints, logs, `exports/`.
- Working tree without local datasets: ~5 MB. With full benchmarks on disk: tens of GB (delete debug PNGs to reclaim space).
