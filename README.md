# CellSeg / New Pipeline

This repository contains the segmentation pipeline we are developing for cell/nuclei analysis.
The current focus is:

- Reproduce Cellpose/CP-SAM inference step by step
- Generate per-cell crops and scribble annotations
- Run alternative scribble-based segmenters (iDISF, PyIFT, UOIFT, SICLE, fusion)
- Compare outputs from different pipeline steps

It is a practical research repo (scripts + outputs), not a packaged library.

## Repository Structure (main files)

- `reproduce_cellpose_pipeline.py`: full step-by-step Cellpose flow with intermediate artifacts (`dP`, `cellprob`, masks)
- `cellpose_to_idisf_pipeline.py`: Cellpose -> crop each cell -> build scribbles -> run selected segmenter
- `cellpose_masks_modified_cellprob.py`: remix/modify masks using cellprob-related logic
- `compare_segmentation_masks_diff.py`: compare two instance label maps and export difference images/stats
- `percell_sicle_cellprob_pipeline.py`: per-cell bounding-box SICLE on cropped `cellprob` saliency → merged mask (`step03` + `step04` from `cp_flow_out`)
- `mask_outline_overlay.py`: border-only overlay (TIFF/SVS + mask `.npy`/`.npz`) without re-running segmentation
- `run_monuseg_cellpose_nuclick.py`: MoNuSeg experiment (Cellpose centroids -> NuClick)
- `run_monuseg_cellpose_sicle.py`: MoNuSeg experiment (Cellpose crops -> SICLE)


Common output folders in this repo:

- `cp_flow_out/`
- `compare_out/`
- `compare_step4_vs_fused/`
- `remix_out/`

## Minimal Environment

Use Python 3.10+ (or similar) and install the libraries needed by the scripts you run.
Core dependencies used across scripts include:

- `numpy`
- `scipy`
- `pillow`
- `imageio`
- `tifffile` (optional but useful)
- `cellpose`

Depending on the pipeline branch, you may also need:

- iDISF Python bindings (`idisf`) built locally
- `pyift`
- SICLE binary (`RunSICLE`) available by path or `SICLE_BIN`
- NuClick weights (for the NuClick script)

## Quick Start

Run from repository root:

```bash
cd /home/lacerda/doutorado/new_pipeline
```

### 1) Reproduce Cellpose inference artifacts

```bash
PYTHONPATH=./cellpose python reproduce_cellpose_pipeline.py \
  GR07-1.svs_slice1.tiff \
  -o ./cp_flow_out \
  --gpu
```

This generates step files like:

- `step01_preprocessed_x.npy`
- `step03_dP_cellprob.npz`
- `step04_masks_uint16.npy`
- `manifest.txt`

### 2) Run Cellpose -> crop/scribble -> segmenter pipeline

Example with SICLE:

```bash
PYTHONPATH=./cellpose python cellpose_to_idisf_pipeline.py \
  --image GR07-1.svs_slice1.tiff \
  --out_dir ./cellpose_idisf_out \
  --segmenter sicle
```

Example with fusion (`idisf + pyift + sicle` majority vote):

```bash
PYTHONPATH=./cellpose python cellpose_to_idisf_pipeline.py \
  --image GR07-1.svs_slice1.tiff \
  --out_dir ./cellpose_idisf_out \
  --segmenter fusion
```

### 3) Compare two mask outputs

```bash
python compare_segmentation_masks_diff.py \
  --mask-a cp_flow_out/step04_masks_uint16.npy \
  --mask-b remix_out/remix_arrays.npz \
  -o compare_out \
  --also-save-diff-only-rgb
```

### 4) Per-cell SICLE on `cellprob` (bbox + merged mask)

Requires `cp_flow_out/step03_dP_cellprob.npz` and `step04_masks_uint16.npy`. From this folder, point `PYTHONPATH` at the repo’s Cellpose package (sibling of `new_pipeline/`):

```bash
PYTHONPATH=../cellpose python percell_sicle_cellprob_pipeline.py \
  --from-dir ./cp_flow_out -o ./percell_sicle_out \
  --image ../GR07-1.svs_slice1.tiff
```

### 5) Border-only overlay (TIFF / SVS + mask file)

```bash
PYTHONPATH=../cellpose python mask_outline_overlay.py \
  --image ../data/my_slice.tif \
  --masks ./cp_flow_out/step04_masks_uint16.npy \
  -o ./viz/step04_outline.png
```

## Notes

- Input to comparison script must be label maps (`.npy`, `.npz`, or single-channel TIFF), not RGB previews.
- Some scripts are GPU-aware; use `--no-gpu` when needed.
- Large generated artifacts are intentionally ignored via `.gitignore` to avoid push problems.
- This repo includes experimental outputs and scripts under active iteration.