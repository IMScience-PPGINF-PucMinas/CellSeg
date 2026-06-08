# Datasets

All patch benchmarks share one layout:

```
data/<dataset>/
  images/<sample_id>.png
  masks/<sample_id>.npy   # int32 instance map (0=bg, 1..N=nucleus id)
  README.md
```

## Included in git

| Dataset | README | Images/masks |
|---------|--------|--------------|
| `oral_epithelium` | yes | annotations tracked; TIFFs optional (see `.gitignore`) |
| `monuseg`, `dsb2018`, `consep`, `pannuke`, `ihc_tma` | yes | **not** in git — download locally |

## Prepare patch datasets

From repo root (`new_pipeline/`):

```bash
python3 tools/prepare_benchmark_datasets.py --dataset all
# or: monuseg | dsb2018 | consep | pannuke
```

IHC TMA: place the published cohort under `data/IHC_TMA_dataset/` (see `data/readme.txt` for citation).

## Paths in code

`oral/_paths.py` defines `PATCH_DATASETS` and `DATA_ORAL` used by all benchmarks.
