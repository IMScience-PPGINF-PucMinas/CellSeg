# IHC TMA cohort (NSCLC)

- **Task:** Multiplex IHC nuclei instance delineation (256×256 patches)
- **Samples:** 266 annotated patches (195 train / 36 val / 35 test in original split)
- **Layout:** `images/*.png`, `masks/*.npy` (3-channel mask; see `data/readme.txt`)
- **Source:** Liaoning Cancer Hospital cohort (SRSA-Net paper)

**Not tracked in git** — download or copy the published dataset locally, then run benchmarks with `--dataset ihc`.
