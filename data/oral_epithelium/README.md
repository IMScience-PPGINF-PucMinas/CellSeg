# Oral Epithelium DB (in-repo layout)

Dataset: **Oral Epithelium DB** — 228 H&E patches (114 healthy, 114 severe). **100 patches** (50 per class) have per-cell instance gold standard; the other **128** have semantic labels only (no instance GT).

## Layout

```
data/oral_epithelium/
├── images/
│   ├── original/{healthy,severe}/*.tif    # ~75 MB (gitignored by default)
│   └── normalized/{healthy,severe}/*.tif  # optional, gitignored
└── annotations/
    ├── instance_colored/{healthy,severe}/*.png   # gold standard (colored instances)
    ├── instance/{healthy,severe}/*.png           # grayscale instance masks
    └── semantic/{healthy,severe}/*.png           # semantic labels
```

## Git

By default only **annotations** are tracked (~2 MB). Original TIFFs are listed in `.gitignore` because of size.

To version images with [Git LFS](https://git-lfs.github.com/):

1. `git lfs track "data/oral_epithelium/images/**/*.tif"`
2. Remove the `images/original/` and `images/normalized/` lines from the repo `.gitignore`
3. `git add .gitattributes data/oral_epithelium/images/`

## Citation

Use the citation from the Oral Epithelium DB publication when publishing results.
