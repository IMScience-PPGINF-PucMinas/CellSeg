"""Shared paths for Oral Epithelium scripts (repo layout)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "oral_epithelium"
DATA_ORAL = DATA
DATA_IHC = REPO / "data" / "IHC_TMA_dataset"
DATA_MONUSEG = REPO / "data" / "monuseg"
DATA_DSB2018 = REPO / "data" / "dsb2018"
DATA_PANNUKE = REPO / "data" / "pannuke"
DATA_CONSEP = REPO / "data" / "consep"

PATCH_DATASETS: dict[str, Path] = {
    "ihc_tma": DATA_IHC,
    "monuseg": DATA_MONUSEG,
    "dsb2018": DATA_DSB2018,
    "pannuke": DATA_PANNUKE,
    "consep": DATA_CONSEP,
}
PIPE = REPO / "pipeline"
RUNS = REPO / "outputs" / "runs"
REVIEWS = REPO / "outputs" / "reviews"
CP_ROOT_ORAL = RUNS / "postprocess_ablation_full"

IMAGES_ORIGINAL = DATA / "images" / "original"
IMAGES_NORMALIZED = DATA / "images" / "normalized"
GT_COLORED = DATA / "annotations" / "instance_colored"
GT_INSTANCE = DATA / "annotations" / "instance"
GT_SEMANTIC = DATA / "annotations" / "semantic"

SINGLE_ROI_RUN = RUNS / "single_roi"
