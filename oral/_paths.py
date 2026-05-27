"""Shared paths for Oral Epithelium scripts (repo layout)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "oral_epithelium"
PIPE = REPO / "pipeline"
RUNS = REPO / "outputs" / "runs"
REVIEWS = REPO / "outputs" / "reviews"

IMAGES_ORIGINAL = DATA / "images" / "original"
IMAGES_NORMALIZED = DATA / "images" / "normalized"
GT_COLORED = DATA / "annotations" / "instance_colored"
GT_INSTANCE = DATA / "annotations" / "instance"
GT_SEMANTIC = DATA / "annotations" / "semantic"

SINGLE_ROI_RUN = RUNS / "single_roi"
