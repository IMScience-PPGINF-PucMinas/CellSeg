#!/usr/bin/env python3
"""Backward-compatible alias for pipeline.evaluate_instances."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.evaluate_instances import *  # noqa: F401,F403
