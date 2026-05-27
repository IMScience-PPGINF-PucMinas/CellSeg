#!/usr/bin/env bash
# Run Cellpose + SICLE (blur05) + metrics on the demo ROI healthy-18-roi2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/pipeline:${ROOT}/cellpose:${PYTHONPATH:-}"
export SICLE_BIN="${SICLE_BIN:-${ROOT}/../SICLE/bin/RunSICLE}"
cd "$ROOT"
exec python3 oral/run_single_roi_cellpose_sicle_test.py "$@"
