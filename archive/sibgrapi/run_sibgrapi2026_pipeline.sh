#!/usr/bin/env bash
# Run full pipeline (Cellpose + per-cell SICLE) on all PNGs of data_sibgrapi2026.
#
# Layout produced (under OUT_ROOT):
#   <OUT_ROOT>/<stem>/cp_flow/        step01..step04 (reproduce_cellpose_pipeline.py)
#   <OUT_ROOT>/<stem>/sicle/          merged_percell_sicle_*.npy/.png + overlays + log
#
# Best config from prior runs (gradvmaxmul + minsc, α=2.0, threshold=0.3, disable AND)
# plus the fixes for cell 350-style flood (min_cell_area, and-unless-round, margin small).
# Tunable via env vars below.
#
# Usage:
#   bash run_sibgrapi2026_pipeline.sh
#   USE_GPU=0 bash run_sibgrapi2026_pipeline.sh
#   DATA_DIR=/path/to/pngs OUT_ROOT=./out_sibgrapi bash run_sibgrapi2026_pipeline.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
OUT_ROOT="${OUT_ROOT:-$HERE/out_sibgrapi2026}"
USE_GPU="${USE_GPU:-1}"

SICLE_CONN="${SICLE_CONN:-gradvmaxmul}"
SICLE_CRIT="${SICLE_CRIT:-minsc}"
SICLE_ALPHA="${SICLE_ALPHA:-2.0}"
SICLE_NF="${SICLE_NF:-2}"
SICLE_N0="${SICLE_N0:-200}"
SICLE_IRREG="${SICLE_IRREG:-0.0}"   # 0 = neutral (compat with old gradvmaxmul / gradvmax)
SICLE_ADHR="${SICLE_ADHR:-1}"        # 1 = no sharpening (compat with old gradvmaxmul / gradvmax)
SICLE_MAX_ITERS="${SICLE_MAX_ITERS:-7}"
SICLE_PEN_OPT="${SICLE_PEN_OPT:-none}"
SICLE_USE_RGB_IMAGE="${SICLE_USE_RGB_IMAGE:-0}"
SICLE_MIN_SOLIDITY="${SICLE_MIN_SOLIDITY:-0.0}"
SAL_BLUR="${SAL_BLUR:-0.5}"  # NEW best from literature sweep: blur cellprob with sigma=0.5
SAL_THR="${SAL_THR:-0.3}"
SAL_LINEARIZE="${SAL_LINEARIZE:-1}"  # 0 = sigmoid only (--no-saliency-linearize)
MARGIN="${MARGIN:-4}"
MIN_CELL_AREA="${MIN_CELL_AREA:-128}"
MIN_CIRC="${MIN_CIRC:-0.70}"
MIN_SOL="${MIN_SOL:-0.85}"
AND_UNLESS_ROUND="${AND_UNLESS_ROUND:-1}"
FILL_HOLES="${FILL_HOLES:-1}"
KEEP_LARGEST_CC="${KEEP_LARGEST_CC:-1}"
CLOSING_RADIUS="${CLOSING_RADIUS:-0}"
OVERLAY_SRC="${OVERLAY_SRC:-both}"
OVERLAY_CELLPOSE_COLOR="${OVERLAY_CELLPOSE_COLOR:-255,255,0}"
OVERLAY_SICLE_COLOR="${OVERLAY_SICLE_COLOR:-0,255,0}"

REPRODUCE_FLAGS=()
if [[ "$USE_GPU" == "1" ]]; then
  REPRODUCE_FLAGS+=("--gpu")
fi

mkdir -p "$OUT_ROOT"

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)
if [[ ${#PNGS[@]} -eq 0 ]]; then
  echo "No PNGs found in $DATA_DIR"
  exit 1
fi

echo "Pipeline on ${#PNGS[@]} PNGs from $DATA_DIR -> $OUT_ROOT"
echo "  conn=$SICLE_CONN crit=$SICLE_CRIT alpha=$SICLE_ALPHA sal_thr=$SAL_THR"
echo "  margin=$MARGIN min_cell_area=$MIN_CELL_AREA and-unless-round (circ>=$MIN_CIRC, sol>=$MIN_SOL)"

for png in "${PNGS[@]}"; do
  stem="$(basename "$png" .png)"
  case_dir="$OUT_ROOT/$stem"
  cp_dir="$case_dir/cp_flow"
  sicle_dir="$case_dir/sicle"
  mkdir -p "$case_dir"

  echo
  echo "============================================================"
  echo "[$stem]"
  echo "============================================================"

  # STEP A: Cellpose reproduce (only if step04 missing)
  if [[ -f "$cp_dir/step04_masks_uint16.npy" && -f "$cp_dir/step03_dP_cellprob.npz" ]]; then
    echo "  cellpose: skip (already done in $cp_dir)"
  else
    echo "  cellpose: $(date +%T) running reproduce_cellpose_pipeline.py ..."
    python3 reproduce_cellpose_pipeline.py "$png" \
      -o "$cp_dir" \
      "${REPRODUCE_FLAGS[@]}"
  fi

  # STEP B: per-cell SICLE merge (only if merged npy missing)
  if [[ -f "$sicle_dir/merged_percell_sicle_masks_int32.npy" ]]; then
    echo "  sicle:    skip (already done in $sicle_dir)"
  else
    echo "  sicle:    $(date +%T) running percell_sicle_cellprob_pipeline.py ..."
    python3 percell_sicle_cellprob_pipeline.py \
      --from-dir "$cp_dir" \
      -o "$sicle_dir" \
      --sicle-conn-opt "$SICLE_CONN" \
      --sicle-crit-opt "$SICLE_CRIT" \
      --sicle-pen-opt "$SICLE_PEN_OPT" \
      --sicle-min-solidity "$SICLE_MIN_SOLIDITY" \
      $( [[ "$SICLE_USE_RGB_IMAGE" == "1" ]] && echo "--sicle-use-rgb-image" ) \
      --sicle-alpha "$SICLE_ALPHA" \
      --sicle-nf "$SICLE_NF" \
      --sicle-n0 "$SICLE_N0" \
      --sicle-irreg "$SICLE_IRREG" \
      --sicle-adhr "$SICLE_ADHR" \
      --sicle-max-iters "$SICLE_MAX_ITERS" \
      --saliency-threshold "$SAL_THR" \
      --saliency-blur-sigma "$SAL_BLUR" \
      $( [[ "$SAL_LINEARIZE" == "0" ]] && echo "--no-saliency-linearize" ) \
      --margin "$MARGIN" \
      --min-cell-area "$MIN_CELL_AREA" \
      --disable-and-merge \
      $( [[ "$AND_UNLESS_ROUND" == "1" ]] && echo "--and-unless-round" ) \
      --min-fg-circularity "$MIN_CIRC" \
      --min-fg-solidity "$MIN_SOL" \
      $( [[ "$FILL_HOLES" == "1" ]] && echo "--fill-holes" ) \
      $( [[ "$KEEP_LARGEST_CC" == "1" ]] && echo "--keep-largest-cc" ) \
      --closing-radius "$CLOSING_RADIUS" \
      --image "$png" \
      --overlay-border-source "$OVERLAY_SRC" \
      --overlay-border-color "$OVERLAY_SICLE_COLOR" \
      --overlay-cellpose-border-color "$OVERLAY_CELLPOSE_COLOR" \
      --overlay-number-labels \
      --translucent-mask-overlay \
      --translucent-alpha "${TRANSLUCENT_ALPHA:-0.45}" \
      --write-compare-vs-step04
  fi
done

echo
echo "Done. Browse results under: $OUT_ROOT"
