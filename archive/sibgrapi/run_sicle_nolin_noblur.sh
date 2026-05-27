#!/usr/bin/env bash
# SICLE: sigmoid only (no Otsu linearize) + no Gaussian blur on saliency.
# Reuses Cellpose from out_sibgrapi2026_blur05 -> out_sibgrapi2026_nolin_noblur

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
CP_ROOT="${CP_ROOT:-$HERE/out_sibgrapi2026_blur05}"
OUT_ROOT="${OUT_ROOT:-$HERE/out_sibgrapi2026_nolin_noblur}"

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)
if [[ ${#PNGS[@]} -eq 0 ]]; then
  echo "No PNGs in $DATA_DIR"
  exit 1
fi

mkdir -p "$OUT_ROOT"
echo "SICLE: no linearize, no blur (thr=0.3) -> $OUT_ROOT"
echo "Cellpose from: $CP_ROOT"

for png in "${PNGS[@]}"; do
  stem="$(basename "$png" .png)"
  cp_dir="$CP_ROOT/$stem/cp_flow"
  sicle_dir="$OUT_ROOT/$stem/sicle"
  if [[ ! -f "$cp_dir/step04_masks_uint16.npy" ]]; then
    echo "Missing Cellpose for $stem in $cp_dir"
    exit 1
  fi
  mkdir -p "$OUT_ROOT/$stem"
  if [[ ! -e "$OUT_ROOT/$stem/cp_flow" ]]; then
    ln -sfn "$(readlink -f "$cp_dir")" "$OUT_ROOT/$stem/cp_flow"
  fi
  if [[ -f "$sicle_dir/merged_percell_sicle_masks_int32.npy" ]]; then
    echo "[$stem] skip (done)"
    continue
  fi
  echo "[$stem] $(date +%T) per-cell SICLE (nolin + noblur)..."
  python3 percell_sicle_cellprob_pipeline.py \
    --from-dir "$cp_dir" \
    -o "$sicle_dir" \
    --no-saliency-linearize \
    --sicle-conn-opt gradvmaxmul \
    --sicle-crit-opt minsc \
    --sicle-alpha 2.0 \
    --sicle-nf 2 \
    --sicle-n0 200 \
    --sicle-irreg 0 \
    --sicle-adhr 1 \
    --sicle-max-iters 7 \
    --saliency-threshold 0.3 \
    --saliency-blur-sigma 0 \
    --margin 4 \
    --min-cell-area 128 \
    --disable-and-merge \
    --and-unless-round \
    --min-fg-circularity 0.70 \
    --min-fg-solidity 0.85 \
    --fill-holes \
    --keep-largest-cc \
    --closing-radius 1 \
    --image "$png"
done

echo "Done: $OUT_ROOT"
