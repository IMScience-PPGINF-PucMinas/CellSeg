#!/usr/bin/env bash
set -euo pipefail

cd ~/doutorado/new_pipeline
export PYTHONPATH=../cellpose

IMG="/home/lacerda/doutorado/GR07-1.svs_slice1.tiff"
BASE="./cp_flow_out/step04_masks_uint16.npy"
mkdir -p path_cost_visuals

while read -r TAG CONN CRIT; do
  # conservative merge: SICLE_fg AND Cellpose-cell
  for MODE in and no_and; do
    OUTDIR="./path_cost_visuals/${TAG}/${MODE}"
    EXTRA_ARGS=()
    if [[ "$MODE" == "no_and" ]]; then
      EXTRA_ARGS+=(--disable-and-merge)
    fi

    python percell_sicle_cellprob_pipeline.py \
      --from-dir ./cp_flow_out \
      -o "$OUTDIR" \
      --sicle-conn-opt "$CONN" \
      --sicle-crit-opt "$CRIT" \
      --image "$IMG" \
      "${EXTRA_ARGS[@]}"

    python mask_outline_overlay.py \
      --image "$IMG" \
      --masks "$OUTDIR/merged_percell_sicle_masks_int32.npy" \
      -o "$OUTDIR/outline_overlay.png"

    python compare_segmentation_masks_diff.py \
      --mask-a "$BASE" \
      --mask-b "$OUTDIR/merged_percell_sicle_masks_int32.npy" \
      -o "$OUTDIR/compare_vs_step04" \
      --also-save-diff-only-rgb

    echo "done $TAG ($CONN,$CRIT) mode=$MODE"
  done
done << 'PAIRS'
irregular_fmax_minsc fmax minsc
compact_fsum_maxsc fsum maxsc
cross_fmax_maxsc fmax maxsc
cross_fsum_minsc fsum minsc
PAIRS
