#!/usr/bin/env bash
# Ultra-fine sigma sweep around the new optimum (sigma=0.5).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
OUT_ROOT="${OUT_ROOT:-$HERE/out_sweep_blur_fine}"
DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
SHARE_CP_DIR="${SHARE_CP_DIR:-$HERE/out_sibgrapi2026}"
mkdir -p "$OUT_ROOT"

declare -a SIGMAS=(0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)

for sig in "${SIGMAS[@]}"; do
  name="sig_${sig//./_}"
  cfg_dir="$OUT_ROOT/$name"
  echo
  echo "=== $name (sigma=$sig) ==="
  mkdir -p "$cfg_dir"
  for png in "${PNGS[@]}"; do
    stem="$(basename "$png" .png)"
    case_dir="$cfg_dir/$stem"; cp_dir="$case_dir/cp_flow"; sicle_dir="$case_dir/sicle"
    mkdir -p "$case_dir"
    if [[ ! -f "$cp_dir/step04_masks_uint16.npy" ]]; then
      shared_cp="$SHARE_CP_DIR/$stem/cp_flow"
      if [[ -f "$shared_cp/step04_masks_uint16.npy" ]]; then
        mkdir -p "$cp_dir"
        cp -al "$shared_cp"/* "$cp_dir/" 2>/dev/null || cp "$shared_cp"/* "$cp_dir/"
      else
        continue
      fi
    fi
    [[ -f "$sicle_dir/merged_percell_sicle_masks_int32.npy" ]] && continue
    mkdir -p "$sicle_dir"
    python3 percell_sicle_cellprob_pipeline.py \
      --from-dir "$cp_dir" -o "$sicle_dir" \
      --sicle-conn-opt gradvmaxmul --sicle-crit-opt minsc \
      --sicle-pen-opt none --sicle-min-solidity 0.0 \
      --sicle-alpha 2.0 --sicle-nf 2 --sicle-n0 200 \
      --sicle-irreg 0.0 --sicle-adhr 1 --sicle-max-iters 7 \
      --saliency-threshold 0.3 --saliency-blur-sigma "$sig" \
      --margin 4 --min-cell-area 128 \
      --disable-and-merge --and-unless-round \
      --min-fg-circularity 0.70 --min-fg-solidity 0.85 \
      --fill-holes --keep-largest-cc --closing-radius 1 \
      --image "$png" \
      --overlay-border-source both --overlay-border-color 0,255,0 \
      --overlay-cellpose-border-color 255,255,0 \
      > "$sicle_dir/run.log" 2>&1 || echo "  $stem FAILED"
  done
  python3 extract_slices_lab_gt.py --out-root "$cfg_dir" > "$cfg_dir/extract_gt.log" 2>&1 || true
  python3 evaluate_sibgrapi2026.py --out-root "$cfg_dir" > "$cfg_dir/eval_metrics.log" 2>&1 || true
  python3 percell_boundary_recall.py --out-root "$cfg_dir" > "$cfg_dir/br_analysis.log" 2>&1 || true
  echo "  done $name"
done

echo
echo "============================================================"
echo "Sigma sweep summary"
echo "============================================================"
printf "%-12s %-7s %-7s %-7s %-7s %-7s %-7s\n" "sigma" "Dice" "AJI" "PQ" "F1" "mAP" "BR_S"
for sig in "${SIGMAS[@]}"; do
  name="sig_${sig//./_}"
  log="$OUT_ROOT/$name/eval_metrics.log"
  [[ ! -f "$log" ]] && { printf "%-12s no-log\n" "$sig"; continue; }
  line="$(grep -E "^  sicle .*\(n=12\)" "$log" | head -1)"
  [[ -z "$line" ]] && { printf "%-12s no-line\n" "$sig"; continue; }
  dice="$(echo "$line" | grep -oE 'Dice=[0-9.]+' | cut -d= -f2)"
  aji="$(echo "$line" | grep -oE 'AJI=[0-9.]+' | cut -d= -f2)"
  pq="$(echo "$line" | grep -oE 'PQ=[0-9.]+' | head -1 | cut -d= -f2)"
  f1="$(echo "$line" | grep -oE 'F1@.5=[0-9.]+' | cut -d= -f2)"
  map="$(echo "$line" | grep -oE 'mAP_DSB=[0-9.]+' | cut -d= -f2)"
  br_s="$(grep -E "^ALL " "$OUT_ROOT/$name/br_analysis.log" 2>/dev/null | head -1 | awk '{print $NF}' | cut -d/ -f1)"
  printf "%-12s %-7s %-7s %-7s %-7s %-7s %-7s\n" "$sig" "$dice" "$aji" "$pq" "$f1" "$map" "$br_s"
done
