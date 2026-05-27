#!/usr/bin/env bash
# Phase 2 sweep: multiscale SICLE + adaptive N0/Nf by cell area.
#
# Baseline: gradvmaxmul + minsc + blur=0.5 (best from v3).
#
# Usage:
#   bash sweep_literature_v4.sh
#   OUT_ROOT=./out_sweep_lit_v4 bash sweep_literature_v4.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_ROOT="${OUT_ROOT:-$HERE/out_sweep_lit_v4}"
DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
SHARE_CP_DIR="${SHARE_CP_DIR:-$HERE/out_sibgrapi2026}"

mkdir -p "$OUT_ROOT"

# name|adaptive|multiscale|scale_select|scale_min_sol
declare -a CFGS=(
  "00_ref_blur05|0|0|last|0.0"
  "01_adaptive_seeds|1|0|last|0.0"
  "02_multiscale_last|0|1|last|0.0"
  "03_multiscale_veta_sol|0|1|veta_solidity|0.0"
  "04_multiscale_veta_comp|0|1|veta_composite|0.0"
  "05_adapt_ms_veta_sol|1|1|veta_solidity|0.0"
  "06_adapt_ms_veta_comp|1|1|veta_composite|0.0"
  "07_ms_veta_sol_min080|0|1|veta_solidity|0.80"
)

echo "Literature v4 sweep (Phase 2): ${#CFGS[@]} configs into $OUT_ROOT"

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)

run_sicle_cfg() {
  local name="$1" adaptive="$2" multiscale="$3" scale_sel="$4" scale_min_sol="$5"
  local cfg_dir="$OUT_ROOT/$name"
  echo
  echo "============================================================"
  echo "[$name] adaptive=$adaptive multiscale=$multiscale select=$scale_sel min_sol=$scale_min_sol"
  echo "============================================================"
  mkdir -p "$cfg_dir"

  for png in "${PNGS[@]}"; do
    stem="$(basename "$png" .png)"
    case_dir="$cfg_dir/$stem"
    cp_dir="$case_dir/cp_flow"
    sicle_dir="$case_dir/sicle"
    mkdir -p "$case_dir"

    if [[ ! -f "$cp_dir/step04_masks_uint16.npy" || ! -f "$cp_dir/step03_dP_cellprob.npz" ]]; then
      shared_cp="$SHARE_CP_DIR/$stem/cp_flow"
      if [[ -f "$shared_cp/step04_masks_uint16.npy" && -f "$shared_cp/step03_dP_cellprob.npz" ]]; then
        mkdir -p "$cp_dir"
        cp -al "$shared_cp"/* "$cp_dir/" 2>/dev/null || cp "$shared_cp"/* "$cp_dir/"
      else
        echo "  [$stem] no shared cp_flow; skipping"
        continue
      fi
    fi

    if [[ -f "$sicle_dir/merged_percell_sicle_masks_int32.npy" ]]; then
      echo "  [$stem] already done, skip"
      continue
    fi
    mkdir -p "$sicle_dir"

    echo "  [$stem] running SICLE ($name)..."
    python3 percell_sicle_cellprob_pipeline.py \
      --from-dir "$cp_dir" \
      -o "$sicle_dir" \
      --sicle-conn-opt gradvmaxmul \
      --sicle-crit-opt minsc \
      --sicle-alpha 2.0 \
      --sicle-nf 2 \
      --sicle-n0 200 \
      --sicle-irreg 0.0 \
      --sicle-adhr 1 \
      --sicle-max-iters 7 \
      $( [[ "$adaptive" == "1" ]] && echo "--sicle-adaptive-seeds" ) \
      $( [[ "$multiscale" == "1" ]] && echo "--sicle-multiscale" ) \
      --sicle-scale-select "$scale_sel" \
      --sicle-scale-min-solidity "$scale_min_sol" \
      --saliency-threshold 0.3 \
      --saliency-blur-sigma 0.5 \
      --margin 4 \
      --min-cell-area 128 \
      --disable-and-merge \
      --and-unless-round \
      --min-fg-circularity 0.70 \
      --min-fg-solidity 0.85 \
      --fill-holes \
      --keep-largest-cc \
      --closing-radius 1 \
      --image "$png" \
      --overlay-border-source both \
      --overlay-border-color 0,255,0 \
      --overlay-cellpose-border-color 255,255,0 \
      > "$sicle_dir/run.log" 2>&1 || {
      echo "    !!! FAILED (see $sicle_dir/run.log)"
      continue
    }
  done

  echo "  [$name] running GT extraction + evaluations..."
  python3 extract_slices_lab_gt.py --out-root "$cfg_dir" \
    > "$cfg_dir/extract_gt.log" 2>&1 || true
  python3 evaluate_sibgrapi2026.py --out-root "$cfg_dir" \
    > "$cfg_dir/eval_metrics.log" 2>&1 || true
  python3 percell_compare_sicle_cellpose.py --out-root "$cfg_dir" \
    > "$cfg_dir/percell_compare.log" 2>&1 || true
  python3 percell_boundary_recall.py --out-root "$cfg_dir" \
    > "$cfg_dir/br_analysis.log" 2>&1 || true
  echo "  [$name] done."
}

for entry in "${CFGS[@]}"; do
  IFS='|' read -r name adaptive multiscale scale_sel scale_min_sol <<<"$entry"
  run_sicle_cfg "$name" "$adaptive" "$multiscale" "$scale_sel" "$scale_min_sol"
done

echo
echo "============================================================"
echo "Summary across configs"
echo "============================================================"
printf "%-35s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
  "config" "Dice" "AJI" "PQ" "F1" "mAP" "BR_S"
for entry in "${CFGS[@]}"; do
  IFS='|' read -r name _ _ _ _ <<<"$entry"
  log="$OUT_ROOT/$name/eval_metrics.log"
  if [[ ! -f "$log" ]]; then
    printf "%-35s %s\n" "$name" "(no log)"
    continue
  fi
  line="$(grep -E "^  sicle .*\(n=12\)" "$log" | head -1)"
  if [[ -z "$line" ]]; then
    printf "%-35s %s\n" "$name" "(no sicle line)"
    continue
  fi
  dice="$(echo "$line" | grep -oE 'Dice=[0-9.]+' | cut -d= -f2)"
  aji="$(echo "$line" | grep -oE 'AJI=[0-9.]+' | cut -d= -f2)"
  pq="$(echo "$line" | grep -oE 'PQ=[0-9.]+' | head -1 | cut -d= -f2)"
  f1="$(echo "$line" | grep -oE 'F1@.5=[0-9.]+' | cut -d= -f2)"
  map="$(echo "$line" | grep -oE 'mAP_DSB=[0-9.]+' | cut -d= -f2)"
  br_log="$OUT_ROOT/$name/br_analysis.log"
  br_line="$(grep -E "^ALL " "$br_log" 2>/dev/null | head -1)"
  br_s="$(echo "$br_line" | awk '{print $NF}' | cut -d/ -f1 || echo "-")"
  printf "%-35s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
    "$name" "$dice" "$aji" "$pq" "$f1" "$map" "$br_s"
done

echo
echo "Browse: $OUT_ROOT"
