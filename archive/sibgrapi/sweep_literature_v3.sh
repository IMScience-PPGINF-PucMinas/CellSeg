#!/usr/bin/env bash
# Phase 1 literature improvements sweep:
#   - maxsc + gradvmaxmul + saliency blur (σ=0.4/0.5/0.6)
#   - composite saliency: cellprob + |∇L| (grad_l_mix)
#   - flow-weighted saliency: cellprob * (1 + γ|dP|)
#   - reference: gradvmaxmul + minsc + blur=0.5 (current best)
#
# Usage:
#   bash sweep_literature_v3.sh
#   OUT_ROOT=./out_sweep_lit_v3 bash sweep_literature_v3.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_ROOT="${OUT_ROOT:-$HERE/out_sweep_lit_v3}"
DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
SHARE_CP_DIR="${SHARE_CP_DIR:-$HERE/out_sibgrapi2026}"

mkdir -p "$OUT_ROOT"

# name|conn|crit|alpha|pen|use_rgb|min_sol|sal_blur|sal_mode|sal_mix_w|sal_flow_g
declare -a CFGS=(
  "00_ref_blur05_minsc|gradvmaxmul|minsc|2.0|none|0|0.0|0.5|cellprob|0.35|0.5"

  "01_maxsc_blur0_4|gradvmaxmul|maxsc|2.0|none|0|0.0|0.4|cellprob|0.35|0.5"
  "02_maxsc_blur0_5|gradvmaxmul|maxsc|2.0|none|0|0.0|0.5|cellprob|0.35|0.5"
  "03_maxsc_blur0_6|gradvmaxmul|maxsc|2.0|none|0|0.0|0.6|cellprob|0.35|0.5"

  "04_grad_l_mix_w035|gradvmaxmul|minsc|2.0|none|0|0.0|0.5|grad_l_mix|0.35|0.5"
  "05_grad_l_mix_w050|gradvmaxmul|minsc|2.0|none|0|0.0|0.5|grad_l_mix|0.50|0.5"

  "06_flow_mul_g050|gradvmaxmul|minsc|2.0|none|0|0.0|0.5|flow_mul|0.35|0.50"
  "07_flow_mul_g030|gradvmaxmul|minsc|2.0|none|0|0.0|0.5|flow_mul|0.35|0.30"
)

echo "Literature v3 sweep: ${#CFGS[@]} configs into $OUT_ROOT"

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)

for entry in "${CFGS[@]}"; do
  IFS='|' read -r name conn crit alpha pen use_rgb min_sol sal_blur sal_mode sal_mix_w sal_flow_g <<<"$entry"
  cfg_dir="$OUT_ROOT/$name"
  echo
  echo "============================================================"
  echo "[$name] conn=$conn crit=$crit alpha=$alpha blur=$sal_blur mode=$sal_mode mix=$sal_mix_w flow_g=$sal_flow_g"
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
      --sicle-conn-opt "$conn" \
      --sicle-crit-opt "$crit" \
      --sicle-pen-opt "$pen" \
      --sicle-min-solidity "$min_sol" \
      $( [[ "$use_rgb" == "1" ]] && echo "--sicle-use-rgb-image" ) \
      --sicle-alpha "$alpha" \
      --sicle-nf 2 \
      --sicle-n0 200 \
      --sicle-irreg 0.0 \
      --sicle-adhr 1 \
      --sicle-max-iters 7 \
      --saliency-threshold 0.3 \
      --saliency-blur-sigma "$sal_blur" \
      --saliency-mode "$sal_mode" \
      --saliency-mix-weight "$sal_mix_w" \
      --saliency-flow-gamma "$sal_flow_g" \
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
done

echo
echo "============================================================"
echo "Summary across configs"
echo "============================================================"
printf "%-35s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
  "config" "Dice" "AJI" "PQ" "F1" "mAP" "BR_S"
for entry in "${CFGS[@]}"; do
  IFS='|' read -r name _ _ _ _ _ _ _ _ _ _ <<<"$entry"
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
