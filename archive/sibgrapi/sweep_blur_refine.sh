#!/usr/bin/env bash
# Refined sweep around the new winner: gradvmaxmul + saliency blur sigma=1.
# Tests: blur sigma fine-tuning, RGB combo, threshold variations, nf variations.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_ROOT="${OUT_ROOT:-$HERE/out_sweep_blur}"
DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
SHARE_CP_DIR="${SHARE_CP_DIR:-$HERE/out_sibgrapi2026}"

mkdir -p "$OUT_ROOT"

# Format: name|conn|alpha|use_rgb|blur|sal_thr|nf
declare -a CFGS=(
  # Reference (already optimal)
  "00_blur1|gradvmaxmul|2.0|0|1.0|0.3|2"

  # Fine sigma
  "S1_blur0_5|gradvmaxmul|2.0|0|0.5|0.3|2"
  "S2_blur0_75|gradvmaxmul|2.0|0|0.75|0.3|2"
  "S3_blur1_25|gradvmaxmul|2.0|0|1.25|0.3|2"
  "S4_blur1_5|gradvmaxmul|2.0|0|1.5|0.3|2"

  # Combo with RGB image
  "R1_blur1_rgb|gradvmaxmul|2.0|1|1.0|0.3|2"
  "R2_blur1_5_rgb|gradvmaxmul|2.0|1|1.5|0.3|2"

  # Saliency threshold tweak (with blur, lower thr might keep useful soft signal)
  "T1_blur1_thr0_0|gradvmaxmul|2.0|0|1.0|0.0|2"
  "T2_blur1_thr0_15|gradvmaxmul|2.0|0|1.0|0.15|2"
  "T3_blur1_thr0_5|gradvmaxmul|2.0|0|1.0|0.5|2"

  # nf variations
  "N1_blur1_nf1|gradvmaxmul|2.0|0|1.0|0.3|1"
  "N2_blur1_nf3|gradvmaxmul|2.0|0|1.0|0.3|3"

  # fmax with same blur (verify gradvmaxmul still wins)
  "F1_fmax_blur1_thr0_15|fmax|1.0|0|1.0|0.15|2"
  "F2_fmax_blur1_rgb|fmax|1.0|1|1.0|0.3|2"
)

echo "Blur refinement sweep: ${#CFGS[@]} configs into $OUT_ROOT"

shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)

for entry in "${CFGS[@]}"; do
  IFS='|' read -r name conn alpha use_rgb blur sal_thr nf <<<"$entry"
  cfg_dir="$OUT_ROOT/$name"
  echo
  echo "============================================================"
  echo "[$name] conn=$conn alpha=$alpha rgb=$use_rgb blur=$blur thr=$sal_thr nf=$nf"
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
      --sicle-crit-opt minsc \
      --sicle-pen-opt none \
      --sicle-min-solidity 0.0 \
      $( [[ "$use_rgb" == "1" ]] && echo "--sicle-use-rgb-image" ) \
      --sicle-alpha "$alpha" \
      --sicle-nf "$nf" \
      --sicle-n0 200 \
      --sicle-irreg 0.0 \
      --sicle-adhr 1 \
      --sicle-max-iters 7 \
      --saliency-threshold "$sal_thr" \
      --saliency-blur-sigma "$blur" \
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
      echo "    !!! FAILED"
      continue
    }
  done

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
printf "%-30s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
  "config" "Dice" "AJI" "PQ" "F1" "mAP" "BR_S"
for entry in "${CFGS[@]}"; do
  IFS='|' read -r name _ _ _ _ _ _ <<<"$entry"
  log="$OUT_ROOT/$name/eval_metrics.log"
  if [[ ! -f "$log" ]]; then
    printf "%-30s %s\n" "$name" "(no log)"
    continue
  fi
  line="$(grep -E "^  sicle .*\(n=12\)" "$log" | head -1)"
  [[ -z "$line" ]] && { printf "%-30s %s\n" "$name" "(no sicle line)"; continue; }
  dice="$(echo "$line" | grep -oE 'Dice=[0-9.]+' | cut -d= -f2)"
  aji="$(echo "$line" | grep -oE 'AJI=[0-9.]+' | cut -d= -f2)"
  pq="$(echo "$line" | grep -oE 'PQ=[0-9.]+' | head -1 | cut -d= -f2)"
  f1="$(echo "$line" | grep -oE 'F1@.5=[0-9.]+' | cut -d= -f2)"
  map="$(echo "$line" | grep -oE 'mAP_DSB=[0-9.]+' | cut -d= -f2)"
  br_log="$OUT_ROOT/$name/br_analysis.log"
  br_line="$(grep -E "^ALL " "$br_log" 2>/dev/null | head -1)"
  br_s="$(echo "$br_line" | awk '{print $NF}' | cut -d/ -f1 || echo "-")"
  printf "%-30s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
    "$name" "$dice" "$aji" "$pq" "$f1" "$map" "$br_s"
done

echo
echo "Browse: $OUT_ROOT"
