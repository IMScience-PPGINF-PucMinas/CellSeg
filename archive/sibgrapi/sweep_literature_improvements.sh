#!/usr/bin/env bash
# Sweep based on the SICLE literature (Belém et al., JMIV 2023; UOIFT-SICLE 2024).
#
# Tests three orthogonal improvements against the gradvmaxmul+clean baseline:
#   1. Canonical FMAX + w_root^alpha (Eq. 2 of JMIV 2023): the SOTA option when
#      saliency is accurate. Bug-fixed (objsm now passed for fmax/fsum too).
#   2. Seed penalizations: pobj/pbord/posb/pbobs (sec. 4.2 of JMIV 2023).
#   3. Veta-style solidity post-filter (UOIFT-SICLE 2024, sec. 3.4): reject
#      SICLE outputs with solidity < 0.80 (revert to Cellpose mask).
#   4. RGB image + cellprob saliency separately (canonical input).
#
# Each cfg writes:
#   <OUT_ROOT>/<cfg>/<stem>/sicle/...
#   <OUT_ROOT>/<cfg>/eval_metrics.log
#   <OUT_ROOT>/<cfg>/percell_compare.log
#   <OUT_ROOT>/<cfg>/br_analysis.log
#
# Usage:
#   bash sweep_literature_improvements.sh
#   OUT_ROOT=./out_sweep_lit bash sweep_literature_improvements.sh
#
# Stop early with Ctrl-C; existing cfgs are not redone (idempotent).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_ROOT="${OUT_ROOT:-$HERE/out_sweep_lit}"
DATA_DIR="${DATA_DIR:-$HERE/data_sibgrapi2026/data_sibgrapi2026}"
USE_GPU="${USE_GPU:-1}"
SHARE_CP_DIR="${SHARE_CP_DIR:-$HERE/out_sibgrapi2026}"  # reuse existing cp_flow

mkdir -p "$OUT_ROOT"

# Configurations to try.  Format: name|conn|crit|alpha|pen|use_rgb|min_sol
# Common: nf=2, n0=200, sal_thr=0.3, fill+keep_cc, closing=1
declare -a CFGS=(
  # --- baseline reference (already exists in out_sibgrapi2026_clean) ---
  # gradvmaxmul + clean; just for log-grouping. Skipped automatically if found.
  "00_gradvmaxmul_clean|gradvmaxmul|minsc|2.0|none|0|0.0"

  # --- 1. canonical fmax + w_root^alpha (Eq.2 JMIV 2023) ---
  "01_fmax_alpha0_5|fmax|minsc|0.5|none|0|0.0"
  "02_fmax_alpha1_0|fmax|minsc|1.0|none|0|0.0"
  "03_fmax_alpha2_0|fmax|minsc|2.0|none|0|0.0"

  # --- 2. fmax + seed penalizations (sec 4.2 JMIV 2023) ---
  "04_fmax_a1_pobj|fmax|minsc|1.0|obj|0|0.0"
  "05_fmax_a1_pbord|fmax|minsc|1.0|bord|0|0.0"
  "06_fmax_a1_pbobs|fmax|minsc|1.0|bobs|0|0.0"

  # --- 3. fmax + RGB image + cellprob saliency (canonical input) ---
  "07_fmax_a1_rgb|fmax|minsc|1.0|none|1|0.0"
  "08_fmax_a1_rgb_pobj|fmax|minsc|1.0|obj|1|0.0"

  # --- 4. fmax + Veta-style solidity filter ---
  "09_fmax_a1_sol80|fmax|minsc|1.0|none|0|0.80"
  "10_fmax_a1_pobj_sol80|fmax|minsc|1.0|obj|0|0.80"
)

echo "Literature sweep: ${#CFGS[@]} configs into $OUT_ROOT"

# 1) reuse existing cellpose outputs from SHARE_CP_DIR if present (idempotent)
shopt -s nullglob
PNGS=("$DATA_DIR"/*.png)
echo "  ${#PNGS[@]} PNGs from $DATA_DIR"
echo "  Reusing cp_flow from $SHARE_CP_DIR if available"

for entry in "${CFGS[@]}"; do
  IFS='|' read -r name conn crit alpha pen use_rgb min_sol <<<"$entry"
  cfg_dir="$OUT_ROOT/$name"
  echo
  echo "============================================================"
  echo "[$name] conn=$conn crit=$crit alpha=$alpha pen=$pen use_rgb=$use_rgb min_sol=$min_sol"
  echo "============================================================"
  mkdir -p "$cfg_dir"

  for png in "${PNGS[@]}"; do
    stem="$(basename "$png" .png)"
    case_dir="$cfg_dir/$stem"
    cp_dir="$case_dir/cp_flow"
    sicle_dir="$case_dir/sicle"
    mkdir -p "$case_dir"

    # Reuse cellpose outputs to save time
    if [[ ! -f "$cp_dir/step04_masks_uint16.npy" || ! -f "$cp_dir/step03_dP_cellprob.npz" ]]; then
      shared_cp="$SHARE_CP_DIR/$stem/cp_flow"
      if [[ -f "$shared_cp/step04_masks_uint16.npy" && -f "$shared_cp/step03_dP_cellprob.npz" ]]; then
        mkdir -p "$cp_dir"
        cp -al "$shared_cp"/* "$cp_dir/" 2>/dev/null || cp "$shared_cp"/* "$cp_dir/"
      else
        echo "  [$stem] no shared cp_flow; skipping (run main pipeline first)"
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
  if [[ -z "$line" ]]; then
    printf "%-30s %s\n" "$name" "(no sicle line)"
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
  printf "%-30s %-7s %-7s %-7s %-7s %-7s %-7s\n" \
    "$name" "$dice" "$aji" "$pq" "$f1" "$map" "$br_s"
done

echo
echo "Browse: $OUT_ROOT"
