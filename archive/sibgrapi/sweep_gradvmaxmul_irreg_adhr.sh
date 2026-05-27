#!/usr/bin/env bash
# Sweep --irreg and --adhr for gradvmaxmul AFTER the iftSICLE.c modification
# (these knobs now take effect for gradvmaxmul/gradvmax, not only fsum).
#
# Base config: best so far = 01_maxsc (gradvmaxmul + maxsc + α=2.0 + AUR + closing=1).
# All configs use the new C build, so 00 with irreg=0, adhr=1 reproduces the old gradvmaxmul.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SWEEP_ROOT="${SWEEP_ROOT:-$HERE/out_sibgrapi2026_sweep_v2}"
SUMMARY="$SWEEP_ROOT/sweep_summary.csv"
mkdir -p "$SWEEP_ROOT"

if [[ ! -f "$SUMMARY" ]]; then
  echo "tag,conn,crit,alpha,irreg,adhr,closing_r,dense_dice,dense_aji,dense_pq,dense_sq,dense_rq,dense_f1_50,dense_map_dsb,pct_sicle_wins,pct_cellpose_wins,pct_ties" > "$SUMMARY"
fi

run_config() {
  local tag="$1"; shift
  local out="$SWEEP_ROOT/$tag"
  echo
  echo "############################################################"
  echo "# CONFIG: $tag"
  echo "# env: $*"
  echo "############################################################"

  if [[ ! -f "$out/eval_nuclick.csv" ]]; then
    env "$@" OUT_ROOT="$out" bash run_sibgrapi2026_pipeline.sh >"$out.run.log" 2>&1 || {
      echo "  RUN FAILED. log: $out.run.log"
      return 1
    }
  else
    echo "  skip pipeline ($out/eval_nuclick.csv exists)"
  fi
  python3 extract_slices_lab_gt.py --out-root "$out" >"$out.gt.log" 2>&1
  python3 evaluate_sibgrapi2026.py --out-root "$out" >"$out.eval.log" 2>&1
  python3 percell_compare_sicle_cellpose.py --out-root "$out" >"$out.percell.log" 2>&1

  local dense_line; dense_line=$(grep -A 99 "Dense-GT subset" "$out.eval.log" | grep -E "^\s+sicle\b" | head -1)
  local dice aji pq sq rq f1 mapd
  dice=$(echo "$dense_line" | grep -oP "Dice=\K[0-9.]+")
  aji=$(echo  "$dense_line" | grep -oP "AJI=\K[0-9.]+")
  pq=$(echo   "$dense_line" | grep -oP "PQ=\K[0-9.]+")
  sq=$(echo   "$dense_line" | grep -oP "SQ=\K[0-9.]+")
  rq=$(echo   "$dense_line" | grep -oP "RQ=\K[0-9.]+")
  f1=$(echo   "$dense_line" | grep -oP "F1@\.5=\K[0-9.]+")
  mapd=$(echo "$dense_line" | grep -oP "mAP_DSB=\K[0-9.]+")

  local all_line; all_line=$(grep -E "^ALL " "$out.percell.log" | tail -1)
  local pct_s pct_c pct_t
  pct_s=$(echo "$all_line" | awk '{c=0; for(i=1;i<=NF;i++)if($i~/%/){c++; if(c==1){print $i; exit}}}' | tr -d '%')
  pct_c=$(echo "$all_line" | awk '{c=0; for(i=1;i<=NF;i++)if($i~/%/){c++; if(c==2){print $i; exit}}}' | tr -d '%')
  pct_t=$(echo "$all_line" | awk '{c=0; for(i=1;i<=NF;i++)if($i~/%/){c++; if(c==3){print $i; exit}}}' | tr -d '%')

  local conn="${SICLE_CONN:-}" crit="${SICLE_CRIT:-}" alpha="${SICLE_ALPHA:-}"
  local irreg="${SICLE_IRREG:-}" adhr="${SICLE_ADHR:-}" cr="${CLOSING_RADIUS:-}"
  for kv in "$@"; do
    case "$kv" in
      SICLE_CONN=*) conn="${kv#*=}";;
      SICLE_CRIT=*) crit="${kv#*=}";;
      SICLE_ALPHA=*) alpha="${kv#*=}";;
      SICLE_IRREG=*) irreg="${kv#*=}";;
      SICLE_ADHR=*) adhr="${kv#*=}";;
      CLOSING_RADIUS=*) cr="${kv#*=}";;
    esac
  done

  echo "$tag,$conn,$crit,$alpha,$irreg,$adhr,$cr,$dice,$aji,$pq,$sq,$rq,$f1,$mapd,$pct_s,$pct_c,$pct_t" >> "$SUMMARY"
  echo "  -> $tag  Dice=$dice  AJI=$aji  PQ=$pq  SQ=$sq  RQ=$rq  mAP=$mapd  %SICLE=$pct_s%  %CP=$pct_c%"
}

# Base = old 01_maxsc behavior (irreg=0, adhr=1). Should match Dice≈0.8879.
run_config 00_base_neutral \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
  SICLE_IRREG=0.0 SICLE_ADHR=1 CLOSING_RADIUS=1 AND_UNLESS_ROUND=1

# Sweep IRREG only (adhr=1, no sharpening)
for I in 0.04 0.08 0.16 0.24 0.50 1.00; do
  tag="01_irreg_${I//./_}"
  run_config "$tag" \
    SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
    SICLE_IRREG=$I SICLE_ADHR=1 CLOSING_RADIUS=1 AND_UNLESS_ROUND=1
done

# Sweep ADHR only (irreg=0)
for A in 2 3 4 6 8; do
  tag="02_adhr_${A}"
  run_config "$tag" \
    SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
    SICLE_IRREG=0.0 SICLE_ADHR=$A CLOSING_RADIUS=1 AND_UNLESS_ROUND=1
done

# Combos around the most promising irreg + light adhr
for I in 0.08 0.16; do
  for A in 2 3; do
    tag="03_combo_irreg_${I//./_}_adhr_${A}"
    run_config "$tag" \
      SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
      SICLE_IRREG=$I SICLE_ADHR=$A CLOSING_RADIUS=1 AND_UNLESS_ROUND=1
  done
done

echo
echo "=========================================================================="
echo "Sweep summary: $SUMMARY"
echo "=========================================================================="
column -s, -t < "$SUMMARY"
