#!/usr/bin/env bash
# Small sweep over SICLE hyperparameters. Each config:
#   1) runs run_sibgrapi2026_pipeline.sh into out_sibgrapi2026_sweep/<tag>/
#   2) runs extract_slices_lab_gt.py
#   3) runs evaluate_sibgrapi2026.py
#   4) appends a row to out_sibgrapi2026_sweep/sweep_summary.csv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SWEEP_ROOT="${SWEEP_ROOT:-$HERE/out_sibgrapi2026_sweep}"
SUMMARY="$SWEEP_ROOT/sweep_summary.csv"
mkdir -p "$SWEEP_ROOT"

if [[ ! -f "$SUMMARY" ]]; then
  echo "tag,conn,crit,alpha,nf,n0,irreg,adhr,sal_thr,and_unless_round,closing_r,dense_dice,dense_aji,dense_pq,dense_sq,dense_rq,dense_f1_50,dense_map_dsb,dense_pct_sicle_wins,dense_pct_cellpose_wins,dense_pct_ties" > "$SUMMARY"
fi

run_config() {
  local tag="$1"
  shift
  local out="$SWEEP_ROOT/$tag"
  echo
  echo "############################################################"
  echo "# CONFIG: $tag"
  echo "# env: $*"
  echo "############################################################"

  if [[ -f "$out/eval_nuclick.csv" ]]; then
    echo "  skip pipeline ($out/eval_nuclick.csv already exists)"
  else
    env "$@" OUT_ROOT="$out" bash run_sibgrapi2026_pipeline.sh >"$out.run.log" 2>&1 || {
      echo "  RUN FAILED. log: $out.run.log"
      return 1
    }
  fi
  # Always (re)build GT + eval logs so we can parse below.
  python3 extract_slices_lab_gt.py --out-root "$out" >"$out.gt.log" 2>&1
  python3 evaluate_sibgrapi2026.py --out-root "$out" >"$out.eval.log" 2>&1
  python3 percell_compare_sicle_cellpose.py --out-root "$out" >"$out.percell.log" 2>&1

  # extract dense-GT row for SICLE from eval log; pull pct from percell ALL row
  local eval_log="$out.eval.log"
  local percell_log="$out.percell.log"
  local dense_line
  dense_line=$(grep -A 99 "Dense-GT subset" "$eval_log" | grep -E "^\s+sicle\b" | head -1)
  local dice aji pq sq rq f1 mapd
  dice=$(echo "$dense_line"  | grep -oP "Dice=\K[0-9.]+")
  aji=$(echo  "$dense_line"  | grep -oP "AJI=\K[0-9.]+")
  pq=$(echo   "$dense_line"  | grep -oP "PQ=\K[0-9.]+")
  sq=$(echo   "$dense_line"  | grep -oP "SQ=\K[0-9.]+")
  rq=$(echo   "$dense_line"  | grep -oP "RQ=\K[0-9.]+")
  f1=$(echo   "$dense_line"  | grep -oP "F1@\.5=\K[0-9.]+")
  mapd=$(echo "$dense_line"  | grep -oP "mAP_DSB=\K[0-9.]+")

  local pct_sicle pct_cp pct_ties
  local all_line
  all_line=$(grep -E "^ALL " "$percell_log" | tail -1)
  pct_sicle=$(echo "$all_line" | awk '{for(i=1;i<=NF;i++)if($i~/%/){print $i; break}}' | tr -d '%')
  pct_cp=$(   echo "$all_line" | awk '{c=0; for(i=1;i<=NF;i++)if($i~/%/){c++; if(c==2){print $i; exit}}}' | tr -d '%')
  pct_ties=$( echo "$all_line" | awk '{c=0; for(i=1;i<=NF;i++)if($i~/%/){c++; if(c==3){print $i; exit}}}' | tr -d '%')

  local conn="${SICLE_CONN:-}" crit="${SICLE_CRIT:-}" alpha="${SICLE_ALPHA:-}"
  local nf="${SICLE_NF:-}" n0="${SICLE_N0:-}" irreg="${SICLE_IRREG:-}" adhr="${SICLE_ADHR:-}"
  local sthr="${SAL_THR:-}" aur="${AND_UNLESS_ROUND:-}" cr="${CLOSING_RADIUS:-}"
  for kv in "$@"; do
    case "$kv" in
      SICLE_CONN=*)         conn="${kv#*=}";;
      SICLE_CRIT=*)         crit="${kv#*=}";;
      SICLE_ALPHA=*)        alpha="${kv#*=}";;
      SICLE_NF=*)           nf="${kv#*=}";;
      SICLE_N0=*)           n0="${kv#*=}";;
      SICLE_IRREG=*)        irreg="${kv#*=}";;
      SICLE_ADHR=*)         adhr="${kv#*=}";;
      SAL_THR=*)            sthr="${kv#*=}";;
      AND_UNLESS_ROUND=*)   aur="${kv#*=}";;
      CLOSING_RADIUS=*)     cr="${kv#*=}";;
    esac
  done

  echo "$tag,$conn,$crit,$alpha,$nf,$n0,$irreg,$adhr,$sthr,$aur,$cr,$dice,$aji,$pq,$sq,$rq,$f1,$mapd,$pct_sicle,$pct_cp,$pct_ties" >> "$SUMMARY"
  echo "  -> $tag  Dice=$dice  AJI=$aji  PQ=$pq  SQ=$sq  RQ=$rq  mAP=$mapd  %SICLE=$pct_sicle%  %CP=$pct_cp%"
}

# baseline (já existe em out_sibgrapi2026_clean) — copia eval para sumarizar sem rerodar
if [[ ! -f "$SWEEP_ROOT/00_baseline_clean/eval_nuclick.csv" && -f "$HERE/out_sibgrapi2026_clean/eval_nuclick.csv" ]]; then
  mkdir -p "$SWEEP_ROOT/00_baseline_clean"
  cp -r "$HERE/out_sibgrapi2026_clean/." "$SWEEP_ROOT/00_baseline_clean/"
fi

# 0. baseline: minsc + gradvmaxmul α=2, closing(r=1), fill+CC, AUR on, sal_thr=0.3
run_config 00_baseline_clean \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=minsc SICLE_ALPHA=2.0 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 1. swap to MAXSC (mantém resto igual)
run_config 01_maxsc \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 2. maxsc + fsum (preset "compact" canônico)
run_config 02_maxsc_fsum \
  SICLE_CONN=fsum SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 3. maxsc sem and-unless-round (deixa o SICLE puro)
run_config 03_maxsc_no_aur \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
  SAL_THR=0.3 AND_UNLESS_ROUND=0 CLOSING_RADIUS=1

# 4. minsc + sal_thr=0.0 (preserva saliência intermediária para gradiente)
run_config 04_minsc_salthr0 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=minsc SICLE_ALPHA=2.0 \
  SAL_THR=0.0 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 5. minsc + nf=3 (menos agressivo na fusão final)
run_config 05_minsc_nf3 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=minsc SICLE_ALPHA=2.0 SICLE_NF=3 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 6. minsc + adhr=24 (boost de adherence na fronteira; afeta fsum, mas testar)
run_config 06_minsc_adhr24 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=minsc SICLE_ALPHA=2.0 SICLE_ADHR=24 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 7. maxsc + closing(r=2) (boundary smoothing mais agressivo)
run_config 07_maxsc_closing2 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=2

# ===== Sweeps focados em --irreg e --adhr (gradvmaxmul + maxsc) =====
# Base = best so far = 01_maxsc (closing_r=1, AUR=1, sal_thr=0.3)

# 8-12: sweep --sicle-irreg (default 0.12)
run_config 08_maxsc_irreg004 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_IRREG=0.04 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 09_maxsc_irreg008 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_IRREG=0.08 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 10_maxsc_irreg020 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_IRREG=0.20 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 11_maxsc_irreg030 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_IRREG=0.30 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 12_maxsc_irreg050 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_IRREG=0.50 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

# 13-16: sweep --sicle-adhr (default 12)
run_config 13_maxsc_adhr4 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_ADHR=4 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 14_maxsc_adhr8 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_ADHR=8 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 15_maxsc_adhr24 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_ADHR=24 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1
run_config 16_maxsc_adhr36 \
  SICLE_CONN=gradvmaxmul SICLE_CRIT=maxsc SICLE_ALPHA=2.0 SICLE_ADHR=36 \
  SAL_THR=0.3 AND_UNLESS_ROUND=1 CLOSING_RADIUS=1

echo
echo "=========================================================================="
echo "Sweep summary: $SUMMARY"
echo "=========================================================================="
column -s, -t < "$SUMMARY"
