#!/usr/bin/env bash
#
# Três alternativas de merge pós-SICLE per-cell (replicável para outras imagens):
#
#   00_raw_no_and     — Uma corrida com --disable-and-merge (base para A e C).
#   01_alt_a_clip     — Alternativa A: recorta o merge com dilatação da máscara Cellpose.
#   02_alt_b_and      — Alternativa B: merge conservador (AND com Cellpose no paste).
#   03_alt_c_component— Alternativa C: por célula, só a componente 8-conexa com maior
#                       sobreposição com a instância Cellpose.
#   04_alt_d_and_unless_round — Opcional (RUN_ROUNDNESS_MERGE=1): raw SICLE se o FG for
#                       suficientemente “redondo”; senão AND com Cellpose no bbox.
#
# Uso (a partir de new_pipeline/)::
#
#   ./run_percell_three_alternatives.sh
#
# Incluir variante “redondo ou AND” (mais uma corrida SICLE)::
#
#   RUN_ROUNDNESS_MERGE=1 MIN_FG_CIRCULARITY=0.72 MIN_FG_SOLIDITY=0.85 ./run_percell_three_alternatives.sh
#
# Outra imagem / pasta Cellpose::
#
#   IMAGE=/path/to/slice.tif FROM_DIR=./cp_flow_out_slice2 OUT_ROOT=./out_slice2 \\
#     SICLE_CONN=gradvmax SICLE_CRIT=minsc SICLE_ALPHA=4 CLIP_DILATE=3 \\
#     ./run_percell_three_alternatives.sh
#
# Requer: Python deps do pipeline, ``compare_segmentation_masks_diff.py``,
# ``merge_postprocess.py``, ``RunSICLE`` (ex.: export SICLE_BIN=...).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../cellpose"

# --- por defeito: ajustar por imagem / experiência ---
: "${IMAGE:=/home/lacerda/doutorado/GR07-1.svs_slice1.tiff}"
: "${FROM_DIR:=./cp_flow_out}"
_IMG_BASE="$(basename "$IMAGE")"
_IMG_SLUG="${_IMG_BASE%.*}"
: "${OUT_ROOT:=./percell_three_alts_${_IMG_SLUG}}"
: "${SICLE_CONN:=gradvmax}"
: "${SICLE_CRIT:=maxsc}"
: "${SICLE_ALPHA:=2.0}"
: "${CLIP_DILATE:=2}"
: "${GEN_AREA_CSV:=1}"
: "${RUN_ROUNDNESS_MERGE:=0}"
: "${MIN_FG_CIRCULARITY:=0.70}"
: "${MIN_FG_SOLIDITY:=0.88}"
: "${OVERLAY_BORDER_SOURCE:=both}"
: "${OVERLAY_CELLPOSE_BORDER_COLOR:=255,0,0}"
: "${WRITE_COMPARE_VS_STEP04:=1}"

BASE_MASK="${FROM_DIR%/}/step04_masks_uint16.npy"
if [[ ! -f "$BASE_MASK" ]]; then
  echo "Erro: não existe $BASE_MASK (defina FROM_DIR corretamente)" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Erro: não existe IMAGE=$IMAGE" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
META="$OUT_ROOT/run_config.txt"
{
  echo "timestamp=$(date -Iseconds)"
  echo "IMAGE=$IMAGE"
  echo "FROM_DIR=$FROM_DIR"
  echo "OUT_ROOT=$OUT_ROOT"
  echo "SICLE_CONN=$SICLE_CONN SICLE_CRIT=$SICLE_CRIT SICLE_ALPHA=$SICLE_ALPHA"
  echo "CLIP_DILATE=$CLIP_DILATE"
  echo "SICLE_BIN=${SICLE_BIN:-}"
  echo "RUN_ROUNDNESS_MERGE=${RUN_ROUNDNESS_MERGE}"
  echo "MIN_FG_CIRCULARITY=${MIN_FG_CIRCULARITY} MIN_FG_SOLIDITY=${MIN_FG_SOLIDITY}"
  echo "OVERLAY_BORDER_SOURCE=${OVERLAY_BORDER_SOURCE} OVERLAY_CELLPOSE_BORDER_COLOR=${OVERLAY_CELLPOSE_BORDER_COLOR}"
  echo "WRITE_COMPARE_VS_STEP04=${WRITE_COMPARE_VS_STEP04}"
} | tee "$META"

RAW="${OUT_ROOT}/00_raw_no_and"
ALT_A="${OUT_ROOT}/01_alt_a_clip_d${CLIP_DILATE}"
ALT_B="${OUT_ROOT}/02_alt_b_cellpose_and"
ALT_C="${OUT_ROOT}/03_alt_c_largest_component"

COMPARE_FLAGS=()
if [[ "${WRITE_COMPARE_VS_STEP04}" == "1" ]]; then
  COMPARE_FLAGS+=(--write-compare-vs-step04)
fi

echo "=== [00] Per-cell SICLE, sem AND (base) ==="
mkdir -p "$RAW"
python percell_sicle_cellprob_pipeline.py \
  --from-dir "$FROM_DIR" \
  -o "$RAW" \
  --sicle-conn-opt "$SICLE_CONN" \
  --sicle-crit-opt "$SICLE_CRIT" \
  --sicle-alpha "$SICLE_ALPHA" \
  --image "$IMAGE" \
  --overlay-border-source "$OVERLAY_BORDER_SOURCE" \
  --overlay-cellpose-border-color "$OVERLAY_CELLPOSE_BORDER_COLOR" \
  "${COMPARE_FLAGS[@]}" \
  --disable-and-merge

MERGED_RAW="${RAW}/merged_percell_sicle_masks_int32.npy"

echo "=== [01] Alternativa A — clip + dilatação Cellpose ==="
mkdir -p "$ALT_A"
python merge_postprocess.py clip \
  --cellpose "$BASE_MASK" \
  --merged "$MERGED_RAW" \
  --out "${ALT_A}/merged_percell_sicle_masks_int32.npy" \
  --dilate "$CLIP_DILATE"
cp -f "${RAW}/percell_sicle_log.txt" "${ALT_A}/" 2>/dev/null || true

echo "=== [02] Alternativa B — AND com Cellpose no paste ==="
mkdir -p "$ALT_B"
python percell_sicle_cellprob_pipeline.py \
  --from-dir "$FROM_DIR" \
  -o "$ALT_B" \
  --sicle-conn-opt "$SICLE_CONN" \
  --sicle-crit-opt "$SICLE_CRIT" \
  --sicle-alpha "$SICLE_ALPHA" \
  --image "$IMAGE" \
  --overlay-border-source "$OVERLAY_BORDER_SOURCE" \
  --overlay-cellpose-border-color "$OVERLAY_CELLPOSE_BORDER_COLOR" \
  "${COMPARE_FLAGS[@]}"

echo "=== [03] Alternativa C — maior componente sobreposta a Cellpose ==="
mkdir -p "$ALT_C"
python merge_postprocess.py components \
  --cellpose "$BASE_MASK" \
  --merged "$MERGED_RAW" \
  --out "${ALT_C}/merged_percell_sicle_masks_int32.npy"
cp -f "${RAW}/percell_sicle_log.txt" "${ALT_C}/" 2>/dev/null || true

DIRS=( "$RAW" "$ALT_A" "$ALT_B" "$ALT_C" )

if [[ "$RUN_ROUNDNESS_MERGE" == "1" ]]; then
  ALT_D="${OUT_ROOT}/04_alt_d_and_unless_round"
  echo "=== [04] And a menos que o FG SICLE pareça redondo (--and-unless-round) ==="
  mkdir -p "$ALT_D"
  python percell_sicle_cellprob_pipeline.py \
    --from-dir "$FROM_DIR" \
    -o "$ALT_D" \
    --sicle-conn-opt "$SICLE_CONN" \
    --sicle-crit-opt "$SICLE_CRIT" \
    --sicle-alpha "$SICLE_ALPHA" \
    --image "$IMAGE" \
    --overlay-border-source "$OVERLAY_BORDER_SOURCE" \
    --overlay-cellpose-border-color "$OVERLAY_CELLPOSE_BORDER_COLOR" \
    "${COMPARE_FLAGS[@]}" \
    --disable-and-merge \
    --and-unless-round \
    --min-fg-circularity "$MIN_FG_CIRCULARITY" \
    --min-fg-solidity "$MIN_FG_SOLIDITY"
  DIRS+=( "$ALT_D" )
fi

compare_one () {
  local name="$1" dir="$2"
  echo "=== compare vs step04: $name ==="
  python compare_segmentation_masks_diff.py \
    --mask-a "$BASE_MASK" \
    --mask-b "${dir}/merged_percell_sicle_masks_int32.npy" \
    -o "${dir}/compare_final_vs_step04" \
    --also-save-diff-only-rgb
}

for d in "${DIRS[@]}"; do
  compare_one "$(basename "$d")" "$d"
done

echo "=== Overlays RGB (todas as variantes; Cellpose+SICLE se OVERLAY_BORDER_SOURCE=both) ==="
for d in "${DIRS[@]}"; do
  python write_merged_percell_overlay.py \
    --image "$IMAGE" \
    --masks "${d}/merged_percell_sicle_masks_int32.npy" \
    --out "${d}/merged_percell_sicle_overlay.png" \
    --overlay-source "$OVERLAY_BORDER_SOURCE" \
    --cellpose "$BASE_MASK" \
    --cellpose-border-color "$OVERLAY_CELLPOSE_BORDER_COLOR"
done

if [[ "$GEN_AREA_CSV" == "1" ]]; then
  echo "=== CSVs de áreas (opcional) ==="
  for d in "${DIRS[@]}"; do
    python percell_sicle_cellpose_area_report.py \
      --from-dir "$FROM_DIR" \
      --out-dir "$d" \
      -o "${d}/sicle_vs_cellpose_areas.csv"
  done
fi

echo "Concluído. Saídas sob: $OUT_ROOT"
for d in "${DIRS[@]}"; do
  echo "  $d"
done
