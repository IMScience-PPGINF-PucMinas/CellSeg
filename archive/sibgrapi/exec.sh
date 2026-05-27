#!/usr/bin/env bash
# Per-cell SICLE (gradvmax + minsc), sem AND com máscara Cellpose no paste;
# compara o merge final com step04_masks_uint16.npy.
#
# Requer RunSICLE com suporte a gradvmax (ex.: doutorado/SICLE/bin/RunSICLE).
# Se o binário não estiver no PATH padrão do script Python, defina:
#   export SICLE_BIN=/home/lacerda/doutorado/SICLE/bin/RunSICLE

set -euo pipefail

cd "/home/lacerda/doutorado/new_pipeline"
export PYTHONPATH=../cellpose

IMG="/home/lacerda/doutorado/GR07-1.svs_slice1.tiff"
OUTDIR="./percell_sicle_out_gradvmax_no_and"
BASE="./cp_flow_out/step04_masks_uint16.npy"

# Opcional: força o binário do fork
# export SICLE_BIN="/home/lacerda/doutorado/SICLE/bin/RunSICLE"

mkdir -p "$OUTDIR"

python percell_sicle_cellprob_pipeline.py \
  --from-dir ./cp_flow_out \
  -o "$OUTDIR" \
  --sicle-conn-opt gradvmax \
  --sicle-crit-opt maxsc \
  --sicle-alpha 2.0 \
  --image "$IMG" \
  --overlay-border-source both \
  --overlay-cellpose-border-color 255,0,0 \
  --write-compare-vs-step04 \
  --disable-and-merge

# Relatório CSV: área Cellpose vs SICLE por label (replicável; ver --help para clip opcional)
python percell_sicle_cellpose_area_report.py \
  --from-dir ./cp_flow_out \
  --out-dir "$OUTDIR" \
  -o "$OUTDIR/sicle_vs_cellpose_areas.csv"

echo "OK: $OUTDIR/compare_final_vs_step04/ (diff A=step04, B=merged)"
echo "CSV: $OUTDIR/sicle_vs_cellpose_areas.csv"
