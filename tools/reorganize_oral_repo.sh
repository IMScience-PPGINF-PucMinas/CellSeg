#!/usr/bin/env bash
# One-time reorganization: Oral Epithelium focus, git-friendly layout.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

echo "=== Creating layout ==="
mkdir -p data/oral_epithelium/images data/oral_epithelium/annotations pipeline oral
mkdir -p outputs/{runs/single_roi,reviews/gold_standard_overlays}
mkdir -p archive/sibgrapi configs docs tools

OE="OralEpitheliumDB"
if [[ -d "$OE" ]]; then
  echo "=== Moving Oral Epithelium data from $OE ==="
  [[ -d "$OE/Original ROI images" ]] && mv "$OE/Original ROI images" data/oral_epithelium/images/original
  [[ -d "$OE/Normalized Images" ]] && mv "$OE/Normalized Images" data/oral_epithelium/images/normalized
  [[ -d "$OE/Gold_Standard_Instance_Segmentation_Colored" ]] && mv "$OE/Gold_Standard_Instance_Segmentation_Colored" data/oral_epithelium/annotations/instance_colored
  [[ -d "$OE/Gold_Standard_Instance_Segmentation" ]] && mv "$OE/Gold_Standard_Instance_Segmentation" data/oral_epithelium/annotations/instance
  [[ -d "$OE/Gold_Standard_Semantic_Segmentation" ]] && mv "$OE/Gold_Standard_Semantic_Segmentation" data/oral_epithelium/annotations/semantic
  [[ -d "$OE/test_single_roi_cellpose_sicle" ]] && mv "$OE/test_single_roi_cellpose_sicle" outputs/runs/single_roi
  [[ -d "$OE/gold_standard_overlay_review" ]] && mv "$OE/gold_standard_overlay_review" outputs/reviews/gold_standard_overlays
  for f in "$OE"/*.py; do
    [[ -f "$f" ]] && mv "$f" oral/
  done
  rm -f "$OE"/*.zip "$OE"/*:Zone.Identifier 2>/dev/null || true
  rm -rf "$OE"
fi

echo "=== Moving pipeline scripts ==="
for f in reproduce_cellpose_pipeline.py percell_sicle_cellprob_pipeline.py \
  percell_boundary_recall.py compare_segmentation_masks_diff.py merge_postprocess.py \
  mask_outline_overlay.py write_merged_percell_overlay.py; do
  [[ -f "$f" ]] && mv "$f" pipeline/
done
[[ -f evaluate_sibgrapi2026.py ]] && mv evaluate_sibgrapi2026.py pipeline/evaluate_instances.py

echo "=== Archiving SIBGRAPI / legacy scripts ==="
for f in build_all_methods_slices1_8.py build_br_merged_masks.py build_macro_methods_table.py \
  cellvit_infer_png.py complete_cellvit_br_benchmark.py extract_slices_lab_gt.py \
  plot_sicle_br_slice_analysis.py run_cellvit_br_benchmark.py percell_compare_sicle_cellpose.py \
  percell_sicle_cellpose_area_report.py test_sicle_path_costs.py regen_translucent_overlays.py \
  cellpose_to_idisf_pipeline.py cellpose_masks_modified_cellprob.py \
  cellpose_segmentation_pipeline_explained.py cellprob_heatmap.py convert_monuseg_tif_to_png.py \
  monuseg_dice_eval.py overlay_monuseg_annotations.py run_monuseg_cellpose_nuclick.py \
  run_monuseg_cellpose_sicle.py; do
  [[ -f "$f" ]] && mv "$f" archive/sibgrapi/
done
for f in sweep_*.sh run_sibgrapi*.sh run_sicle_nolin*.sh run_percell_three_alternatives.sh \
  run_path_cost_visuals.sh exec.sh; do
  [[ -f "$f" ]] && mv "$f" archive/sibgrapi/
done

echo "=== Removing large generated artifacts ==="
rm -rf data_sibgrapi2026 out_sibgrapi2026 out_sibgrapi2026_blur05 out_sibgrapi2026_clean \
  out_sibgrapi2026_fillonly out_sibgrapi2026_gradvmax out_sibgrapi2026_nolin \
  out_sibgrapi2026_nolin_noblur out_sibgrapi2026_sweep out_sibgrapi2026_sweep_v2 \
  out_sweep_blur out_sweep_blur_fine out_sweep_lit out_sweep_lit_v2 out_sweep_lit_v3 \
  out_gradvmaxmul out_cellvit_br percell_sicle_out cp_flow_out compare_out \
  compare_step4_vs_fused compare_step4_vs_percell remix_out reports gt checkpoints \
  cellpose_remix_out reproduce_cellpose_out mask_compare_out \
  cellpose_pipeline_demo_out cellpose_idisf_out cellpose_to_idisf_out 2>/dev/null || true

rm -f *.zip *:Zone.Identifier GR07-1.svs_slice1.tiff cellprob_heatmap.png \
  macro_methods_comparison.csv macro_methods_comparison.md \
  new_pipeline.zip new_pipeline_2.zip OralEpitheliumDB.zip data_sibgrapi2026.zip \
  *.log 2>/dev/null || true

echo "=== Done ==="
du -sh . data/oral_epithelium pipeline oral outputs archive 2>/dev/null || true
ls -la
