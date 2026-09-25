#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
FIXED_INPUTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../data/benchmark_inputs" && pwd)"
MANIFEST="$BASE/fig3h_manifest_strict_adduct_v4"
INPUTS="$BASE/fig3h_readout_reuse_v1/inputs"
RUN="$BASE/fig3h_s2s_readout_v1"
MODELS="$RUN/models"
SEARCH="$RUN/search"
BACKBONE_SCORES="$ROOT/train/comparison/lib_search/search"
LOGS="$RUN/logs"
mkdir -p "$LOGS"

if [[ ! -f "$MANIFEST/manifest_summary.json" ]]; then
  python -u "$SCRIPTS/build_library_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --skip-reference-regression --output-dir "$MANIFEST" 2>&1 | tee "$LOGS/00_manifest.log"
fi
if [[ ! -f "$INPUTS/DONE.json" ]]; then
  python -u "$SCRIPTS/prepare_fig3h_readout_inputs.py" \
    --manifest-dir "$MANIFEST" --output-dir "$INPUTS" \
    --skip-molecule-targets \
    --retrieval-script "$SCRIPTS/evaluate_molecule_identification.py" \
    --device "$DEVICE" 2>&1 | tee "$LOGS/00_inputs.log"
fi

arguments=()
for model in linear deepsets fourier_projection ultrams_codebook; do
  case "$model" in
    linear) display=Linear ;;
    deepsets) display=DeepSets ;;
    fourier_projection) display=Fourier ;;
    ultrams_codebook) display=Codebook ;;
  esac
  for mode in pos neg; do
    python -u "$SCRIPTS/train_fig3h_s2s_readout.py" \
      --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --output-dir "$MODELS" \
      --model "$model" --mode "$mode" --seed 0 --device "$DEVICE" \
      2>&1 | tee "$LOGS/01_train_${model}_${mode}.log"
    python -u "$SCRIPTS/embed_search_fig3h_s2s_readout.py" \
      --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --model-dir "$MODELS" \
      --output-dir "$SEARCH" --model "$model" --mode "$mode" --seed 0 --device "$DEVICE" \
      2>&1 | tee "$LOGS/02_search_${model}_${mode}.log"
    for split in val test; do
      arguments+=(--input "$display|$mode|$split|0|$SEARCH/$model/$mode/seed_0/$split/top1.npz")
    done
  done
done

backbone_args=(--existing-summary "$FIXED_INPUTS/same_adduct_search_summary.json")
if [[ -s "$BACKBONE_SCORES/top1_ultra_pos_test.npz" && -s "$BACKBONE_SCORES/top1_ultra_neg_test.npz" \
   && -s "$BACKBONE_SCORES/top1_dreams_pos_test.npz" && -s "$BACKBONE_SCORES/top1_dreams_neg_test.npz" ]]; then
  backbone_args=()
  for mode in pos neg; do
    backbone_args+=(--input "UltraMS|$mode|test|none|$BACKBONE_SCORES/top1_ultra_${mode}_test.npz")
    backbone_args+=(--input "DreaMS|$mode|test|none|$BACKBONE_SCORES/top1_dreams_${mode}_test.npz")
  done
fi
python -u "$SCRIPTS/evaluate_exact_fdr.py" \
  --manifest-summary "$MANIFEST/manifest_summary.json" \
  "${backbone_args[@]}" \
  "${arguments[@]}" --output-dir "$RUN/exact_fdr" \
  2>&1 | tee "$LOGS/03_exact_fdr.log"
