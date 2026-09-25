#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
FIXED_INPUTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../data/benchmark_inputs" && pwd)"
MANIFEST="$BASE/fig3h_manifest_composite_adduct_v1"
INPUTS="$BASE/fig3h_composite_readout_v1/inputs"
RUN="$BASE/fig3h_composite_readout_v1"
MODELS="$RUN/models"
EMBEDDINGS="$RUN/embeddings"
LOGS="$RUN/logs"
RESULT="$BASE/fig3h_other_strict_adduct_v3_six_methods"
MOLECULE_CACHE="${ULTRAMS_ADDUCT_MOLECULE_CACHE:-$ROOT/train/output/comparison/mol_emb_cache/msnlib_mass_chemberta_full.pt}"
mkdir -p "$LOGS"

if [[ ! -f "$MANIFEST/manifest_summary.json" ]]; then
  reference_args=()
  if [[ ! -f "$ROOT/train/comparison/output/emb_msnlib_pos_query_smis.json" ]]; then
    reference_args+=(--skip-reference-regression)
  fi
  python -u "$SCRIPTS/build_composite_readout_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --reference-embedding-dir "$ROOT/train/comparison/output" \
    --output-dir "$MANIFEST" "${reference_args[@]}" 2>&1 | tee "$LOGS/00_manifest.log"
fi
if [[ ! -f "$INPUTS/DONE.json" ]]; then
  python -u "$SCRIPTS/prepare_fig3h_readout_inputs.py" \
    --manifest-dir "$MANIFEST" --output-dir "$INPUTS" \
    --molecule-cache "$MOLECULE_CACHE" \
    --retrieval-script "$SCRIPTS/evaluate_molecule_identification.py" \
    --device "$DEVICE" 2>&1 | tee "$LOGS/00_inputs.log"
fi

for model in linear deepsets fourier_projection ultrams_codebook; do
  for mode in pos neg; do
    python -u "$SCRIPTS/train_fig3h_readout.py" \
      --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --output-dir "$MODELS" \
      --model "$model" --mode "$mode" --seed 0 --epochs 30 --device "$DEVICE" \
      2>&1 | tee "$LOGS/01_train_${model}_${mode}.log"
    python -u "$SCRIPTS/embed_composite_readout.py" \
      --cache-dir "$INPUTS" --model-dir "$MODELS" --output-dir "$EMBEDDINGS" \
      --model "$model" --mode "$mode" --seed 0 --device "$DEVICE" \
      2>&1 | tee "$LOGS/02_embed_${model}_${mode}.log"
  done
done

python -u "$SCRIPTS/evaluate_other_strict_adduct_six_methods.py" \
  --csv "$ROOT/datasets/MSnLib/MSnLib.csv" --manifest-dir "$MANIFEST" \
  --embedding-dir "$ROOT/train/comparison/output" \
  --readout-embedding-dir "$EMBEDDINGS" \
  --reference-task-rows "$FIXED_INPUTS/additional_adduct_reference_rows.csv.gz" \
  --output-dir "$RESULT" --device "$DEVICE" \
  2>&1 | tee "$LOGS/03_evaluate_additional.log"
