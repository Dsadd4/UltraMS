#!/usr/bin/env bash
set -euo pipefail

# Run from a benchmark workspace containing the task inputs and generated features.
ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
PUBLIC_SOURCE="$(dirname "$SCRIPTS")"
SOURCE_MANIFEST="$BASE/fig3h_manifest_strict_adduct_v4"
INPUTS="$BASE/fig3h_readout_reuse_v1/inputs"
BACKBONE_RUN="${ULTRAMS_COLLISION_ENERGY_OUTPUT_DIR:-$ROOT/train/output/showcase}"
export ULTRAMS_COLLISION_ENERGY_OUTPUT_DIR="$BACKBONE_RUN"
BACKBONE_LOGS="$BACKBONE_RUN/logs"
RUN="$BASE/crossce_readout_fixed_v1"
MANIFEST="$RUN/manifest"
MODELS="$RUN/models"
SEARCH="$RUN/search"
LOGS="$RUN/logs"
mkdir -p "$LOGS"
mkdir -p "$BACKBONE_LOGS"

ULTRA_CROSS_CE_CKPT="${ULTRAMS_CROSS_CE_ULTRA_CHECKPOINT:-$BACKBONE_RUN/cross_ce_ultra_v2_ul2_best.pt}"
DREAMS_CROSS_CE_CKPT="${ULTRAMS_CROSS_CE_DREAMS_CHECKPOINT:-$BACKBONE_RUN/cross_ce_dreams_v2_ul2_best.pt}"
checkpoint_ready() {
  local checkpoint="$1" backbone="$2"
  [[ -s "$checkpoint" ]] || return 1
  if [[ "${ULTRAMS_COLLISION_ENERGY_REQUIRE_COMPLETE_TRAIN:-0}" != 1 ]]; then return 0; fi
  local summary="$BACKBONE_RUN/cross_ce_finetune_v2.json"
  [[ -s "$summary" ]] || return 1
  python -c 'import json,sys; run=json.load(open(sys.argv[1])).get(sys.argv[2] + "_finetune", {}); sys.exit(0 if run.get("epochs") == 20 and run.get("unfreeze_last") == 2 and len(run.get("history", [])) == 20 else 1)' "$summary" "$backbone"
}
if ! checkpoint_ready "$ULTRA_CROSS_CE_CKPT" ultra; then
  python -u "$SCRIPTS/train_collision_energy_backbones.py" \
    --backbone ultra --epochs 20 --unfreeze-last 2 --tag ul2 --device "$DEVICE" \
    2>&1 | tee "$BACKBONE_LOGS/01_train_ultrams.log"
fi
if ! checkpoint_ready "$DREAMS_CROSS_CE_CKPT" dreams; then
  python -u "$SCRIPTS/train_collision_energy_backbones.py" \
    --backbone dreams --epochs 20 --unfreeze-last 2 --tag ul2 --device "$DEVICE" \
    2>&1 | tee "$BACKBONE_LOGS/01_train_dreams.log"
fi
export ULTRAMS_CROSS_CE_ULTRA_CHECKPOINT="$ULTRA_CROSS_CE_CKPT"
export ULTRAMS_CROSS_CE_DREAMS_CHECKPOINT="$DREAMS_CROSS_CE_CKPT"
python -u "$SCRIPTS/evaluate_collision_energy_backbones.py" --device "$DEVICE" \
  2>&1 | tee "$BACKBONE_LOGS/02_evaluate_backbones.log"

if [[ ! -f "$SOURCE_MANIFEST/manifest_summary.json" ]]; then
  python -u "$SCRIPTS/build_library_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --skip-reference-regression --output-dir "$SOURCE_MANIFEST" 2>&1 | tee "$LOGS/00_manifest.log"
fi
if [[ ! -f "$INPUTS/DONE.json" ]]; then
  python -u "$SCRIPTS/prepare_fig3h_readout_inputs.py" \
    --manifest-dir "$SOURCE_MANIFEST" --output-dir "$INPUTS" \
    --skip-molecule-targets \
    --retrieval-script "$SCRIPTS/evaluate_molecule_identification.py" \
    --device "$DEVICE" 2>&1 | tee "$LOGS/00_inputs.log"
fi
if [[ ! -f "$INPUTS/pos/target_summary.json" || ! -f "$INPUTS/neg/target_summary.json" ]]; then
  python -u "$PUBLIC_SOURCE/prepare_collision_energy_targets.py" \
    --manifest-dir "$SOURCE_MANIFEST" --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --model-dir "$ROOT/model/feature/ChemBERTa-100M-MLM" \
    --output-dir "$INPUTS" --device "$DEVICE" \
    2>&1 | tee "$LOGS/00_targets.log"
fi

python -u "$SCRIPTS/prepare_crossce_readout_manifest.py" \
  --manifest-dir "$SOURCE_MANIFEST" --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
  --output-dir "$MANIFEST" 2>&1 | tee "$LOGS/01_manifest.log"
python -u "$PUBLIC_SOURCE/evaluate_collision_energy_test.py" \
  --manifest-dir "$SOURCE_MANIFEST" --crossce-manifest-dir "$MANIFEST" \
  --backbone-results-dir "$BACKBONE_RUN" --output-dir "$BACKBONE_RUN" \
  2>&1 | tee "$LOGS/01_backbone_test.log"

for model in linear deepsets fourier_projection ultrams_codebook; do
  for mode in pos neg; do
    python -u "$SCRIPTS/train_crossce_readout_fixed.py" \
      --source-manifest-dir "$SOURCE_MANIFEST" --crossce-manifest-dir "$MANIFEST" \
      --cache-dir "$INPUTS" --output-dir "$MODELS" \
      --model "$model" --mode "$mode" --seed 0 --epochs 30 --device "$DEVICE" \
      2>&1 | tee "$LOGS/02_train_${model}_${mode}.log"
    python -u "$SCRIPTS/evaluate_crossce_readout.py" \
      --source-manifest-dir "$SOURCE_MANIFEST" --crossce-manifest-dir "$MANIFEST" \
      --cache-dir "$INPUTS" --model-dir "$MODELS" --output-dir "$SEARCH" \
      --model "$model" --mode "$mode" --seed 0 --device "$DEVICE" \
      2>&1 | tee "$LOGS/03_search_${model}_${mode}.log"
  done
done

python -u "$PUBLIC_SOURCE/build_collision_energy_rows.py" \
  --source-directory "$BASE" --backbone-results-dir "$BACKBONE_RUN" \
  --output "$RUN/collision_energy_queries.csv.gz" \
  2>&1 | tee "$LOGS/04_combine_six_methods.log"
