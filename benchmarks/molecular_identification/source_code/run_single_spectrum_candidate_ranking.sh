#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to a workspace prepared from public inputs}"
MODE="${1:-train}"
if [[ "$MODE" != train && "$MODE" != reported ]]; then
  echo "Select train or reported" >&2
  exit 2
fi
export LIGHT_ULTRA_ROOT="$ROOT"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
CSV="$ROOT/datasets/MSnLib/MSnLib.csv"
FFN_CACHE="$BASE/ffn_cache"
PEAK_CACHE="$BASE/additional_baselines_v2_fullprecision/peak_cache_float32"
CODEBOOK_CACHE="$BASE/ultrams_codebook_pool_v1"
TARGETS="$BASE/spec2vec3abg_projection_targets"
MOLECULE_CACHE="$ROOT/train/output/comparison/mol_emb_cache/msnlib_mass_chemberta_full.pt"
RUN="$BASE/chemberta_readout_v1/formal_runs"
LOGS="$BASE/benchmark_logs"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
mkdir -p "$LOGS"
cd "$ROOT/train"

encoders=()
for model in rt_only_d11 dreams; do
  test_jsonl="$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_${model}_test.jsonl"
  if [[ ! -s "$test_jsonl" ]] || [[ $(wc -l < "$test_jsonl") -ne 57437 ]]; then
    encoders+=("$model")
  fi
done
if (( ${#encoders[@]} )); then
  projection_mode=(--early-stop --proj-epochs 30 --save-proj)
  if [[ "$MODE" == reported ]]; then
    projection_mode=(--load-proj)
  fi
  python -u "$SCRIPTS/evaluate_molecule_identification.py" \
    --models "${encoders[@]}" --datasets msnlib "${projection_mode[@]}" \
    --save-raw-scores --save-per-spectrum --per-spectrum-topk 50 --device "$DEVICE" \
    2>&1 | tee "$LOGS/01_ultrams_dreams_candidates.log"
fi

if [[ ! -f "$BASE/effective_candidates/MSnLib_candidates_effective_chemberta.json" ]]; then
  python -u "$SCRIPTS/prepare_effective_candidates.py" \
    --csv "$CSV" --candidates "$ROOT/datasets/MSnLib/MSnLib_candidates.json" \
    --chemberta-cache "$MOLECULE_CACHE" \
    --ultrams-jsonl "$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl" \
    --dreams-jsonl "$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_dreams_test.jsonl" \
    --output-dir "$BASE/effective_candidates" \
    2>&1 | tee "$LOGS/02_effective_candidates.log"
fi

if [[ ! -f "$FFN_CACHE/prepare_summary.json" ]]; then
  python -u "$SCRIPTS/prepare_ffn_cache.py" --csv "$CSV" --output-dir "$FFN_CACHE" \
    2>&1 | tee "$LOGS/02_spectrum_bins.log"
fi
if [[ ! -f "$PEAK_CACHE/DONE.json" ]]; then
  python -u "$SCRIPTS/prepare_peakset_cache.py" --csv "$CSV" \
    --ffn-cache-dir "$FFN_CACHE" --output-dir "$PEAK_CACHE" \
    2>&1 | tee "$LOGS/03_peak_sets.log"
fi
if [[ ! -f "$CODEBOOK_CACHE/DONE.json" ]]; then
  python -u "$SCRIPTS/extract_ultrams_codebook_pool.py" --csv "$CSV" \
    --output-dir "$CODEBOOK_CACHE" --device "$DEVICE" \
    2>&1 | tee "$LOGS/04_ultrams_codebook.log"
fi
if [[ ! -f "$TARGETS/projection_target_summary.json" ]]; then
  python -u "$SCRIPTS/prepare_chemberta_targets.py" \
    --manifest-dir "$FFN_CACHE" --chemberta-cache "$MOLECULE_CACHE" \
    --output-dir "$TARGETS" 2>&1 | tee "$LOGS/05_molecule_targets.log"
fi

if [[ "$MODE" == train ]]; then
  for model in linear deepsets fourier_projection ultrams_codebook; do
    python -u "$SCRIPTS/train_chemberta_readout.py" \
      --model "$model" --ffn-cache-dir "$FFN_CACHE" --peak-cache-dir "$PEAK_CACHE" \
      --targets-dir "$TARGETS" --codebook-cache-dir "$CODEBOOK_CACHE" \
      --output-dir "$RUN" --seed 0 --epochs 30 --device "$DEVICE" \
      2>&1 | tee "$LOGS/06_train_${model}.log"
  done
else
  for mapping in linear:linear deepsets:deepsets fourier_projection:fourier ultrams_codebook:codebook; do
    model="${mapping%%:*}"
    asset="${mapping#*:}"
    source="$ROOT/benchmark_assets/molecular_identification/single_spectrum_readouts/$asset.pt"
    destination="$RUN/$model/seed_0/best.pt"
    mkdir -p "$(dirname "$destination")"
    if [[ ! -f "$destination" ]]; then
      cp "$source" "$destination"
    elif ! cmp -s "$source" "$destination"; then
      echo "Existing $model readout differs from the reported weight; use another workspace" >&2
      exit 1
    fi
  done
fi

python -u "$SCRIPTS/evaluate_chemberta_readout.py" \
  --models linear deepsets fourier_projection ultrams_codebook \
  --csv "$CSV" \
  --candidates "$BASE/effective_candidates/MSnLib_candidates_effective_chemberta.json" \
  --chemberta-cache "$MOLECULE_CACHE" \
  --ffn-cache-dir "$FFN_CACHE" --peak-cache-dir "$PEAK_CACHE" \
  --codebook-cache-dir "$CODEBOOK_CACHE" --run-dir "$RUN" --seed 0 \
  --reference-test-jsonl "$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl" \
  --device "$DEVICE" 2>&1 | tee "$LOGS/07_evaluate_six_methods.log"
