#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
READOUT="$BASE/chemberta_readout_v1/formal_runs"
ULTRA="$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl"
DREAMS="$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_dreams_test.jsonl"
inputs=(
  --input "UltraMS|none|$ULTRA"
  --input "DreaMS|none|$DREAMS"
  --input "Linear|0|$READOUT/linear/seed_0/evaluation/per_spectrum_msnlib_mass_linear_chemberta_test.jsonl"
  --input "DeepSets|0|$READOUT/deepsets/seed_0/evaluation/per_spectrum_msnlib_mass_deepsets_chemberta_test.jsonl"
  --input "Fourier projection|0|$READOUT/fourier_projection/seed_0/evaluation/per_spectrum_msnlib_mass_fourier_projection_chemberta_test.jsonl"
  --input "UltraMS codebook|0|$READOUT/ultrams_codebook/seed_0/evaluation/per_spectrum_msnlib_mass_ultrams_codebook_chemberta_test.jsonl"
)

python -u "$SCRIPTS/evaluate_multiple_spectrum_voting.py" \
  "${inputs[@]}" --depth 50 --formal-final-six \
  --output-dir "$BASE/final_user_selected_six_v1/multiple_spectrum_voting"
