#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
READOUT="$BASE/chemberta_readout_v1/formal_runs"
GROUPING_DIR="$BASE/label_blind_grouping"
ORACLE_SUMMARY="$ROOT/train/output/comparison/14_cluster_consensus/cluster_consensus_summary.json"
if [[ ! -f "$ORACLE_SUMMARY" ]]; then
  mkdir -p "$(dirname "$ORACLE_SUMMARY")"
  cp "$(dirname "$SCRIPTS")/data/label_blind_grouping/cluster_consensus_summary.json" "$ORACLE_SUMMARY"
fi
if [[ ! -f "$ROOT/datasets/MSnLib/MSnLib_with_rt.csv" ]]; then
  python -u "$(dirname "$SCRIPTS")/prepare_msnlib_rt.py" --root "$ROOT"
fi
inputs=(
  --input "UltraMS|none|$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl"
  --input "DreaMS|none|$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_dreams_test.jsonl"
  --input "Linear|0|$READOUT/linear/seed_0/evaluation/per_spectrum_msnlib_mass_linear_chemberta_test.jsonl"
  --input "DeepSets|0|$READOUT/deepsets/seed_0/evaluation/per_spectrum_msnlib_mass_deepsets_chemberta_test.jsonl"
  --input "Fourier projection|0|$READOUT/fourier_projection/seed_0/evaluation/per_spectrum_msnlib_mass_fourier_projection_chemberta_test.jsonl"
  --input "UltraMS codebook|0|$READOUT/ultrams_codebook/seed_0/evaluation/per_spectrum_msnlib_mass_ultrams_codebook_chemberta_test.jsonl"
)

python -u "$SCRIPTS/label_blind_grouping/scripts/run_analysis.py" \
  --project-root "$ROOT" --output-dir "$GROUPING_DIR"
python -u "$SCRIPTS/evaluate_frozen_groups.py" \
  --frozen-package "$GROUPING_DIR" "${inputs[@]}" \
  --candidate-depth 50 --bootstrap-replicates 10000 --formal-final-six \
  --output-dir "$BASE/final_user_selected_six_v1/frozen_groups"
