#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset ultrams_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset dreams_ssl_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset chemberta
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" || ! -s "$ROOT/datasets/MSnLib/MSnLib_candidates.json" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT"
fi
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib_with_rt.csv" ]]; then
  python "$PUBLIC/prepare_msnlib_rt.py" --root "$ROOT"
fi
bash "$PUBLIC/run_single_spectrum_candidate_ranking.sh"
bash "$PUBLIC/run_multiple_spectrum_identification.sh"
bash "$PUBLIC/run_label_blind_grouping.sh"
