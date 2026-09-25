#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib_spectra
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset single_spectrum_predictions
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT" --spectra-only
fi
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib_with_rt.csv" ]]; then
  python "$PUBLIC/prepare_msnlib_rt.py" --root "$ROOT"
fi
bash "$PUBLIC/run_multiple_spectrum_identification.sh"
bash "$PUBLIC/run_label_blind_grouping.sh"
