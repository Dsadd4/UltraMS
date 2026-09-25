#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset nplib1
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset base_weights
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset benchmark_weights
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
python "$PUBLIC/prepare_nplib1.py" --root "$ROOT"
bash "$PUBLIC/run_nplib1_candidate_ranking.sh"
