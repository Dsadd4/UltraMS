#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass an empty or existing benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"
mkdir -p "$ROOT/train/output/comparison/benchmark_logs"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset ultrams_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset chemberta
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" || ! -s "$ROOT/datasets/MSnLib/MSnLib_candidates.json" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT"
fi
cd "$ROOT/train"
python -u "$PUBLIC/original/evaluate_molecule_identification.py" \
  --models rt_only_d11 --datasets msnlib --proj-epochs 50 --seed 42 --temperature 0.05 \
  --save-proj --save-raw-scores --save-per-spectrum --per-spectrum-topk 50 \
  --device "${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}" \
  2>&1 | tee "$ROOT/train/output/comparison/benchmark_logs/single_spectrum_ultrams.log"

mkdir -p "$ROOT/results"
cp "$ROOT/train/output/comparison/10_plot_results.json" "$ROOT/results/single_spectrum_ultrams.json"
cp "$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl" \
  "$ROOT/results/single_spectrum_ultrams_test.jsonl"
python - "$ROOT/results/single_spectrum_ultrams.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))['msnlib_mass']['rt_only_d11']['test']
print(f"UltraMS MSnLib test: {result['n_queries']} spectra, Top-1 {result['top1']:.2f}%")
if result['n_queries'] != 57437:
    raise SystemExit('Unexpected test-query count; inspect the source conversion and folds')
PY
