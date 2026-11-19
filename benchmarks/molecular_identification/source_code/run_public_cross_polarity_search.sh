#!/usr/bin/env bash
set -euo pipefail
PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
MANIFEST="$ROOT/train/output/comparison/molecular_benchmarks/same_adduct_manifest"
OUT="$ROOT/results/cross_polarity"
for dataset in msnlib_spectra ultrams_weight dreams_ssl_weight; do
  python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset "$dataset"
done
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
python "$PUBLIC/prepare_msnlib.py" --root "$ROOT" --spectra-only
if [[ ! -s "$MANIFEST/manifest_summary.json" ]]; then
  python "$PUBLIC/original/build_library_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" --skip-reference-regression --output-dir "$MANIFEST"
fi
for model in ultra dreams; do
  python -u "$PUBLIC/original/train_cross_polarity.py" \
    --root "$ROOT" --out "$OUT/models/$model" --backbone "$model" --device "$DEVICE"
done
python -u "$PUBLIC/original/evaluate_cross_polarity.py" --root "$ROOT" --out "$OUT/molecular_ranking"
