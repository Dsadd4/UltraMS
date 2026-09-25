#!/usr/bin/env bash
set -euo pipefail
RUN="${ULTRAMS_BENCHMARK_ROOT:?Set ULTRAMS_BENCHMARK_ROOT}"
export LIGHT_ULTRA_ROOT="$RUN"
export PYTHONUNBUFFERED=1
cd "$RUN/train"
exec python -u "${ULTRAMS_PUBLIC_RELEASE_ROOT:?Set ULTRAMS_PUBLIC_RELEASE_ROOT}/benchmarks/molecular_identification/source_code/original/evaluate_molecule_identification.py" \
  --models dreams --datasets msnlib --device cuda:0 \
  --seed 42 --temperature 0.05 --early-stop --proj-epochs 30 \
  --save-proj --save-raw-scores --save-per-spectrum --per-spectrum-topk 50
