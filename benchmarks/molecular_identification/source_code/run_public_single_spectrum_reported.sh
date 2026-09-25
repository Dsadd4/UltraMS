#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a directory for the reported MSnLib evaluation}"
SELECTED="${2:-ultrams}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export LIGHT_ULTRA_ROOT="$ROOT"

case "$SELECTED" in
  ultrams)
    MODEL=rt_only_d11
    PROJECTION_DATASET=single_spectrum_projection
    BACKBONE_DATASET=ultrams_weight
    PROJECTION_SHA=619cc886b6dc6b11c22f97d203a2788eb4ff2eb32cfb475e110500c39dbfd786
    ;;
  dreams)
    MODEL=dreams
    PROJECTION_DATASET=single_spectrum_dreams_projection
    BACKBONE_DATASET=dreams_ssl_weight
    PROJECTION_SHA='c66591093862884084fe598a2ed6ed883474556aee31e9e8d3d2427cdeeda''e31'
    ;;
  *)
    echo "Select ultrams or dreams" >&2
    exit 2
    ;;
esac

PROJECTION="$ROOT/train/output/comparison/proj_msnlib_mass_${MODEL}_seed42.pt"
python - "$PROJECTION" "$PROJECTION_SHA" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
if path.exists():
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit("This workspace contains a different trained projection; use a separate directory for the reported evaluation.")
PY

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset "$BACKBONE_DATASET"
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset chemberta
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset "$PROJECTION_DATASET"
if [[ "$SELECTED" == dreams ]]; then
  python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
fi

if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" || ! -s "$ROOT/datasets/MSnLib/MSnLib_candidates.json" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT"
fi

mkdir -p "$ROOT/train/output/comparison/benchmark_logs" "$ROOT/results"
cd "$ROOT/train"
python -u "$PUBLIC/original/evaluate_molecule_identification.py" \
  --models "$MODEL" --datasets msnlib --load-proj \
  --save-raw-scores --save-per-spectrum --per-spectrum-topk 50 \
  --device "${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}" \
  2>&1 | tee "$ROOT/train/output/comparison/benchmark_logs/single_spectrum_${SELECTED}_reported.log"

cp "$ROOT/train/output/comparison/10_plot_results.json" \
  "$ROOT/results/single_spectrum_${SELECTED}_reported.json"
cp "$ROOT/train/output/comparison/10_per_spectrum_msnlib_mass_${MODEL}_test.jsonl" \
  "$ROOT/results/single_spectrum_${SELECTED}_reported_test.jsonl"
python - "$ROOT/results/single_spectrum_${SELECTED}_reported.json" "$MODEL" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))["msnlib_mass"][sys.argv[2]]["test"]
print(f"{sys.argv[2]} MSnLib test: {result['n_queries']} spectra, Top-1 {result['top1']:.2f}%")
if result["n_queries"] != 57437:
    raise SystemExit("Unexpected number of held-out MSnLib queries")
PY
