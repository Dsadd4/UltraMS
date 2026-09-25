#!/usr/bin/env bash
set -euo pipefail

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$PUBLIC/../../.." && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
EMBEDDINGS="$ROOT/train/comparison/output"
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
MANIFEST="$BASE/fig3h_manifest_composite_adduct_v1"
LOGS="$BASE/fig3h_composite_readout_v1/logs"
mkdir -p "$LOGS" "$EMBEDDINGS"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib_spectra
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset ultrams_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset dreams_ssl_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset chemberta
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT" --spectra-only
fi

for backbone in ultra dreams; do
  trained=false
  training_result="$EMBEDDINGS/lib_search_finetune.json"
  if [[ -s "$EMBEDDINGS/lib_search_${backbone}_ul2_best.pt" && -s "$training_result" ]] && \
     python -c 'import json,sys; run=json.load(open(sys.argv[1])).get(sys.argv[2], {}); sys.exit(0 if len(run.get("history", [])) == 20 else 1)' "$training_result" "$backbone"; then
    trained=true
  fi
  complete=true
  for mode in pos neg; do
    for split in query lib; do
      [[ -s "$EMBEDDINGS/emb_msnlib_${mode}_${split}_${backbone}.npy" ]] || complete=false
      [[ -s "$EMBEDDINGS/emb_msnlib_${mode}_${split}_smis.json" ]] || complete=false
    done
  done
  if [[ "$trained" == true && "$complete" == true ]]; then
    continue
  fi
  resume=()
  if [[ "$trained" == true ]]; then
    resume+=(--embed-only)
  fi
  python -u "$PUBLIC/train_library_search.py" \
    --project-root "$ROOT" --source-model-root "$REPO/training/ultrams_training/model" \
    --backbone "$backbone" --device "$DEVICE" "${resume[@]}" \
    2>&1 | tee "$LOGS/00_${backbone}_library_encoder.log"
done

if [[ ! -s "$MANIFEST/manifest_summary.json" ]]; then
  python -u "$PUBLIC/original/build_composite_readout_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --reference-embedding-dir "$EMBEDDINGS" --output-dir "$MANIFEST" \
    2>&1 | tee "$LOGS/01_manifest.log"
fi

export ULTRAMS_ADDUCT_MOLECULE_CACHE="$ROOT/train/output/comparison/mol_emb_cache/additional_adduct_chemberta_targets.pt"
python -u "$PUBLIC/prepare_adduct_molecule_targets.py" \
  --manifest-dir "$MANIFEST" \
  --model-dir "$ROOT/model/feature/ChemBERTa-100M-MLM" \
  --output "$ULTRAMS_ADDUCT_MOLECULE_CACHE" --device "$DEVICE" \
  2>&1 | tee "$LOGS/02_molecule_targets.log"

bash "$PUBLIC/run_additional_adduct_search.sh"
