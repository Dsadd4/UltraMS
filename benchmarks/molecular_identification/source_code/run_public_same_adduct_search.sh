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
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
MANIFEST="$BASE/fig3h_manifest_strict_adduct_v4"
LIBRARY="$ROOT/train/comparison/lib_search"
LOGS="$BASE/fig3h_s2s_readout_v1/logs"
mkdir -p "$LOGS" "$LIBRARY/output" "$LIBRARY/search"

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib_spectra
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset ultrams_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset dreams_ssl_weight
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT" --spectra-only
fi
if [[ ! -s "$MANIFEST/manifest_summary.json" ]]; then
  python -u "$PUBLIC/original/build_library_manifest.py" \
    --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --skip-reference-regression --output-dir "$MANIFEST" \
    2>&1 | tee "$LOGS/00_manifest.log"
fi

for backbone in ultra dreams; do
  checkpoint="$LIBRARY/output/lib_search_${backbone}_ul2_best.pt"
  training_result="$LIBRARY/output/lib_search_finetune.json"
  if [[ ! -s "$checkpoint" || ! -s "$training_result" ]] || \
     ! python -c 'import json,sys; run=json.load(open(sys.argv[1])).get(sys.argv[2], {}); sys.exit(0 if len(run.get("history", [])) == 20 else 1)' "$training_result" "$backbone"; then
    python -u "$PUBLIC/train_library_search.py" \
      --project-root "$ROOT" --source-model-root "$REPO/training/ultrams_training/model" \
      --backbone "$backbone" --adduct-scope strict --train-only \
      --epochs 20 --batch 256 --n-unfreeze 2 --device "$DEVICE" \
      --output-dir "$LIBRARY/output" \
      2>&1 | tee "$LOGS/01_train_${backbone}.log"
  fi
  complete=true
  for mode in pos neg; do
    for role in query lib; do
      [[ -s "$LIBRARY/output/emb_msnlib_${mode}_${role}_${backbone}.npy" ]] || complete=false
      [[ -s "$LIBRARY/output/emb_msnlib_${mode}_${role}_smis.json" ]] || complete=false
    done
  done
  if [[ "$complete" != true ]]; then
    python -u "$PUBLIC/embed_same_adduct_library.py" \
      --workspace "$ROOT" --manifest-dir "$MANIFEST" \
      --checkpoint "$checkpoint" --backbone "$backbone" \
      --output-dir "$LIBRARY/output" --device "$DEVICE" \
      2>&1 | tee "$LOGS/02_embed_${backbone}.log"
  fi
done
if [[ ! -s "$LIBRARY/search/top1_ultra_pos_test.npz" || ! -s "$LIBRARY/search/top1_ultra_neg_test.npz" \
   || ! -s "$LIBRARY/search/top1_dreams_pos_test.npz" || ! -s "$LIBRARY/search/top1_dreams_neg_test.npz" ]]; then
  python -u "$PUBLIC/evaluate_same_adduct_backbones.py" \
    --manifest-dir "$MANIFEST" --embedding-dir "$LIBRARY/output" \
    --output-dir "$LIBRARY/search" \
    2>&1 | tee "$LOGS/03_backbone_search.log"
fi

bash "$PUBLIC/run_same_adduct_search.sh"
