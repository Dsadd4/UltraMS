#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
  echo "Usage: bash run_public_collision_energy_search.sh WORKSPACE [--weights published|train] [--backbones-only]"
  echo "Published weights are used by default; --weights train runs the original 20-epoch fine-tuning."
  [[ $# -gt 0 ]] && exit 0 || exit 2
fi

PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?Pass a benchmark workspace directory}"
shift
WEIGHTS=published
SCOPE=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="${2:?Pass published or train}"; shift 2 ;;
    --backbones-only) SCOPE=backbones; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$WEIGHTS" in
  published|train) ;;
  *) echo "--weights must be published or train" >&2; exit 2 ;;
esac

mkdir -p "$ROOT"
ROOT="$(cd -- "$ROOT" && pwd)"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
BACKBONE_RUN="$ROOT/train/output/showcase"
if [[ "$WEIGHTS" == train ]]; then
  BACKBONE_RUN="$BACKBONE_RUN/retrained"
  export ULTRAMS_COLLISION_ENERGY_REQUIRE_COMPLETE_TRAIN=1
else
  export ULTRAMS_COLLISION_ENERGY_REQUIRE_COMPLETE_TRAIN=0
fi
export ULTRAMS_COLLISION_ENERGY_OUTPUT_DIR="$BACKBONE_RUN"
export ULTRAMS_CROSS_CE_ULTRA_CHECKPOINT="$BACKBONE_RUN/cross_ce_ultra_v2_ul2_best.pt"
export ULTRAMS_CROSS_CE_DREAMS_CHECKPOINT="$BACKBONE_RUN/cross_ce_dreams_v2_ul2_best.pt"
mkdir -p "$BACKBONE_RUN/logs"

checkpoint_ready() {
  local checkpoint="$1" backbone="$2"
  [[ -s "$checkpoint" ]] || return 1
  if [[ "$WEIGHTS" == published ]]; then return 0; fi
  local summary="$BACKBONE_RUN/cross_ce_finetune_v2.json"
  [[ -s "$summary" ]] || return 1
  python -c 'import json,sys; run=json.load(open(sys.argv[1])).get(sys.argv[2] + "_finetune", {}); sys.exit(0 if run.get("epochs") == 20 and run.get("unfreeze_last") == 2 and len(run.get("history", [])) == 20 else 1)' "$summary" "$backbone"
}

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset msnlib_spectra
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset ultrams_weight
python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset dreams_ssl_weight
if [[ "$WEIGHTS" == published ]]; then
  python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset collision_energy_weights
fi
python "$PUBLIC/prepare_dreams_source.py" --root "$ROOT"
if [[ ! -s "$ROOT/datasets/MSnLib/MSnLib.csv" ]]; then
  python "$PUBLIC/prepare_msnlib.py" --root "$ROOT" --spectra-only
fi

if [[ "$SCOPE" == backbones ]]; then
  for backbone in ultra dreams; do
    checkpoint="$BACKBONE_RUN/cross_ce_${backbone}_v2_ul2_best.pt"
    if ! checkpoint_ready "$checkpoint" "$backbone"; then
      python -u "$PUBLIC/original/train_collision_energy_backbones.py" \
        --backbone "$backbone" --epochs 20 --unfreeze-last 2 --tag ul2 --device "$DEVICE" \
        2>&1 | tee "$BACKBONE_RUN/logs/01_train_${backbone}.log"
    fi
  done
  python -u "$PUBLIC/original/evaluate_collision_energy_backbones.py" --device "$DEVICE" \
    2>&1 | tee "$BACKBONE_RUN/logs/02_evaluate_backbones.log"
  BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
  SOURCE_MANIFEST="$BASE/fig3h_manifest_strict_adduct_v4"
  CROSSCE_MANIFEST="$BASE/crossce_readout_fixed_v1/manifest"
  if [[ ! -f "$SOURCE_MANIFEST/manifest_summary.json" ]]; then
    python -u "$PUBLIC/original/build_library_manifest.py" \
      --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
      --skip-reference-regression --output-dir "$SOURCE_MANIFEST" \
      2>&1 | tee "$BACKBONE_RUN/logs/03_manifest.log"
  fi
  python -u "$PUBLIC/original/prepare_crossce_readout_manifest.py" \
    --manifest-dir "$SOURCE_MANIFEST" --csv "$ROOT/datasets/MSnLib/MSnLib.csv" \
    --output-dir "$CROSSCE_MANIFEST" \
    2>&1 | tee "$BACKBONE_RUN/logs/04_test_queries.log"
  python -u "$PUBLIC/evaluate_collision_energy_test.py" \
    --manifest-dir "$SOURCE_MANIFEST" --crossce-manifest-dir "$CROSSCE_MANIFEST" \
    --backbone-results-dir "$BACKBONE_RUN" --output-dir "$BACKBONE_RUN" \
    2>&1 | tee "$BACKBONE_RUN/logs/05_test_results.log"
  exit 0
fi

python "$PUBLIC/fetch_public_inputs.py" --root "$ROOT" --dataset chemberta
bash "$PUBLIC/run_collision_energy_search.sh"
