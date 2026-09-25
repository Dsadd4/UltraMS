#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
PUBLIC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$PUBLIC/original"
BASE="$ROOT/train/output/comparison/massspecgym_ms2fp_sourcefusion"
SOURCE=sourcefusion_mass_all_fair_dp20_wd3e4_e15_ce2_b2048_none_w005_20260704ae
FORMULA_VIEW=sourcefusion_formula_view_from_ae_20260705
LOGS="$BASE/benchmark_logs"
mkdir -p "$LOGS"
cd "$ROOT/train"

python -u "$SCRIPTS/39_massspecgym_ms2fp_sourcefusion.py" \
  --run-name "$SOURCE" --tasks mass --scenarios all \
  --sources dreams_embedding dreams_peak_iw u5elr8_rank_src u5elr8_proj_mean_iw \
  --ensembles dreams_global_iw u5_cls_projmean --score tanimoto \
  --epochs 15 --batch-size 2048 --lr 5e-4 --weight-decay 3e-4 --dropout 0.20 \
  --candidate-ce-epochs 2 --candidate-ce-lr 5e-5 --candidate-ce-temperature 24 \
  --normalize none --weight-step 0.05 --seed 2026 --device "$DEVICE" \
  2>&1 | tee "$LOGS/01_train_fingerprint_sources.log"

python -u "$SCRIPTS/finish_massspecgym_candidate_ranking.py" \
  --source-run "$SOURCE" \
  --run-name sourcefusion_mass_all_fair_dp20_score_tanimoto_rank_w005_20260704af \
  --task mass --scenario all --ensembles dreams_global_iw u5_cls_projmean \
  --score tanimoto --normalize rank --weight-step 0.05 --device "$DEVICE" \
  2>&1 | tee "$LOGS/02_evaluate_mass_candidates.log"

python -u "$PUBLIC/prepare_massspecgym_formula_candidates.py" \
  --source-run "$SOURCE" --run-name "$FORMULA_VIEW" \
  2>&1 | tee "$LOGS/03_prepare_formula_candidates.log"

python -u "$SCRIPTS/finish_massspecgym_candidate_ranking.py" \
  --source-run "$FORMULA_VIEW" \
  --run-name sourcefusion_formula_all_fair_dp20_score_tanimoto_rank_aeview_w005_20260705bj \
  --task formula --scenario all --ensembles dreams_global_iw u5_cls_projmean \
  --score tanimoto --normalize rank --weight-step 0.05 --hidden 2048 \
  --device "$DEVICE" 2>&1 | tee "$LOGS/04_evaluate_formula_candidates.log"
