#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-${LIGHT_ULTRA_ROOT:-}}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export LIGHT_ULTRA_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
BASE="$ROOT/train/output/comparison"
GLOBAL="$BASE/nplib1_ms2fp_contrastive_model_compare/rankfusion_u5elr8_cls_3fold_20260703a"
DREAMS_TOKEN="$BASE/nplib1_token_summary_ms2fp/dreams_peak_iw_3fold_20260703b"
PROJECTED_TOKEN="$BASE/nplib1_ms2fp_triplet_token_summary/rich_token_u5elr8_proj_mean_iw_3fold_20260703d"
RAW_PROJECTION="$BASE/nplib1_ms2fp_triplet_token_summary/hard4_raw_m010_best_projmean_3fold_20260703n"
ULTRA_TRIPLET="output/comparison/dreams_contrastive_ultrams_finetune/u5_fp005_t01_elr8e6_e3_top100_seed3407_20260702e/best.pt"
RAW_TRIPLET="output/comparison/dreams_contrastive_ultrams_finetune/raw_hard4_elr8e6_m010_e3_top100_seed3407_20260703h/best.pt"
LOGS="$BASE/nplib1_ms2fp_member_ensemble/logs"
mkdir -p "$LOGS"
cd "$ROOT/train"

python -u "$SCRIPTS/34_nplib1_ms2fp_contrastive_model_compare.py" \
  --run-name rankfusion_u5elr8_cls_3fold_20260703a \
  --ultra-triplet "u5elr8_rank_src=$ULTRA_TRIPLET" \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score tanimoto \
  --epochs 30 --batch-size 512 --candidate-ce-epochs 10 --seed 2026 \
  --device "$DEVICE" 2>&1 | tee "$LOGS/01_global.log"

python -u "$SCRIPTS/22_nplib1_token_summary_ms2fp.py" \
  --run-name dreams_peak_iw_3fold_20260703b --models dreams \
  --summary-parts peak_iw_mean --splits 1 2 3 --scenarios ms2peaks_min50 \
  --score tanimoto --epochs 30 --batch-size 512 \
  --candidate-ce-epochs 10 --candidate-ce-lr 5e-5 \
  --candidate-ce-temperature 24 --seed 2026 --force-emb --device "$DEVICE" \
  2>&1 | tee "$LOGS/02_dreams_peaks.log"

python -u "$SCRIPTS/36_nplib1_ms2fp_triplet_token_summary.py" \
  --run-name rich_token_u5elr8_proj_mean_iw_3fold_20260703d \
  --ultra-triplet-token "u5elr8_proj_mean_iw=$ULTRA_TRIPLET" \
  --summary-parts peak_mean peak_iw_mean --splits 1 2 3 \
  --scenarios ms2peaks_min50 --score tanimoto --epochs 30 --batch-size 512 \
  --candidate-ce-epochs 10 --seed 2026 --force-emb --device "$DEVICE" \
  2>&1 | tee "$LOGS/03_ultrams_projected_peaks.log"

python -u "$SCRIPTS/36_nplib1_ms2fp_triplet_token_summary.py" \
  --run-name hard4_raw_m010_best_projmean_3fold_20260703n \
  --ultra-triplet-token "hard4_raw_m010_best_projmean=$RAW_TRIPLET" \
  --no-dreams-embedding --summary-parts peak_mean peak_iw_mean \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score tanimoto \
  --epochs 30 --batch-size 512 --candidate-ce-epochs 10 --seed 2026 \
  --force-emb --device "$DEVICE" 2>&1 | tee "$LOGS/04_ultrams_raw_peaks.log"

python -u "$SCRIPTS/38_nplib1_ms2fp_member_ensemble.py" \
  --run-name sourcefusion_existing_u5_proj_compact_rank_3fold_20260703o \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score-mode tanimoto \
  --normalize rank --weight-step 0.05 --baseline dreams_global_iw --seed 2026 \
  --ensembles \
  "dreams_global_iw=$GLOBAL::dreams_embedding,$DREAMS_TOKEN::dreams" \
  "u5_cls_projmean_rawproj=$GLOBAL::u5elr8_rank_src,$PROJECTED_TOKEN::u5elr8_proj_mean_iw,$RAW_PROJECTION::hard4_raw_m010_best_projmean" \
  --device "$DEVICE" 2>&1 | tee "$LOGS/05_candidate_ranking.log"
