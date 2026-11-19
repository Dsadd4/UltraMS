#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRAMS_BENCHMARK_ROOT:-}"
: "${ROOT:?Set ULTRAMS_BENCHMARK_ROOT to the benchmark workspace root}"
export ULTRAMS_BENCHMARK_ROOT="$ROOT"
DEVICE="${ULTRAMS_BENCHMARK_DEVICE:-cuda:0}"
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/original" && pwd)"
BASE="$ROOT/train/output/comparison"
GLOBAL="$BASE/nplib1_ms2fp_contrastive_model_compare/global_fingerprint_cv3"
DREAMS_TOKEN="$BASE/nplib1_token_summary_ms2fp/dreams_peak_fingerprint_cv3"
PROJECTED_TOKEN="$BASE/nplib1_ms2fp_triplet_token_summary/ultrams_projected_peak_fingerprint_cv3"
RAW_PROJECTION="$BASE/nplib1_ms2fp_triplet_token_summary/ultrams_fragment_peak_fingerprint_cv3"
ULTRA_TRIPLET="output/comparison/candidate_ranking/contrastive_candidate_ranking.pt"
RAW_TRIPLET="output/comparison/candidate_ranking/fragment_candidate_ranking.pt"
LOGS="$BASE/nplib1_ms2fp_member_ensemble/logs"
mkdir -p "$LOGS"
cd "$ROOT/train"

python -u "$SCRIPTS/nplib1_contrastive_fingerprint.py" \
  --run-name global_fingerprint_cv3 \
  --ultra-triplet "u5elr8_rank_src=$ULTRA_TRIPLET" \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score tanimoto \
  --epochs 30 --batch-size 512 --candidate-ce-epochs 10 --seed 2026 \
  --device "$DEVICE" 2>&1 | tee "$LOGS/01_global.log"

python -u "$SCRIPTS/nplib1_token_summary_fingerprint.py" \
  --run-name dreams_peak_fingerprint_cv3 --models dreams \
  --summary-parts peak_iw_mean --splits 1 2 3 --scenarios ms2peaks_min50 \
  --score tanimoto --epochs 30 --batch-size 512 \
  --candidate-ce-epochs 10 --candidate-ce-lr 5e-5 \
  --candidate-ce-temperature 24 --seed 2026 --force-emb --device "$DEVICE" \
  2>&1 | tee "$LOGS/02_dreams_peaks.log"

python -u "$SCRIPTS/nplib1_contrastive_token_summary.py" \
  --run-name ultrams_projected_peak_fingerprint_cv3 \
  --ultra-triplet-token "u5elr8_proj_mean_iw=$ULTRA_TRIPLET" \
  --summary-parts peak_mean peak_iw_mean --splits 1 2 3 \
  --scenarios ms2peaks_min50 --score tanimoto --epochs 30 --batch-size 512 \
  --candidate-ce-epochs 10 --seed 2026 --force-emb --device "$DEVICE" \
  2>&1 | tee "$LOGS/03_ultrams_projected_peaks.log"

python -u "$SCRIPTS/nplib1_contrastive_token_summary.py" \
  --run-name ultrams_fragment_peak_fingerprint_cv3 \
  --ultra-triplet-token "hard4_raw_m010_best_projmean=$RAW_TRIPLET" \
  --no-dreams-embedding --summary-parts peak_mean peak_iw_mean \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score tanimoto \
  --epochs 30 --batch-size 512 --candidate-ce-epochs 10 --seed 2026 \
  --force-emb --device "$DEVICE" 2>&1 | tee "$LOGS/04_ultrams_raw_peaks.log"

python -u "$SCRIPTS/nplib1_fingerprint_ensemble.py" \
  --run-name fingerprint_ensemble_cv3 \
  --splits 1 2 3 --scenarios ms2peaks_min50 --score-mode tanimoto \
  --normalize rank --weight-step 0.05 --baseline dreams_global_iw --seed 2026 \
  --ensembles \
  "dreams_global_iw=$GLOBAL::dreams_embedding,$DREAMS_TOKEN::dreams" \
  "u5_cls_projmean_rawproj=$GLOBAL::u5elr8_rank_src,$PROJECTED_TOKEN::u5elr8_proj_mean_iw,$RAW_PROJECTION::hard4_raw_m010_best_projmean" \
  --device "$DEVICE" 2>&1 | tee "$LOGS/05_candidate_ranking.log"
