# Molecular identification experiments

The `data/` directory contains the test results used in the manuscript. The two commands below run directly from this repository and regenerate the reported metrics:

```bash
python benchmarks/molecular_identification/reproduce.py
python benchmarks/molecular_identification/recompute_library_search.py
```

The first command needs only Python. The second needs NumPy and recalculates the six methods' same-adduct FDR results from their saved test-query scores. Both check their outputs against the saved experiment tables.

## Training and test code

The scripts under `source_code/original/` are the experiment programs. The task-named `source_code/run_*.sh` commands invoke these public scripts and read datasets, feature caches, and model weights from `ULTRAMS_BENCHMARK_ROOT`, the benchmark workspace. [INPUTS.md](INPUTS.md) gives the public source and preparation route for each task. The compact result files in this repository are test outputs, not substitutes for the train/validation/test datasets.

From a fresh checkout, train a new UltraMS spectrum-to-molecule projection and evaluate the complete MSnLib test fold with:

```bash
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_ultrams.sh "$PWD/ultrams_benchmark_workspace"
```

To evaluate the selected projection used for the reported result, run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported.sh "$PWD/ultrams_single_spectrum_reported"
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported.sh "$PWD/ultrams_single_spectrum_reported" dreams
```

Both commands download and prepare public spectra, candidates, and model weights. A complete public-input test with the selected projection recovered the reported 15,597 Top-1 hits among 57,437 spectra. A full 50-epoch run trained a new projection from the published starting weights and obtained 15,373 Top-1 hits; its [complete evaluator output](data/single_spectrum_full_training/results.json) and [training record](data/single_spectrum_full_training/run.json) are included. The other task runners use the inputs listed in [INPUTS.md](INPUTS.md).

To run all six single-spectrum methods followed by multiple-spectrum voting and label-blind grouping, use the research environment in [INPUTS.md](INPUTS.md) and run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_molecular_identification.sh "$PWD/ultrams_benchmark_workspace"
```

The training command downloads all public inputs, reconstructs the MSnLib and retention-time tables, then runs training and evaluation in that order. The full downloads and neural training require substantial time and GPU memory. It can resume with the completed input files and encoder test lists.

To evaluate the six selected manuscript weights on the full test fold and run both grouping evaluators, use `bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported_six_methods.sh "$PWD/ultrams_selected_models"` in a separate workspace.

For a direct evaluation from the six published top-50 test-query lists, run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_grouping_evaluation.sh "$PWD/ultrams_benchmark_workspace"
```

This uses the same multiple-spectrum and label-blind evaluator programs. The label-blind program rebuilds validation and test groups from the original MSnLib spectra and retention times, chooses the threshold on the validation fold, and then evaluates the six methods on held-out test groups.

The released result tables contain six 57,437-query lists, 5,002 voting groups, and 15,541 label-blind groups covering 52,070 spectra. The grouping evaluator uses 10,000 bootstrap samples.

| Task | Train or prepare | Test | Recalculate from released results |
| --- | --- | --- | --- |
| Single-spectrum candidate ranking | `evaluate_molecule_identification.py` trains the UltraMS and DreaMS spectrum–molecule projections; `prepare_ffn_cache.py`, `prepare_peakset_cache.py`, `extract_ultrams_codebook_pool.py`, `prepare_chemberta_targets.py`, and `train_chemberta_readout.py` prepare and train the four displayed comparison methods | `evaluate_molecule_identification.py --save-per-spectrum` and `evaluate_chemberta_readout.py` write six test-query JSONL files; `evaluate_multiple_spectrum_voting.py` evaluates their ranks as multiple-spectrum voting | `reproduce.py`: `single_spectrum`, `multiple_spectrum_voting` |
| Label-blind spectrum grouping | `label_blind_grouping/scripts/run_analysis.py` selects the grouping threshold on the validation fold | The same program forms test groups; `evaluate_frozen_groups.py` measures grouped identification from the test retrieval outputs | `reproduce.py`: `label_blind_voting` |
| Same-adduct library search | `build_library_manifest.py` fixes the MSnLib split; `prepare_fig3h_readout_inputs.py` prepares strict-adduct features; `train_fig3h_s2s_readout.py` trains the four comparison readouts | `embed_search_fig3h_s2s_readout.py` searches the library; `evaluate_exact_fdr.py` evaluates all six methods, including the frozen UltraMS and DreaMS encoders | `recompute_library_search.py` starts from the released top-1 test scores; `reproduce.py` reads the full FDR frontiers |
| Additional-adduct library search | `build_composite_readout_manifest.py` fixes the composite-polarity training split; `prepare_fig3h_readout_inputs.py` prepares its features; `train_fig3h_readout.py` trains four comparison readouts against ChemBERTa targets | `embed_composite_readout.py` encodes the spectra; `evaluate_other_strict_adduct_six_methods.py` evaluates the additional adducts with these readouts and the frozen UltraMS/DreaMS encoders | `reproduce.py`: `additional_adducts` |
| Low-to-high collision-energy search | `train_collision_energy_backbones.py` fine-tunes UltraMS and DreaMS; `prepare_crossce_readout_manifest.py` fixes low/high-energy spectrum pairs; `train_crossce_readout_fixed.py` trains four comparison readouts for 30 epochs | `evaluate_collision_energy_backbones.py` and `evaluate_crossce_readout.py` test all six methods; `build_collision_energy_rows.py` combines held-out queries | `reproduce.py`: `collision_energy` |
| NPLIB1 fingerprint-based identification | `34_nplib1_ms2fp_contrastive_model_compare.py`, `22_nplib1_token_summary_ms2fp.py`, and `36_nplib1_ms2fp_triplet_token_summary.py` train the four member sources used by the two reported methods | `38_nplib1_ms2fp_member_ensemble.py` selects fusion on validation queries and evaluates all three test folds | `reproduce_fingerprint_prediction.py` calculates Top-k from all saved test queries; `data/nplib1_metrics.csv` preserves the original summary |
| MassSpecGym fingerprint-based identification | `39_massspecgym_ms2fp_sourcefusion.py` trains the four fingerprint sources on the mass task; `prepare_massspecgym_formula_candidates.py` reuses these heads with formula candidates | `finish_massspecgym_candidate_ranking.py` selects the two reported fusions on validation queries and evaluates mass and formula candidates separately | `reproduce_fingerprint_prediction.py` calculates Top-k from all saved test queries; the two `massspecgym_*_metrics.csv` files preserve the original summaries |

The single-spectrum program contains both projection training and candidate testing. The grouping program selects a validation threshold and then groups test spectra; it does not train a neural model. Multiple-spectrum voting consumes the single-spectrum retrieval output. The same-adduct controls use same-molecule spectrum pairs and symmetric InfoNCE for 20 epochs. The released additional-adduct results use separately trained composite-polarity readouts against ChemBERTa targets. The low-to-high-energy search uses the strict-adduct source spectra but its own low/high-energy training pairs.

`prepare_chemberta_targets.py` extracts model targets from a supplied `target_smiles.json` list and ChemBERTa cache for the single-spectrum candidate-ranking controls.

## Running the original programs

From this repository, after preparing the MSnLib data, candidates, encoders, and feature caches under an arbitrary benchmark workspace:

```bash
export ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
bash benchmarks/molecular_identification/source_code/run_single_spectrum_candidate_ranking.sh
bash benchmarks/molecular_identification/source_code/run_multiple_spectrum_identification.sh
bash benchmarks/molecular_identification/source_code/run_label_blind_grouping.sh
```

The first script trains the two spectrum–molecule projections and four ChemBERTa readouts, then writes six held-out per-spectrum candidate lists. It uses the frozen MSnLib folds, ChemBERTa molecule cache, and effective candidate list; the two projection encoders save their top-50 candidates with `--save-per-spectrum`. The second script runs multiple-spectrum voting. The third selects a label-blind grouping threshold on validation spectra and evaluates the frozen test groups for the same six methods. Its retention-time input is generated from the public SpecBridge MGF by `prepare_msnlib_rt.py` when absent.

For the learned readouts, the programs accept explicit input and output directories. For example, one strict-adduct comparison run is:

```bash
SCRIPTS="$(pwd)/benchmarks/molecular_identification/source_code/original"
python -u "$SCRIPTS/train_fig3h_s2s_readout.py" \
  --manifest-dir "$STRICT_ADDUCT_MANIFEST" --cache-dir "$FEATURE_CACHE" \
  --output-dir "$READOUT_MODELS" --model linear --mode pos --seed 0

python -u "$SCRIPTS/embed_search_fig3h_s2s_readout.py" \
  --manifest-dir "$STRICT_ADDUCT_MANIFEST" --cache-dir "$FEATURE_CACHE" \
  --model-dir "$READOUT_MODELS" --output-dir "$LIBRARY_SEARCH" \
  --model linear --mode pos --seed 0
```

Repeat those two commands for `linear`, `deepsets`, `fourier_projection`, and `ultrams_codebook`, each in `pos` and `neg` mode. The original `evaluate_exact_fdr.py` takes each model/mode test score as `--input 'method|mode|test|seed|path/to/top1.npz'` and the manifest summary as `--manifest-summary`. The released `recompute_library_search.py` executes that metric calculation directly on the included twelve test-score files.

### Library-search encoders

`source_code/train_library_search.py` retains the UltraMS/DreaMS spectrum-pair fine-tuning and test calculations without plotting. It reads the downloaded MSnLib CSV and starting checkpoints in the benchmark workspace and imports the UltraMS model source from this repository. The following command uses the composite-adduct scope for additional-adduct search:

```bash
ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
for backbone in ultra dreams; do
  python -u benchmarks/molecular_identification/source_code/train_library_search.py \
    --project-root "$ULTRAMS_BENCHMARK_ROOT" --backbone "$backbone" --device cuda:0
done
```

This preserves the original 70/15/15 compound split (seed 42), symmetric InfoNCE, 20 epochs, test retrieval, and the full-library embeddings under `$ULTRAMS_BENCHMARK_ROOT/train/comparison/output`. Those composite-scope embeddings are inputs to the additional-adduct evaluator. The run writes `lib_search_finetune.json` and `lib_search_eval_{ultra,dreams}.json` as well as checkpoints and embeddings.

### Same-adduct library search

The released six-method scores use the strict-adduct manifest and four 256-dimensional spectrum-pair readouts. The strict-adduct UltraMS/DreaMS encoder training uses only `[M+H]+` and `[M-H]-`; unlike the composite task above, its full-library extraction retains singleton compounds. From public inputs, run training, embedding, six-method search, and exact-FDR evaluation with:

```bash
bash benchmarks/molecular_identification/source_code/run_public_same_adduct_search.sh "$PWD/ultrams_benchmark_workspace"
```

The public strict-adduct training and embedding programs reconstruct the reported 20-epoch, batch-256, last-two-layer run from its saved configuration and query/library manifest; the original strict-adduct embedding script is not available in this release. The published query scores and original FDR evaluator are included. To train the four controls and recalculate their exact-FDR test results against the published UltraMS/DreaMS scores, run:

```bash
export ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
bash benchmarks/molecular_identification/source_code/run_same_adduct_search.sh
```

The script uses this sequence for each `linear`, `deepsets`, `fourier_projection`, or `ultrams_codebook` model and each `pos` or `neg` mode:

```bash
ROOT=/path/to/benchmark_workspace
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
SCRIPTS="$(pwd)/benchmarks/molecular_identification/source_code/original"
MANIFEST="$BASE/fig3h_manifest_strict_adduct_v4"
INPUTS="$BASE/fig3h_readout_reuse_v1/inputs"
RUN="$BASE/fig3h_s2s_readout_v1"
model=linear
mode=pos

python -u "$SCRIPTS/train_fig3h_s2s_readout.py" \
  --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --output-dir "$RUN/models" \
  --model "$model" --mode "$mode" --seed 0 --device cuda:0
python -u "$SCRIPTS/embed_search_fig3h_s2s_readout.py" \
  --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --model-dir "$RUN/models" \
  --output-dir "$RUN/search" --model "$model" --mode "$mode" --seed 0 --device cuda:0
```

Run all eight model/mode combinations, then pass their `val/top1.npz` and `test/top1.npz` paths to `evaluate_exact_fdr.py` using its `--input` format. The manifest producer is `build_library_manifest.py`; `prepare_fig3h_readout_inputs.py --skip-molecule-targets` creates the spectrum-pair features from MSnLib. The released `strict_adduct_scores/` and `strict_adduct_manifest_summary.json` are sufficient to rerun the published FDR evaluation with `recompute_library_search.py`.

### Additional-adduct library search

The released per-query table matches the **composite-polarity ChemBERTa readout** run. Its training is distinct from the same-adduct spectrum-pair controls. To download public inputs, train the encoders and controls, and evaluate all six methods, run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_additional_adduct_search.sh "$PWD/ultrams_benchmark_workspace"
```

With inputs already prepared, rerun the readouts and evaluator with:

```bash
export ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
bash benchmarks/molecular_identification/source_code/run_additional_adduct_search.sh
```

The script performs this train-and-embed sequence:

```bash
ROOT=/path/to/benchmark_workspace
BASE="$ROOT/train/output/comparison/fig3_simple_baselines_20260821"
SCRIPTS="$(pwd)/benchmarks/molecular_identification/source_code/original"
MANIFEST="$BASE/fig3h_manifest_composite_adduct_v1"
INPUTS="$BASE/fig3h_composite_readout_v1/inputs"
RUN="$BASE/fig3h_composite_readout_v1"
model=linear
mode=pos

python -u "$SCRIPTS/train_fig3h_readout.py" \
  --manifest-dir "$MANIFEST" --cache-dir "$INPUTS" --output-dir "$RUN/models" \
  --model "$model" --mode "$mode" --seed 0 --epochs 30 --device cuda:0
python -u "$SCRIPTS/embed_composite_readout.py" \
  --cache-dir "$INPUTS" --model-dir "$RUN/models" --output-dir "$RUN/embeddings" \
  --model "$model" --mode "$mode" --seed 0 --device cuda:0
```

Repeat for the four models and both polarities, then evaluate all six methods:

```bash
python -u "$SCRIPTS/evaluate_other_strict_adduct_six_methods.py" \
  --csv "$ROOT/datasets/MSnLib/MSnLib.csv" --manifest-dir "$MANIFEST" \
  --embedding-dir "$ROOT/train/comparison/output" \
  --readout-embedding-dir "$RUN/embeddings" \
  --reference-task-rows "benchmarks/molecular_identification/data/benchmark_inputs/additional_adduct_reference_rows.csv.gz" \
  --output-dir "$BASE/fig3h_other_strict_adduct_v3_six_methods" --device cuda:0
```

`build_composite_readout_manifest.py` and `prepare_fig3h_readout_inputs.py` prepare `MANIFEST` and `INPUTS`. The evaluator writes the per-query, exact-FDR frontier, and summary CSV files used by `reproduce.py`.

### Low-to-high collision-energy search

The released comparison uses fine-tuned UltraMS/DreaMS backbones and the fixed 30-epoch readout run. To use the published backbone checkpoints, or to train the backbones for 20 epochs from the released starting weights, run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_collision_energy_search.sh "$PWD/ultrams_benchmark_workspace" --weights published
bash benchmarks/molecular_identification/source_code/run_public_collision_energy_search.sh "$PWD/ultrams_benchmark_workspace" --weights train
```

Both commands derive low/high-energy pairs, train the four comparison methods in both polarities, and combine all six held-out search results. Retrained checkpoints are written under `train/output/showcase/retrained/`, separate from the downloaded checkpoints. Add `--backbones-only` to run `evaluate_collision_energy_backbones.py` on the full pool, then select test queries with `evaluate_collision_energy_test.py`. The latter writes Top-1 results and per-query hits under the backbone output directory. The six-method result uses the test split selected by `prepare_crossce_readout_manifest.py` and `build_collision_energy_rows.py`.

### NPLIB1 and MassSpecGym fingerprint-based identification

The NPLIB1 command trains the four source representations used by the two reported methods, selects ensemble weights on validation queries, and evaluates three held-out folds. The MassSpecGym command trains its four fingerprint heads on mass-restricted candidates, completes the mass test, reuses those trained heads with formula-restricted candidates, and completes the formula test:

```bash
export ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
bash benchmarks/molecular_identification/source_code/run_nplib1_candidate_ranking.sh
bash benchmarks/molecular_identification/source_code/run_massspecgym_candidate_ranking.sh
```

NPLIB1 uses `ms2peaks_min50`, splits 1–3, and validation-selected rank fusion. MassSpecGym uses the official `mass` and `formula` candidate files under `datasets/MassSpecGym/`. The historical formula view was a saved summary and fingerprint-cache interface; `prepare_massspecgym_formula_candidates.py` constructs that interface from the mass source run and the formula candidates. The final MassSpecGym metrics come from `finish_massspecgym_candidate_ranking.py`.

`source_code/original/` contains the task entry points, their local helpers, and metric evaluators; `training/ultrams_training/model/` supplies the UltraMS model code. Full neural training also needs the original dataset files, feature caches, starting weights, and the scientific Python dependencies. The released result-recalculation commands above need neither a GPU nor those training inputs.
