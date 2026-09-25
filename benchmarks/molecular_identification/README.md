# Molecular identification

[Install the research dependencies](INPUTS.md#research-environment), then run all six single-spectrum methods, multiple-spectrum voting, and label-blind grouping from public inputs:

```bash
bash benchmarks/molecular_identification/source_code/run_public_molecular_identification.sh "$PWD/ultrams_benchmark_workspace"
```

This downloads the original MSnLib spectra and candidate table plus model weights, trains the two spectrum-to-molecule projections and four comparison readouts, writes the six held-out top-50 candidate lists, and runs both grouping evaluations. The downloads and full training take substantial time and GPU memory. Completed inputs and encoder test lists are reused when the command is restarted.

To run the six selected manuscript models over the original MSnLib test fold, including multiple-spectrum voting and label-blind grouping:

```bash
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported_six_methods.sh "$PWD/ultrams_selected_models"
```

This command downloads the published encoder, projection, and readout weights, prepares the public spectra and molecule features, and runs the original evaluators. The reported per-query outputs are included in `data/`.

To evaluate the selected UltraMS and DreaMS projections used for the reported single-spectrum result, using public spectra, candidates, encoder weights, and projection weights:

```bash
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported.sh "$PWD/ultrams_single_spectrum_reported"
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported.sh "$PWD/ultrams_single_spectrum_reported" dreams
```

The original evaluator processes all 57,437 MSnLib test spectra. A full public-input run recovered 15,597 Top-1 hits (27.15497%), 19,988 Top-5 hits (34.79987%), and 25,007 Top-20 hits (43.53814%). All test-query identities, Top-1 molecules, and Top-1/5/20/50 hit decisions matched the released test-query list. Its [validation and test results](data/single_spectrum_reported_evaluation/results.json) are included. Use a separate workspace for this selected-projection evaluation and fresh projection training.

To rerun multiple-spectrum voting and label-blind grouping directly from the six published test-query lists, use:

```bash
bash benchmarks/molecular_identification/source_code/run_public_grouping_evaluation.sh "$PWD/ultrams_benchmark_workspace"
```

This downloads the official MSnLib spectra and six reported 57,437-query top-50 lists, reconstructs retention times, selects the grouping threshold on validation spectra, and executes the original held-out evaluators.

[Public inputs and a clean-workspace training route](INPUTS.md) cover the MSnLib UltraMS single-spectrum projection. On Linux with a CUDA GPU, from the repository root:

```bash
python -m pip install -e '.[io]' pandas pyarrow rdkit pyyaml tqdm
bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_ultrams.sh "$PWD/ultrams_benchmark_workspace"
```

That command downloads the MSnLib MGF and candidate table, the UltraMS checkpoint, and ChemBERTa; converts the dataset; trains the projection; and evaluates the complete 57,437-spectrum test fold. Results are written to `ultrams_benchmark_workspace/results/`. The [four selected comparison readout weights](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/tree/main/molecular_identification/single_spectrum_readouts/reported) and the [reported DeepSets training record](data/single_spectrum_deepsets_reported/) are also public.

## Fingerprint-based molecular identification

UltraMS and DreaMS spectrum representations are used to predict 2,048-bit, radius-2 Morgan molecular fingerprints. The predicted fingerprints rank the candidate molecules in NPLIB1 and MassSpecGym. To download only the public datasets:

```bash
python benchmarks/molecular_identification/source_code/fetch_public_inputs.py --root "$PWD/ultrams_benchmark_workspace" --dataset nplib1
python benchmarks/molecular_identification/source_code/fetch_public_inputs.py --root "$PWD/ultrams_benchmark_workspace" --dataset massspecgym
```

To download the remaining starting weights, train the fingerprint predictors, and evaluate the test sets, use the [research environment](INPUTS.md#research-environment) and run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_nplib1_candidate_ranking.sh "$PWD/ultrams_benchmark_workspace"
bash benchmarks/molecular_identification/source_code/run_public_massspecgym_candidate_ranking.sh "$PWD/ultrams_benchmark_workspace"
```

The [training scripts](source_code/original/) and [three result tables](data/) are included. To calculate Top-1, Top-5, and Top-20 directly from the saved test-query results:

```bash
python benchmarks/molecular_identification/reproduce_fingerprint_prediction.py
```

The saved query results are [NPLIB1](data/fingerprint_prediction/nplib1_test_queries.csv.gz), [MassSpecGym mass](data/fingerprint_prediction/massspecgym_mass_test_queries.csv.gz), and [MassSpecGym formula](data/fingerprint_prediction/massspecgym_formula_test_queries.csv.gz).

For spectrum-library search by the same or an additional adduct, run the corresponding complete public-input route:

```bash
bash benchmarks/molecular_identification/source_code/run_public_same_adduct_search.sh "$PWD/ultrams_benchmark_workspace"
bash benchmarks/molecular_identification/source_code/run_public_additional_adduct_search.sh "$PWD/ultrams_benchmark_workspace"
```

Each command trains the two spectrum-pair encoders and four comparison readouts, searches the held-out libraries, and runs the task evaluator. The reported strict-adduct scores can also be recalculated with `recompute_library_search.py`.

For low-to-high collision-energy search, use the published fine-tuned backbones or train them for 20 epochs from the released starting weights:

```bash
bash benchmarks/molecular_identification/source_code/run_public_collision_energy_search.sh "$PWD/ultrams_benchmark_workspace" --weights published
bash benchmarks/molecular_identification/source_code/run_public_collision_energy_search.sh "$PWD/ultrams_benchmark_workspace" --weights train
```

Both routes train the four comparison readouts and run the six-method held-out evaluator. Add `--backbones-only` to evaluate the two spectrum-pair backbones on the held-out queries; their Top-1 results are saved in `train/output/showcase/collision_energy_backbone_test.json` (under `retrained/` for `--weights train`).

[Training, testing, and result-recalculation programs](EXPERIMENTS.md) are listed by task.

Run the results reported in the current manuscript's six-method Figure 3:

```bash
python benchmarks/molecular_identification/reproduce.py
```

This command reads the saved evaluation outputs in `data/`, recalculates the displayed metrics, checks them against the original result tables, and writes `results/results.json`. It uses only the Python standard library. To recalculate the same-adduct FDR curves directly from saved query scores with the original evaluator:

```bash
python -m pip install numpy
python benchmarks/molecular_identification/recompute_library_search.py
```

That writes `results/library_search_from_scores.json` and checks all 12 model and ion-mode results against the original working points.

## Tasks and results

| Panel | Task | UltraMS | DreaMS | Evaluation output |
| --- | --- | ---: | ---: | --- |
| 3a | Single-spectrum identification, Top-1, 57,437 spectra | 27.15% | 20.25% | `data/single_spectrum_query_ranks.csv.gz` |
| 3a | Multiple-spectrum voting, Top-1, 5,002 molecules | 42.56% | 33.49% | `data/multiple_spectrum_voting_per_group.csv.gz` |
| 3b | Score-margin distribution and Top-k for the 3a identification task | Top-20 43.54% single; 64.87% voting | Top-20 35.35% single; 57.74% voting | 3a ranks and `data/score_margin_density.csv` |
| 3d | Label-blind multiple-spectrum identification, Top-1 on 52,070 grouped spectra | 27.36% → 29.25% | 20.51% → 22.03% | `data/label_blind_per_query.csv.gz` |
| 3e | Same-adduct spectrum library search, maximum recall for `[M+H]+` / `[M-H]-` | 97.13% / 95.70% | 95.44% / 93.73% | `data/strict_adduct_scores/` |
| 3e | Additional-adduct search, mean micro recall over empirical FDR 0–5%, positive / negative | 17.58% / 35.71% | 10.83% / 32.38% | `data/other_strict_adduct_per_query.csv.gz` |
| 3f | Low-to-high collision-energy search, Recall@1, positive / negative | 94.66% / 92.40% | 88.56% / 86.30% | `data/collision_energy_queries.csv.gz` |

For panel 3d, the paired Top-1 gain is **1.88 pp [1.51, 2.25]** for UltraMS and **1.52 pp [1.16, 1.87]** for DreaMS. In same-molecule groups, the gains are **2.59 / 2.15 pp**; in mixed-molecule groups, they are **−14.38 / −13.00 pp** (UltraMS / DreaMS). `reproduce.py` recalculates these group values from the saved query results and checks them against `data/model_performance.csv`.

Figure 3c is the 8-methoxyquercetin 7-`O`-β-D-glucopyranoside identification example; its case, ranks, and supporting peaks are in `data/`.

The manuscript reports the following fingerprint-based molecular identification results. Each cell gives Top-1 / Top-5 / Top-20 accuracy:

| Task | UltraMS | DreaMS |
| --- | ---: | ---: |
| NPLIB1, three-fold evaluation | 32.98% / 57.91% / 80.54% | 27.93% / 54.64% / 79.10% |
| MassSpecGym, mass-restricted candidates | 17.09% / 33.35% / 54.89% | 13.24% / 26.27% / 45.68% |
| MassSpecGym, formula-restricted candidates | 14.08% / 29.03% / 51.68% | 11.41% / 23.95% / 43.40% |

The result table is `data/additional_molecule_identification.csv`. Published external-method results in that table retain their own source and protocol labels.

The same-adduct maximum recall in the table is the unthresholded endpoint. `results/results.json` also reports the recall at an empirical FDR of at most 5% and the mean recall across the 0–5% FDR range. These are separate readouts of the same test results.

For the additional adducts, the strongest other method in the current six-method figure reaches **14.25%** for positive ions (Fourier) and **32.38%** for negative ions (DreaMS) over the 0–5% FDR range.

## Experimental code

`source_code/original/` retains the experiment programs. The corresponding task entry points are:

```bash
export ULTRAMS_BENCHMARK_ROOT=/path/to/benchmark_workspace
bash benchmarks/molecular_identification/source_code/run_single_spectrum_candidate_ranking.sh
bash benchmarks/molecular_identification/source_code/run_multiple_spectrum_identification.sh
bash benchmarks/molecular_identification/source_code/run_label_blind_grouping.sh
bash benchmarks/molecular_identification/source_code/run_same_adduct_search.sh
bash benchmarks/molecular_identification/source_code/run_additional_adduct_search.sh
bash benchmarks/molecular_identification/source_code/run_collision_energy_search.sh
bash benchmarks/molecular_identification/source_code/run_nplib1_candidate_ranking.sh
bash benchmarks/molecular_identification/source_code/run_massspecgym_candidate_ranking.sh
```

`ULTRAMS_BENCHMARK_ROOT` points to a benchmark workspace with the task inputs and generated outputs. [INPUTS.md](INPUTS.md) gives the public download and preparation commands. The training, test, and result-reproduction steps for each task are listed in [EXPERIMENTS.md](EXPERIMENTS.md).

| Task | Source entry points |
| --- | --- |
| Single-spectrum identification and multiple-spectrum voting | `evaluate_molecule_identification.py`, `train_chemberta_readout.py`, `evaluate_chemberta_readout.py`, `evaluate_multiple_spectrum_voting.py` |
| Label-blind grouping and identification | `label_blind_grouping/scripts/run_analysis.py`, `evaluate_frozen_groups.py` |
| Same-adduct library search | `build_library_manifest.py`, `prepare_fig3h_readout_inputs.py`, `train_fig3h_s2s_readout.py`, `embed_search_fig3h_s2s_readout.py`, `evaluate_exact_fdr.py` |
| Additional-adduct library search | `build_composite_readout_manifest.py`, `train_fig3h_readout.py`, `embed_composite_readout.py`, `evaluate_other_strict_adduct_six_methods.py` |
| Low-to-high collision-energy search | `train_collision_energy_backbones.py`, `evaluate_collision_energy_backbones.py`, `prepare_crossce_readout_manifest.py`, `train_crossce_readout_fixed.py`, `evaluate_crossce_readout.py` |
| NPLIB1 candidate identification | `34_nplib1_ms2fp_contrastive_model_compare.py`, `22_nplib1_token_summary_ms2fp.py`, `36_nplib1_ms2fp_triplet_token_summary.py`, `38_nplib1_ms2fp_member_ensemble.py` |
| MassSpecGym candidate identification | `39_massspecgym_ms2fp_sourcefusion.py`, `prepare_massspecgym_formula_candidates.py`, `finish_massspecgym_candidate_ranking.py` |

The two result-recalculation commands at the top run from the included test outputs on a laptop. For a new train/test run, the public task commands download the MSnLib, NPLIB1, or MassSpecGym data and starting weights, then prepare the feature caches under `ULTRAMS_BENCHMARK_ROOT`. Follow [Public inputs](INPUTS.md) and [Training and evaluation](EXPERIMENTS.md); no internal project files are needed. `source_code/extract_single_spectrum_ranks.py` and `source_code/build_collision_energy_rows.py` record how the single-spectrum and collision-energy query tables were extracted from the completed experiments. The main figure uses these six methods; later seven-method exploratory charts are kept separate.

The label-blind grouping run uses the original settings in `source_code/original/label_blind_grouping/config/analysis_config.json`. With the original MSnLib inputs and saved retrieval results available under an experiment directory, run:

```bash
python -m pip install numpy pandas scipy
python benchmarks/molecular_identification/source_code/original/label_blind_grouping/scripts/run_analysis.py \
  --project-root /path/to/benchmark_workspace --output-dir /path/to/grouping_results
```

The released test assignments, validation threshold table, and `data/grouping_summary.csv` match that producer's output: 15,541 groups, 52,070 grouped spectra, and 98.10% spectrum-weighted purity at the validation-selected similarity threshold of 0.5. The result replay checks the saved assignments against the six methods' grouped-query results.
