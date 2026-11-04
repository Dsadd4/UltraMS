# Public benchmark inputs

The `source_code/fetch_public_inputs.py` downloader resumes partial downloads and checks every file against its published size and checksum. Pass a separate workspace root to keep datasets, weights, caches, and trained outputs outside the code checkout.

## Research environment

On Linux with a CUDA GPU, use a separate Python environment for benchmark training. From the repository root:

```bash
conda create -n ultrams-benchmark -c conda-forge python=3.11 rdkit=2023.09.5 -y
conda activate ultrams-benchmark
python -m pip install -e '.[io]' -r benchmarks/molecular_identification/research_dependencies.txt
```

The full DreaMS comparisons additionally use its official research package:

```bash
export ULTRAMS_BENCHMARK_ROOT="$PWD/ultrams_benchmark_workspace"
python benchmarks/molecular_identification/source_code/prepare_dreams_source.py --root "$ULTRAMS_BENCHMARK_ROOT"
```

The released DreaMS loader imports that pinned source directly. Its model package does not need a separate installation for these benchmarks.

The formal GPU environment used Python 3.11.0, PyTorch 2.2.1, Transformers 4.40.0, NumPy 1.24.4, pandas 2.2.1, PyArrow 15.0.2, RDKit 2023.9.5, SciPy 1.10.1, scikit-learn 1.5.0, h5py 3.11.0, matchms 0.24.2, and PyTorch Lightning 2.0.8. The installation commands are provided for a new research environment. Full public-input runs completed for the reported UltraMS selected projection and newly trained UltraMS and DreaMS projections; their results are in [README.md](README.md).

After installation, this single command downloads the six-method MSnLib inputs, trains and tests all six single-spectrum models, then evaluates multiple-spectrum voting and label-blind grouping:

```bash
bash benchmarks/molecular_identification/source_code/run_public_molecular_identification.sh "$PWD/ultrams_benchmark_workspace"
```

The complete run takes substantial download time, disk space, and CUDA training time. Completed verified datasets and encoder test lists are reused after an interruption.

The six formal single-spectrum test-query lists are also published in [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets) at revision `a974c40d65af90b217589c8a6980cdd114868fed`. Each has 57,437 spectra, top-50 candidate identities, scores, and ranks. To reproduce the multiple-spectrum and label-blind results with their original evaluators, without retraining the six encoders, run:

```bash
bash benchmarks/molecular_identification/source_code/run_public_grouping_evaluation.sh "$PWD/ultrams_benchmark_workspace"
```

The command downloads and verifies the released MSnLib tables and the published test-query lists, builds the retention-time table from an RT-bearing spectrum record (see the label-blind grouping row below), selects the label-blind grouping threshold on validation spectra, then evaluates both grouping tasks on the test fold. All six 57,437-query files passed SHA-256 checks; a clean Python 3.11 run completed 5,002 voting groups and 15,541 label-blind groups covering 52,070 spectra, with 10,000 bootstrap samples and `AUDIT.json` status `complete`. The six neural models' full training rerun remains separate and unfinished.

## MSnLib benchmark partitions

The UltraMS MSnLib benchmark dataset (Hugging Face
[`dsadd4/UltraMS-MSnLib-Benchmark`](https://huggingface.co/datasets/dsadd4/UltraMS-MSnLib-Benchmark);
DOI pending) publishes two partitions of the public
[MSnLib release](https://doi.org/10.5281/zenodo.11163380) (Brungs et al., CC BY 4.0).
They are different objects and are not interchangeable:

| Partition | Split unit | Ratio | Spectra | Used by |
| --- | --- | --- | --- | --- |
| Full-library formula-level split | molecular formula | 80 / 10 / 10 | 446,667 / 55,980 / 57,437 (21,107 formulas in 16,885 / 2,110 / 2,112 groups) | Repository-wide MSnLib tables: candidate ranking, multiple-spectrum voting, label-blind grouping |
| Strict-adduct molecule-level split | canonical isomeric SMILES | 70 / 15 / 15 | `[M+H]+`: 211,745 / 44,881 / 45,010; `[M-H]-`: 98,401 / 20,927 / 20,865 | Figure 3e/3h spectrum-to-spectrum library search (6,202 + 3,193 test queries) |

In the formula-level split every spectrum sharing a molecular formula falls in the
same partition, so no molecular formula, exact SMILES or full InChIKey crosses a
partition. The released `MSnLib.csv` already carries this train/validation/test
assignment in its `fold` column. In the strict-adduct split a molecule lies in exactly
one partition and a query is never part of its own reference library. The two
partitions must not be compared fold-to-fold or used to claim the same test set.

## MSnLib single-spectrum candidate ranking

`source_code/run_public_single_spectrum_ultrams.sh WORKSPACE` is the clean-workspace UltraMS train/test route. It obtains:

| Input | Public source | Workspace location |
| --- | --- | --- |
| MSnLib tables | UltraMS MSnLib benchmark release ([Hugging Face dataset](https://huggingface.co/datasets/dsadd4/UltraMS-MSnLib-Benchmark), DOI pending), derived from the public [MSnLib release](https://doi.org/10.5281/zenodo.11163380) (Brungs et al., CC BY 4.0): `MSnLib.csv` 752,707,651 B, SHA-256 `be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07`; `MSnLib_candidates.json` 1,368,264,026 B, SHA-256 `125a1cdcc57a3fcef8cc76d5322ac0c434cd09b0bfb9e0c9d28bddf052f45fcc` | `datasets/MSnLib/MSnLib.csv`, `datasets/MSnLib/MSnLib_candidates.json` |
| UltraMS unsupervised checkpoint | [UltraMS-Unsupervised](https://huggingface.co/dsadd4/UltraMS-Unsupervised), 834,511,727 B | `train/output/phase2_rt_only/stage_d_epoch_11.pt` |
| ChemBERTa-100M-MLM | [DeepChem/ChemBERTa-100M-MLM](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM), revision `f5c45f44` | `model/feature/ChemBERTa-100M-MLM/` |
| Selected UltraMS spectrum-to-molecule projection | [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), revision `e9fe24af878f29ff1dda8cef919aab5d68d8ea33` | `train/output/comparison/proj_msnlib_mass_rt_only_d11_seed42.pt` |
| Selected DreaMS spectrum-to-molecule projection | [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), revision `171426d7754368f02f5d7ab9bfd5e65b69c17c88` | `train/output/comparison/proj_msnlib_mass_dreams_seed42.pt` |
| Selected Linear, DeepSets, Fourier, and Codebook readouts | [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets/tree/main/molecular_identification/single_spectrum_readouts), revision `b15af172a7b875fedcc3aa85554d097f8d250829` | `benchmark_assets/molecular_identification/single_spectrum_readouts/` |

`prepare_msnlib.py` verifies the two released tables: it checks the published byte sizes and SHA-256 values, reads `MSnLib.csv` and confirms the 560,084 data rows and the fold counts of the UltraMS formula-level 80/10/10 split (446,667 training, 55,980 validation and 57,437 test spectra; 21,107 molecular formulas in 16,885 / 2,110 / 2,112 groups). It does not rebuild or reparse spectra; `--spectra-only` checks the CSV alone, and `--rebuild-from-original` only prints the documented rebuild route from the MSnLib original release (that route is not executed or verified here). The training script saves a newly trained projection and writes the complete test JSONL and Top-k summary to `results/`. To evaluate a selected projection used for the reported result, run `source_code/run_public_single_spectrum_reported.sh WORKSPACE ultrams` or `source_code/run_public_single_spectrum_reported.sh WORKSPACE dreams`; each command downloads its public inputs and uses the original full-fold evaluator.

The four comparison readouts additionally use `prepare_ffn_cache.py`, `prepare_peakset_cache.py`, `extract_ultrams_codebook_pool.py`, `prepare_chemberta_targets.py`, `train_chemberta_readout.py`, and `evaluate_chemberta_readout.py`. The public `run_public_molecular_identification.sh` command above downloads the DreaMS starting checkpoint and runs these steps. `prepare_effective_candidates.py` builds the filtered candidate lists from the original candidate JSON and generated ChemBERTa cache, then checks the two frozen encoder JSONLs.

To download the four selected readout weights directly, run `python benchmarks/molecular_identification/source_code/fetch_public_inputs.py --root WORKSPACE --dataset single_spectrum_readouts`. The DeepSets trainer in `source_code/original/` is the archived manuscript version.

`bash benchmarks/molecular_identification/source_code/run_public_single_spectrum_reported_six_methods.sh WORKSPACE` downloads these selected weights and all other public MSnLib inputs, prepares the required caches, and invokes the original six-method test and grouping evaluators. The grouping command also uses the published `data/label_blind_grouping/cluster_consensus_summary.json` for the paper's upper-bound context table.

## Other identification tasks

| Task | Public raw input and producer | Additional input for the reported result |
| --- | --- | --- |
| MassSpecGym candidate ranking | [Official MassSpecGym HF dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym): TSV (262,334,768 B), mass candidates (454,710,480 B), formula candidates (370,650,823 B). `prepare_massspecgym.py` creates the CSV; `prepare_massspecgym_formula_candidates.py` creates the formula-view fingerprint cache from the trained mass run. | The UltraMS contrastive candidate-ranking checkpoint is available in [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets) at revision `df713440a1e0fec11e24e935d36f8dfc2e601379`; `fetch_public_inputs.py --dataset benchmark_weights` downloads it. The two DreaMS starting checkpoints are downloadable from the [official DreaMS model repository](https://huggingface.co/roman-bushuiev/DreaMS). |
| NPLIB1 candidate ranking | [MIST NPLIB1 source](https://github.com/samgoldman97/mist/blob/main_v2/README.md) and [Zenodo `canopus_train.zip`](https://zenodo.org/records/8151490) provide `labels.tsv` and `spec_files/`. The experiment's three split tables (246,418 / 246,505 / 246,400 B) and candidate table (49,873,428 B) are supplied by [UltraMS Benchmark Assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets). `fetch_public_inputs.py --dataset nplib1` obtains all five files; `prepare_nplib1.py` then creates MS2-block spectra and candidate maps. | The UltraMS contrastive and fragment candidate-ranking checkpoints are supplied by the same dataset; `fetch_public_inputs.py --dataset benchmark_weights` downloads both. |
| Multiple-spectrum voting | Six single-spectrum test JSONLs are produced by the encoder projections and four readouts, then consumed by `evaluate_multiple_spectrum_voting.py`. | Full six-method single-spectrum training inputs. |
| Label-blind grouping | `label_blind_grouping/scripts/run_analysis.py` and `evaluate_frozen_groups.py`. The released `MSnLib.csv` has no retention-time column, so `prepare_msnlib_rt.py` joins `RTINSECONDS` from an RT-bearing spectrum record supplied with `--rt-source PATH` or `ULTRAMS_MSNLIB_RT_SOURCE`; a first-party RT table for the benchmark release is pending. The join reproduces the experiment `MSnLib_with_rt.csv`: 560,084 rows with 446,032 positive RT values (SHA-256 `a5b7c38228287c5c1de401050445319b331ee9a5f17993d6e104e9a734fe2964`). The wrapper runs this converter when needed. | The six test-query files from single-spectrum candidate ranking. |
| Same-adduct library search | `run_public_same_adduct_search.sh WORKSPACE` downloads the released MSnLib table, UltraMS checkpoint, DreaMS SSL checkpoint, and DreaMS source. It trains strict-adduct spectrum-pair encoders for 20 epochs, embeds the complete strict-adduct library, trains four readouts, then tests all six methods with the original FDR evaluator. The frozen UltraMS/DreaMS score summary is also supplied as `data/benchmark_inputs/same_adduct_search_summary.json`. | The release reconstruction matches the historical manifest arrays, spectrum caches, training group counts, and four saved backbone score and FDR curves. Its full new GPU training run is pending. |
| Additional-adduct search | `run_public_additional_adduct_search.sh WORKSPACE` downloads the released MSnLib table, checkpoints, DreaMS source, and ChemBERTa, trains the composite-adduct spectrum-pair encoders, builds their full-library embeddings, and prepares the matched molecule targets and four readouts. The original query identities are supplied as `data/benchmark_inputs/additional_adduct_reference_rows.csv.gz`. | The manifest and spectrum caches match the historical inputs; full public ChemBERTa target preparation is complete with matching identities and near-equal CPU/GPU vectors. A complete new GPU training run is pending. |
| Collision-energy search | `run_public_collision_energy_search.sh WORKSPACE --weights published` downloads the released MSnLib table, UltraMS and DreaMS starting weights, the two [published collision-energy checkpoints](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), and the official DreaMS source; `--weights train` instead trains the two backbones for 20 epochs. Both routes train four comparison readouts and test all six methods. Add `--backbones-only` to run the original backbone evaluator without the four readouts. | The two saved `data/benchmark_inputs/collision_energy_target_keys_{pos,neg}.json` files preserve the molecule-target order and raw-SMILES choices of the reported readout run. They contain only SMILES derived from the UltraMS MSnLib benchmark release (DOI pending), which is derived from the public [MSnLib release](https://doi.org/10.5281/zenodo.11163380) under CC BY 4.0. The public ChemBERTa model encodes those keys. A complete new six-method GPU run is pending. |

The public NPLIB1 preparation was run in full from the official ZIP and the four released split/candidate tables: 10,709 labelled spectra yielded 19,687 MS2 blocks and 18,979 valid blocks. Every fold contains 18,979 block rows and 18,979 block-to-candidate keys; the train/validation/test counts are 15,256/1,765/1,958, 15,475/1,583/1,921, and 15,248/1,750/1,981 for folds 1–3. The prepared fold CSV and candidate JSON files matched the reported training inputs. The preparer checks that blocks of the same molecule receive consistent candidate lists.

The supplied `data/` files allow result recalculation without any of these large inputs. They do not replace a train/test run. For the task-specific historical settings and outputs, see [EXPERIMENTS.md](EXPERIMENTS.md).

The official MassSpecGym TSV and the experiment CSV contain the same 231,104 ordered identities, spectra, SMILES, folds, and other text fields. The three numeric fields `parent_mass`, `precursor_mz`, and `collision_energy` differ only in high-precision text serialization: all 231,104 values in each field are identical after conversion to the float32 inputs used by the models. `prepare_massspecgym.py` saves these input hashes in `MassSpecGym_conversion.json` and checks them against the experiment values.

The historical DreaMS `ssl_model.ckpt` file has a different whole-file checksum from the current official release. Both files have the same 68 model-state keys and every tensor is exactly equal; epoch and global step also agree. `fetch_public_inputs.py --dataset dreams_ssl_weight` obtains the official file.

The DreaMS comparison methods also require the [official DreaMS source](https://github.com/pluskal-lab/DreaMS). Run `python benchmarks/molecular_identification/source_code/prepare_dreams_source.py --root WORKSPACE` to fetch revision `dbec3a0b514a99e5056cfccde4559fda8cfe8129` into the location expected by the released loader. The model implementation is byte-identical to the research copy, and the loader's `SpectrumPreprocessor`, `DataFormat`, and `DataFormatA` classes have identical syntax trees. With the official source and SSL checkpoint, three real MSnLib spectra produced exactly the same preprocessed peak tensors and 1,024-dimensional embeddings as the research source and checkpoint (maximum absolute difference 0.0).
