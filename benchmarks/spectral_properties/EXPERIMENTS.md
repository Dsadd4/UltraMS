# Training, testing, and result reproduction

Run `python benchmarks/spectral_properties/reproduce.py` from the repository root to recalculate the published measurements from the saved test outputs. The scripts below are the experiment producers. They use the original research data and pretrained comparison models; their inputs are not part of the small `ultrams` installation.

| Task | Training | Testing and saved output |
| --- | --- | --- |
| Masked peak reconstruction | The pretrained encoders are evaluated without a task-specific head. | `original/masked_peak_reconstruction/evaluate_masked_peaks.py` evaluates MassSpecGym and GeMS-A10 and saves raw reconstruction arrays. `reproduce.py` computes the displayed 0.05 Da accuracy from those arrays. The source uses the v9 epoch-3 reconstruction checkpoint. |
| Ion mode recognition | `original/ion_mode/train_test_ion_mode.py --protocol in_domain_msnlib --models dreams rt_only_d --probes both --seed 42` fits the probes on the MSnLib training split and selects on validation. | The same run evaluates the balanced MSnLib test split and writes probe metrics. The supplied `data/ion_mode.json` is the original evaluation record. |
| Negative-ion structural families | No structure-family classifier is trained. `original/negative_ion_structure/prepare_spectraverse_structure_clusters.py` freezes the SpectraVerse split; `extract_supervised_cls_only.py` extracts the two pretrained representations. | `evaluate_butina_structure_clusters.py` and `formalize_butina_metrics.py` evaluate the held-out spectra. The three helper modules they import are included alongside them. |
| Neutral loss prediction | `original/prepare_massspecgym.py` creates the seven-label parquet. `original/neutral_loss/train_test_neutral_loss.py --backbone ultra` and `--backbone dreams` fit the task heads on the MassSpecGym training split and select on validation. | Each training run tests its selected head and saves probabilities and targets. `reproduce.py` recalculates each loss's ROC-AUC and average precision from the saved 16,440-row predictions. |
| Element-bearing peak localization | The atom-query heads are trained by the heteroatom-count experiment below. | `original/element_peak_localization/evaluate_element_peaks.py` runs five repeats of 100 test spectra against MAGMa annotations; `summarize_element_peaks.py` gathers the Top-k hit rates. |
| Heteroatom counts | `original/heteroatom_count/train_atom_query.py --backbone ultra --epochs 15 --seed 42` and `--backbone dreams --epochs 15 --seed 42` fit the atom-query heads. `run_linear_spectrum_baseline.py` fits the spectrum-feature controls. | `evaluate_atom_query_on_test.py` evaluates the supplied probe checkpoints on the MassSpecGym test split and records the actual epoch of each. The saved figure test used UltraMS epoch 13 and DreaMS epoch 1 checkpoints. `reproduce.py` recalculates the reported R² from saved predictions. |
| Peak-neighbor chemical fidelity | `original/peak_neighbors/prepare_msnlib_magma_subset.py`, `run_magma_msnlib_subset.py`, and `extract_peak_embeddings.py` prepare the MSnLib/MAGMa peak representations. `peak_neighbor_evaluation.py` fits its classical controls on spectrum-disjoint training peaks. | `peak_neighbor_evaluation.py` searches neighbors for held-out spectra and saves the Top-k metrics as CSV and JSON. `reproduce.py` reads the published evaluation curves. |

The peak-neighbor evaluator is a computation-only version of the original experiment; it contains no figure rendering. It requires `pandas`, `numpy`, `scikit-learn`, `rdkit`, and `faiss-cpu`. Given the extracted peak files, run:

```bash
python benchmarks/spectral_properties/original/peak_neighbors/peak_neighbor_evaluation.py \
  --data-dir /path/to/extracted_peak_data \
  --subset-csv /path/to/msnlib_magma_subset.csv \
  --output-dir /path/to/peak_neighbor_results
```

The source producers retain the training project's original imports. [`INPUTS.md`](INPUTS.md) gives the public downloads and dataset preparation commands for MassSpecGym, MSnLib, SpectraVerse, MoNA exclusion, and the released model checkpoints. The benchmark's MAGMa annotation modules are included in `magma_support/`. The supplied test outputs and `reproduce.py` run independently of these inputs.

## Run the original experiments

Install the research dependencies alongside the minimal UltraMS package:

```bash
conda create -n ultrams-bench python=3.11 --yes
conda activate ultrams-bench
python -m pip install -e . -r benchmarks/spectral_properties/research_dependencies.txt
export INPUTS=/path/to/spectral_property_inputs
mkdir -p "$INPUTS"
git clone https://github.com/pluskal-lab/DreaMS.git "$INPUTS/DreaMS"
git -C "$INPUTS/DreaMS" checkout dbec3a0b514a99e5056cfccde4559fda8cfe8129
python -m pip check
export EXPERIMENT=$INPUTS
export RUN=$PWD/spectral_property_runs
export STAGE_D=$INPUTS/ultrams_unsupervised/model.pt
export DREAMS_ROOT=$INPUTS/DreaMS
export ULTRAMS_DREAMS_CKPT=$INPUTS/dreams_ssl/ssl_model.ckpt
export ULTRAGO=$PWD/benchmarks/spectral_properties/magma_support
export SP=benchmarks/spectral_properties/run_experiment.py
mkdir -p "$RUN"
```

`run_experiment.py` starts the original program with the required source modules on `PYTHONPATH`; the program's remaining options follow `--`. It runs under `--run-dir`, so relative output paths land there. The released three UltraMS checkpoints are starting models; the task heads and supervised comparison checkpoints are separate experimental outputs. The frozen MassSpecGym training table is downloaded by `download_public_inputs.py --asset massspecgym_labels`.

### Masked peak reconstruction

The producer requests 5,000 MassSpecGym or GeMS-A10 spectra and masks 15% of peaks. Its archived outputs contain 4,187 and 4,999 valid spectra, respectively. The displayed hit definition is absolute m/z error at most 0.05 Da.

```bash
export RECONSTRUCTION=$INPUTS/benchmark_assets/spectral_properties/masked_peak_reconstruction.pt
export GEMS=$INPUTS/gems/data/spectra/GeMS_A10.hdf5
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/reconstruction" \
  --reconstruction-checkpoint "$RECONSTRUCTION" --dreams-root "$DREAMS_ROOT" \
  --massgym-csv "$EXPERIMENT/datasets/MassSpecGym/MassSpecGym.csv" \
  masked_peak_reconstruction -- --data massgym --n-samples 5000 --mask-ratio 0.15
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/reconstruction" \
  --reconstruction-checkpoint "$RECONSTRUCTION" --gems-hdf5 "$GEMS" \
  --dreams-root "$DREAMS_ROOT" masked_peak_reconstruction -- \
  --data gems --n-samples 5000 --mask-ratio 0.15
```

The benchmark asset contains the original v9 epoch-3 reconstruction model's tensors; optimizer and internal-path metadata have been removed. The stage-D checkpoint is a different model stage.
The evaluator selects CUDA, MPS, or CPU automatically; `--device` can override it.

A full rerun of both commands with the public inputs retained 4,187 MassSpecGym and 4,999 GeMS-A10 spectra. On GeMS-A10, the 0.05 Da hit counts exactly matched the saved results: UltraMS 26,609/48,191 and DreaMS 7,189/30,189. On MassSpecGym, the new counts were UltraMS 7,643/20,259 and DreaMS 541/15,786; the saved counts are 7,644/20,259 and 546/15,786. The MassSpecGym raw-array comparison found different selected peaks in 13 UltraMS and 23 DreaMS spectra. The saved test arrays remain the source of the displayed values.

### Ion mode recognition

One program extracts embeddings, fits probes on the MSnLib training fold, selects on validation, and evaluates the balanced test fold.

```bash
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/ion_mode" \
  --msnlib-csv "$EXPERIMENT/datasets/MSnLib/MSnLib.csv" --dreams-root "$DREAMS_ROOT" \
  --stage-d-checkpoint "$STAGE_D" ion_mode -- \
  --protocol in_domain_msnlib --models dreams rt_only_d --probes both --seed 42
```

The saved 94.92% result identifies `rt_only_d` but does not record the checkpoint epoch or exact command. A complete run of the command above using the released Unsupervised checkpoint and official DreaMS SSL checkpoint selected the MLP probes on 6,000 validation spectra and obtained 95.06% UltraMS and 88.67% DreaMS accuracy on 10,000 test spectra. Its [training and test output](data/ion_mode_full_training.json) is separate from the saved figure evaluation.

### Negative-ion structural families

The SpectraVerse benchmark excludes MoNA molecules. The two supervised checkpoints come from molecular-identification training. The selected six-family protocol uses a 0.75 fingerprint-distance cutoff, followed by held-out test metrics. The downloaded frozen structure inputs retain the exact cluster assignments used for the reported test.

```bash
export SV=$EXPERIMENT/datasets/Spectraverse/Spectraverse.csv
export MONA="$INPUTS/mona_exclusion/data/auxiliary/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl"
export ULTRA_SUPERVISED=$INPUTS/ultrams_mona/model.pt
export DREAMS_SUPERVISED=$INPUTS/dreams_embedding/embedding_model.ckpt
export STRUCTURE_INPUTS=$INPUTS/benchmark_assets/spectral_properties/negative_ion_structure
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/structure" \
  --stage-d-checkpoint "$STAGE_D" --dreams-root "$DREAMS_ROOT" \
  negative_ion_structure_extract -- --root "$EXPERIMENT" \
  --benchmark-dir "$STRUCTURE_INPUTS" --out-dir "$RUN/structure/embeddings" \
  --ultra "$ULTRA_SUPERVISED" --dreams "$DREAMS_SUPERVISED" --input-peaks 150
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/structure" \
  negative_ion_structure_select -- --benchmark-dir "$STRUCTURE_INPUTS" \
  --embedding-dir "$RUN/structure/embeddings" --out-dir "$RUN/structure/selected" \
  --distance-cutoff 0.75 --top-clusters 6 --input-peaks 150
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/structure" \
  negative_ion_structure_test -- --benchmark-dir "$STRUCTURE_INPUTS" \
  --embedding-dir "$RUN/structure/embeddings" --locked-dir "$RUN/structure/selected" \
  --out-dir "$RUN/structure/test" --input-peaks 150 --bootstrap 10000
```

To regenerate the structure inputs from the public SpectraVerse and MoNA files, run `negative_ion_structure_prepare -- --csv "$SV" --mona "$MONA" --out-dir "$RUN/structure/input_regenerated"`. The frozen inputs above carry the cluster assignments used for the reported test; a newer scikit-learn/RDKit environment produced the same 1,562 validation and 1,592 test structures with slightly different cluster assignments.

A clean Python 3.11 run extracted all 1,562 validation and 1,592 test spectra with both encoders, selected 627 and 661 spectra, and completed the original 10,000-resample test. All 18 point estimates and confidence intervals matched the saved results within 2.24 × 10⁻⁸; every per-spectrum cluster label matched exactly.

### Neutral loss prediction

The frozen MassSpecGym table contains the seven neutral-loss labels and train/validation/test folds used in the research runs. Each model trains for 15 epochs with batch size 256 and seed 42, then evaluates its selected validation checkpoint on test.
The completed 15-epoch training histories and test metrics are included in [`data/training_histories/`](data/training_histories/).

```bash
export MASSGYM=$EXPERIMENT/datasets/MassSpecGym/MassSpecGym.csv
export PARQUET=$INPUTS/benchmark_assets/spectral_properties/massspecgym_labels.parquet
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/neutral_loss" \
  --massgym-parquet "$PARQUET" --stage-d-checkpoint "$STAGE_D" \
  neutral_loss_train_and_test -- --backbone ultra --epochs 15 --batch-size 256 \
  --seed 42 --out-dir "$RUN/neutral_loss/ultrams"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/neutral_loss" \
  --massgym-parquet "$PARQUET" --dreams-root "$DREAMS_ROOT" \
  neutral_loss_train_and_test -- --backbone dreams --epochs 15 --batch-size 256 \
  --seed 42 --out-dir "$RUN/neutral_loss/dreams"
```

`prepare_massspecgym` regenerates a table from the public MassSpecGym CSV if desired. The frozen table above is the input for the reported benchmark runs; the current public-source conversion and producer differ in three H2O labels because their top-150 peak selections differ.

### Heteroatom counts and element-bearing peak localization

The MassSpecGym atom-query heads learn S, Cl, F and Br counts. Both original training jobs completed 15 epochs. The same heads provide attention scores for five repeats of 100 spectra in the element-bearing peak test.
Their 15-epoch training histories are included in [`data/training_histories/`](data/training_histories/).

```bash
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/heteroatoms" \
  --massgym-parquet "$PARQUET" --stage-d-checkpoint "$STAGE_D" \
  heteroatom_count_train -- --backbone ultra --atoms S Cl F Br --epochs 15 \
  --batch-size 512 --seed 42 --out-dir "$RUN/heteroatoms/ultrams"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/heteroatoms" \
  --massgym-parquet "$PARQUET" --dreams-root "$DREAMS_ROOT" \
  heteroatom_count_train -- --backbone dreams --atoms S Cl F Br --epochs 15 \
  --batch-size 512 --seed 42 --out-dir "$RUN/heteroatoms/dreams"
export ULTRA_PROBE=$RUN/heteroatoms/ultrams/best_ultra.pt
export DREAMS_PROBE=$RUN/heteroatoms/dreams/best_dreams.pt
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/heteroatoms" \
  --dreams-root "$DREAMS_ROOT" \
  heteroatom_count_test -- --parquet "$PARQUET" --ultra-probe "$ULTRA_PROBE" \
  --dreams-probe "$DREAMS_PROBE" --ultra-backbone "$STAGE_D" \
  --output-dir "$RUN/heteroatoms/test"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/heteroatoms" \
  heteroatom_count_classical -- --parquet "$PARQUET" \
  --output-dir "$RUN/heteroatoms/classical" --evaluation-fold test
```

The reported element-bearing peak test used the selected UltraMS epoch 13 and DreaMS epoch 1 probes. Download those exact heads to evaluate the saved model state:

```bash
python benchmarks/spectral_properties/download_public_inputs.py \
  --output-root "$INPUTS" --asset atom_query_heads
export ULTRA_PROBE=$INPUTS/benchmark_assets/spectral_properties/heteroatom_count/ultrams_probe.pt
export DREAMS_PROBE=$INPUTS/benchmark_assets/spectral_properties/heteroatom_count/dreams_probe.pt
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/heteroatoms" \
  --dreams-root "$DREAMS_ROOT" \
  heteroatom_count_test -- --parquet "$PARQUET" --ultra-probe "$ULTRA_PROBE" \
  --dreams-probe "$DREAMS_PROBE" --ultra-backbone "$STAGE_D" \
  --output-dir "$RUN/heteroatoms/selected_test"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/element_peaks" \
  --ultrago-root "$ULTRAGO" --dreams-root "$DREAMS_ROOT" \
  element_peak_localization_test -- --all-csv "$MASSGYM" \
  --ultra-ckpt "$ULTRA_PROBE" --dreams-ckpt "$DREAMS_PROBE" \
  --ultra-backbone-ckpt "$STAGE_D" --out-dir "$RUN/element_peaks" \
  --repeats 5 --samples-per-repeat 100 --seed 20260530 --without-xgboost
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$RUN/element_peaks" \
  element_peak_localization_summary -- --out-dir "$RUN/element_peaks"
```

The included MAGMa modules produce fragment annotations at element-peak evaluation time. The formal test compares UltraMS, DreaMS, Linear SVM, and Random forest; the original saved run had no XGBoost installed. The summary command writes the five-repeat element hit-rate and peak-recall curves for k=5–150 as CSV, alongside cosine, precision, and capture summaries. The saved five-repeat measurements are in `data/`.
A full public-input rerun selected the same 470 distinct test spectra and reproduced the MAGMa annotation cache byte-for-byte. Its repeat and summary results are in [`data/element_peak_localization_rerun/`](data/element_peak_localization_rerun/). All UltraMS and DreaMS hit-rate and recall values at k=5–150 matched the saved curves; Linear SVM hit rates also matched. Random-forest hit rates differed by at most 0.02792 in proportion across the curve, with the largest difference at Br k=15. The Top-5 UltraMS hit rates for S, F, Cl, and Br were 0.86069, 0.95637, 0.93918, and 0.92729 in both runs.
The classical controls were rerun on all 186,511 training and 16,440 test spectra in the released frozen table. All four random-forest test R² values matched the saved results; the four LinearSVR values differed by at most 0.0000209.

### Peak-neighbor chemical fidelity

The MSnLib/MAGMa producer builds the x10 subset, annotates peaks, and extracts the two encoders' peak embeddings. The final program fits spectrum-feature controls on training peaks and evaluates nearest neighbors on spectra held out by its 20% split.

To rerun the published training and held-out test from the frozen MSnLib-derived peak representations, first download the seven original files as shown in [`INPUTS.md`](INPUTS.md), then run:

```bash
export PEAK_INPUTS=$INPUTS/benchmark_assets/spectral_properties/peak_neighbors
python benchmarks/spectral_properties/original/peak_neighbors/peak_neighbor_evaluation.py \
  --data-dir "$PEAK_INPUTS" \
  --subset-csv "$PEAK_INPUTS/msnlib_magma_subset.csv" \
  --output-dir "$RUN/peak_neighbors_from_frozen_inputs" \
  --max-points 60000 --test-size 0.2 --seed 42 --nn-method faiss_hnsw \
  --k-values 1,2,5,10,20,50,100
```

The following commands additionally rebuild MAGMa annotations and both peak representation arrays from the public MSnLib CSV and the included MAGMa modules:

```bash
export PEAKS=$RUN/peak_neighbors
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$PEAKS" \
  peak_neighbors_prepare -- --msnlib-csv "$EXPERIMENT/datasets/MSnLib/MSnLib.csv" \
  --n-spectra 10000 --min-peaks 18 --max-peaks 130 --seed 42 \
  --output "$PEAKS/msnlib_magma_subset.csv"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$PEAKS" \
  peak_neighbors_annotate -- --input "$PEAKS/msnlib_magma_subset.csv" \
  --output "$PEAKS/msnlib_magma_annotations.json" --ultrago-root "$ULTRAGO" --ppm 10
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$PEAKS" \
  --stage-d-checkpoint "$STAGE_D" --dreams-root "$DREAMS_ROOT" \
  peak_neighbors_extract -- --subset-csv "$PEAKS/msnlib_magma_subset.csv" \
  --annotations "$PEAKS/msnlib_magma_annotations.json" --output-dir "$PEAKS" \
  --ultra-ckpt "$STAGE_D"
python "$SP" --experiment-root "$EXPERIMENT" --run-dir "$PEAKS" \
  peak_neighbors_train_and_test -- --data-dir "$PEAKS" \
  --subset-csv "$PEAKS/msnlib_magma_subset.csv" --output-dir "$PEAKS/results" \
  --max-points 60000 --test-size 0.2 --seed 42 --nn-method faiss_hnsw \
  --k-values 1,2,5,10,20,50,100
```

The downloadable x10 inputs are `msnlib_magma_subset.csv` (21 MB), `msnlib_magma_annotations.json` (126 MB), `peak_embedding_metadata.csv` (42 MB), and `peak_embeddings_{ultra,dreams}.npy` (688 MB each). The saved test curves in `data/peak_neighbor_fidelity.csv` remain directly available for result recalculation.
The complete public-input run's [35-row result table](data/peak_neighbor_full_evaluation/peak_neighbor_curves.csv) and [training/test configuration and curves](data/peak_neighbor_full_evaluation/peak_neighbor_curves.json) are also included. All 35 method-by-k combinations and their valid-neighbor counts match the saved test table. Formula-hit rates differ by at most 0.00416 across the five methods.

### Recalculate the published results

```bash
python benchmarks/spectral_properties/reproduce.py
```

This recalculates `results.csv` from the supplied test outputs. It does not rerun representation extraction, probe training, MAGMa, or held-out neighbor search.
