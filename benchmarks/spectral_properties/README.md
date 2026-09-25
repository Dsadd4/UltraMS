# Spectral properties

This directory contains the Figure 2 evaluation outputs and the code to recalculate their reported measurements. It follows the panels in the manuscript: masked peak reconstruction (a), ion mode recognition (b), negative-ion structural families (c), neutral loss prediction (d–e), element-bearing peak localization (g), heteroatom counts (h), and peak-neighbor chemical fidelity (j). Panels f and i are explanatory visualizations; their source tables are also in `data/`.

```bash
python -m pip install numpy scikit-learn
python benchmarks/spectral_properties/reproduce.py
```

The command writes [`results.csv`](results.csv). Values are proportions (multiply by 100 for percentages), except NMI, ARI, silhouette, (R^2), Tanimoto similarity, and (m/z) error in Da. `n` is the number of masked peaks, spectra, or valid neighbors indicated by each row's `level`.

| Panel | Result in the current figure | Saved evaluation input | Calculation |
| --- | --- | --- | --- |
| a | GeMS-A10 55.2% / 23.8%; MassSpecGym 37.7% / 3.5% (UltraMS / DreaMS) | `reconstruction_*.npz` | Fraction of masked peaks with absolute (m/z) error ≤0.05 Da |
| b | MSnLib ion mode accuracy 94.92% / 88.79% | `ion_mode.json` | Reads the original MLP test evaluation, including class F1 scores |
| c | SpectraVerse NMI 0.4407 / 0.3993, ARI 0.4163 / 0.2958 | `structure_families_test.csv` | Recalculates NMI, ARI, and mean per-spectrum silhouette from test labels |
| d–e | Mean ROC-AUC across the five displayed neutral losses: 94.2% / 82.6% | `neutral_loss_*_{targets,probabilities}.npy` | Recalculates ROC-AUC and AP from 16,440 test predictions per model |
| g | S, Cl, F, Br element-bearing peak hit rate | `element_peak_hit_rate.csv` | Reads the original five-repeat hit-rate curves; Top-5 values are highlighted in the figure |
| h | Test (R^2), UltraMS / DreaMS: S 0.300 / 0.223; Cl 0.188 / 0.123; F 0.118 / 0.079; Br 0.231 / 0.200 | `heteroatom_*_predictions.csv.gz` | Recalculates (R^2) from 16,440 predictions per model |
| j | Nearest-neighbor formula, fragment-family, and structural agreement | `peak_neighbor_fidelity.csv` | Reads the original Top-k search evaluation curves |

The neutral-loss entries in `results.csv` are recalculated for each label directly from the saved 16,440-spectrum test predictions. The five AUC labels printed in the figure average to 94.2% for UltraMS and 82.6% for DreaMS; averaging their unrounded values gives 94.27% and 82.48%.

A separate full training run from the public MSnLib spectra and released starting weights obtained **95.06% for UltraMS** and **88.67% for DreaMS** on the balanced 10,000-spectrum ion-mode test set. It trained the probes on 200,000 spectra, selected on 6,000 validation spectra, and used the released UltraMS Unsupervised checkpoint. The [complete training and test results](data/ion_mode_full_training.json) include linear, MLP, and polarity-head probes; the figure values above remain the original saved evaluation.

The [complete public-input peak-neighbor evaluation](data/peak_neighbor_full_evaluation/peak_neighbor_curves.csv) trains the classical controls on 47,979 peaks and evaluates all five methods on 12,021 peaks from held-out spectra. Its [configuration and full curves](data/peak_neighbor_full_evaluation/peak_neighbor_curves.json) cover all 35 method-by-k results. The saved figure values remain in `data/peak_neighbor_fidelity.csv`.

## Training and testing

[`INPUTS.md`](INPUTS.md) lists the public data and checkpoint downloads with preparation commands. [`EXPERIMENTS.md`](EXPERIMENTS.md) gives training and test commands for every task. [`run_experiment.py`](run_experiment.py) launches the original task programs with explicit input paths. [`reproduce.py`](reproduce.py) recalculates metrics from the saved test outputs. [`original/`](original/) contains the original dataset preparation, task-head training, embedding extraction, and evaluator code. [`magma_support/`](magma_support/) contains the MAGMa annotation modules. The peak-neighbor evaluator exports metrics without rendering a figure.

| Panel | Original experiment scripts |
| --- | --- |
| a | `original/masked_peak_reconstruction/evaluate_masked_peaks.py` |
| b | `original/ion_mode/train_test_ion_mode.py` |
| c | `original/negative_ion_structure/prepare_spectraverse_structure_clusters.py`, `extract_supervised_cls_only.py`, `evaluate_butina_structure_clusters.py`, `formalize_butina_metrics.py` |
| d–e | `original/prepare_massspecgym.py`, `original/neutral_loss/train_test_neutral_loss.py` |
| f | Source table `data/attention_examples.csv` |
| g | `original/element_peak_localization/evaluate_element_peaks.py`, `summarize_element_peaks.py` |
| h | `original/heteroatom_count/train_atom_query.py`, `evaluate_atom_query_on_test.py`, `run_linear_spectrum_baseline.py` |
| i–j | `original/peak_neighbors/prepare_msnlib_magma_subset.py`, `run_magma_msnlib_subset.py`, `extract_peak_embeddings.py`, `peak_neighbor_evaluation.py`; map data `data/peak_embedding_map.csv` |

The source modules imported by these scripts are included under `original/support/`.
