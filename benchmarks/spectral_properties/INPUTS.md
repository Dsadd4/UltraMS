# Public inputs for spectral property experiments

The published `results.csv` can be recalculated from this repository's saved test outputs. Training and testing from spectra use the files below. The downloader fetches individual public sources; it records exact source revisions and verifies the downloaded bytes. Downloads resume through the Hugging Face Hub cache or a Zenodo `.part` file.

```bash
python -m pip install huggingface-hub pyteomics pandas numpy
export INPUTS=/path/to/spectral_property_inputs
python benchmarks/spectral_properties/download_public_inputs.py --list
python benchmarks/spectral_properties/download_public_inputs.py \
  --output-root "$INPUTS" \
  --asset massspecgym gems massspecgym_labels masked_peak_reconstruction \
          negative_ion_structure atom_query_heads peak_neighbors \
          msnlib_spectra spectraverse_spectra \
          mona_exclusion ultrams_unsupervised ultrams_mona \
          dreams_ssl dreams_embedding
python benchmarks/spectral_properties/prepare_massspecgym_csv.py \
  --source "$INPUTS/massspecgym/data/MassSpecGym.tsv" \
  --output "$INPUTS/datasets/MassSpecGym/MassSpecGym.csv"
python benchmarks/spectral_properties/prepare_library_spectra.py \
  --dataset msnlib \
  --source "$INPUTS/msnlib_spectra/MSnLib.csv" \
  --output "$INPUTS/datasets/MSnLib/MSnLib.csv"
python benchmarks/spectral_properties/prepare_library_spectra.py \
  --dataset spectraverse \
  --source "$INPUTS/spectraverse_spectra/Spectraverse_dataset.mgf" \
  --output "$INPUTS/datasets/Spectraverse/Spectraverse.csv"
```

This converts the original [MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) table to the column and fold layout consumed by the benchmark programs. It preserves all 231,104 source spectra, their train/validation/test labels, and the float32 numerical values used by the models. The frozen `massspecgym_labels.parquet` is the input for the reported neutral-loss and heteroatom training and tests. `original/prepare_massspecgym.py` can regenerate a task table from the CSV; in our full local regeneration, 3 of 221,010 H2O labels differed because peak selection differed. For the library records, `prepare_library_spectra.py --dataset msnlib` verifies the downloaded released `MSnLib.csv` (published size and SHA-256, 560,084 rows, formula-level fold counts) and places it at the output path without reparsing spectra; `--dataset spectraverse` converts the legacy SpectraVerse-derived MGF record to the same CSV layout (peak arrays, SMILES, precursor m/z, adduct, collision energy and partition label).

| Input | Public source and license | Used by | Local output or derivation |
| --- | --- | --- | --- |
| MassSpecGym spectra | [MassSpecGym dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym), MIT | Reconstruction, element-bearing peaks, task-table regeneration | `massspecgym/data/MassSpecGym.tsv` → `prepare_massspecgym_csv.py` → `datasets/MassSpecGym/MassSpecGym.csv` |
| Frozen MassSpecGym task table | [UltraMS benchmark assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), derived from MassSpecGym | Neutral loss and heteroatom training/testing | `benchmark_assets/spectral_properties/massspecgym_labels.parquet` |
| Selected heteroatom-count probes | [UltraMS benchmark assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), trained on MassSpecGym | Element-bearing peak localization and heteroatom-count test | `benchmark_assets/spectral_properties/heteroatom_count/{ultrams_probe.pt,dreams_probe.pt}`; selected UltraMS epoch 13 and DreaMS epoch 1 heads, with backbones downloaded separately |
| Frozen negative-ion structure split | [UltraMS benchmark assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), derived from SpectraVerse | Negative-ion structure evaluation | `benchmark_assets/spectral_properties/negative_ion_structure/`; seven files preserve the selected validation/test families |
| Masked peak reconstruction weights | [UltraMS benchmark assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets), Apache-2.0 | Masked peak reconstruction | `benchmark_assets/spectral_properties/masked_peak_reconstruction.pt`; original v9 epoch-3 model tensors retained |
| GeMS-A10 spectra | [MassSpecGym dataset, GeMS_A10.hdf5](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main/data/spectra), MIT; 14.63 GB | Masked peak reconstruction | `gems/data/spectra/GeMS_A10.hdf5` |
| MoNA exclusion table | [GeMS dataset](https://huggingface.co/datasets/roman-bushuiev/GeMS), MIT; 81 MB | Negative-ion structure split | `mona_exclusion/data/auxiliary/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl`; `original/negative_ion_structure/prepare_spectraverse_structure_clusters.py` consumes its `inchi14` column |
| UltraMS Unsupervised | [UltraMS model](https://huggingface.co/dsadd4/UltraMS-Unsupervised), Apache-2.0 | Ion mode, neutral loss, heteroatom counts, element-bearing peaks, peak neighbors | `ultrams_unsupervised/model.pt`; exact same bytes as the stage-D epoch-11 research checkpoint |
| UltraMS MoNA contrastive | [UltraMS model](https://huggingface.co/dsadd4/UltraMS-MoNA-Contrastive), Apache-2.0 | Negative-ion structure representations | `ultrams_mona/model.pt`; exact same bytes as the supervised checkpoint recorded for that test |
| DreaMS SSL and embedding weights | [DreaMS model](https://huggingface.co/roman-bushuiev/DreaMS), model card: MIT | DreaMS comparisons | `dreams_ssl/ssl_model.ckpt` and `dreams_embedding/embedding_model.ckpt`; the embedding file matches the research copy byte-for-byte, and all 68 SSL model tensors match despite checkpoint metadata-byte differences |
| MSnLib spectrum table | UltraMS MSnLib benchmark release ([Hugging Face dataset](https://huggingface.co/datasets/dsadd4/UltraMS-MSnLib-Benchmark), DOI pending), derived from the public [MSnLib release](https://doi.org/10.5281/zenodo.11163380) (Brungs et al., CC BY 4.0): `MSnLib.csv`, 752,707,651 B, SHA-256 `be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07`, 560,084 rows with the UltraMS formula-level 80/10/10 split in the `fold` column | Ion mode, peak neighbors | `msnlib_spectra/MSnLib.csv` → `prepare_library_spectra.py --dataset msnlib` → `datasets/MSnLib/MSnLib.csv` |
| SpectraVerse spectrum source | Legacy SpectraVerse-derived distribution ([Zenodo record 23963489](https://zenodo.org/records/23963489)), MIT; 751 MB MGF. Moving the frozen SpectraVerse split to a first-party release is pending | Negative-ion structure | `spectraverse_spectra/Spectraverse_dataset.mgf` → `prepare_library_spectra.py --dataset spectraverse` → `datasets/Spectraverse/Spectraverse.csv` |
| DreaMS implementation | [DreaMS source](https://github.com/pluskal-lab/DreaMS), MIT | All DreaMS comparisons | Checkout commit `dbec3a0b514a99e5056cfccde4559fda8cfe8129` and pass its root via `--dreams-root` |
| MAGMa implementation | [Included benchmark modules](magma_support/NOTICE.md), with MIST's MIT license | Element-bearing peaks, peak neighbors | `magma_support/fragments/magma/`; `run_experiment.py` supplies this path by default |

The MSnLib spectra originate from the public [MSnLib release](https://doi.org/10.5281/zenodo.11163380) by Brungs, Schmid and Pluskal (CC BY 4.0). The UltraMS MSnLib benchmark release redistributes the converted `MSnLib.csv` table together with the UltraMS formula-level 80/10/10 train/validation/test split; retain the MSnLib attribution when redistributing MSnLib-derived files. The SpectraVerse spectra currently come from the legacy SpectraVerse-derived record listed above; their frozen split will move to a first-party release (DOI pending) once its file layout and attribution record are verified.

The formula-level split assigns the 21,107 molecular formulas of MSnLib to 16,885 / 2,110 / 2,112 disjoint groups, giving the 446,667 / 55,980 / 57,437 spectra of the train / validation / test partitions; the released `MSnLib.csv` already carries this assignment in its `fold` column. The strict-adduct library-search tasks (Figure 3e/3h) use a separate molecule-level 70/15/15 split of the `[M+H]+` / `[M-H]-` subsets; see [`../molecular_identification/INPUTS.md`](../molecular_identification/INPUTS.md#msnlib-benchmark-partitions) for both partitions.

The MSnLib table is the released `MSnLib.csv` (560,084 rows: train 446,667; validation 55,980; test 57,437; the UltraMS formula-level 80/10/10 split, SHA-256 `be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07`). The SpectraVerse MGF conversion produces 464,346 rows (train 370,362; validation 47,825; test 46,159; the legacy SpectraVerse-derived split) and is byte-identical to the research input: SHA-256 `06e084501e91f92cc642c3d60de2863266ff3d0db958354a011e32851bef3976`. The full peak-neighbor run also produced a 21 MB selected-spectrum table, a 126 MB MAGMa annotation file, a 42 MB peak metadata table and two 688 MB embedding arrays. These are distributed as separate benchmark assets.

## Peak-neighbor benchmark from the frozen representations

The benchmark assets contain `msnlib_magma_subset.csv`, `msnlib_magma_annotations.json`, `peak_embedding_metadata.csv`, `peak_embeddings_ultra.npy`, and `peak_embeddings_dreams.npy` under `spectral_properties/peak_neighbors/`. These are derived from the public MSnLib spectra. The annotations cover 10,000 selected spectra; the two representation arrays align with all 176,138 rows in the metadata. Download them with:

```bash
python benchmarks/spectral_properties/download_public_inputs.py \
  --output-root "$INPUTS" --asset peak_neighbors
```

The complete neighbor baseline training and held-out evaluation then runs without MAGMa or backbone inference:

```bash
export PEAK_INPUTS=$INPUTS/benchmark_assets/spectral_properties/peak_neighbors
python benchmarks/spectral_properties/original/peak_neighbors/peak_neighbor_evaluation.py \
  --data-dir "$PEAK_INPUTS" \
  --subset-csv "$PEAK_INPUTS/msnlib_magma_subset.csv" \
  --output-dir "$INPUTS/peak_neighbor_results" \
  --max-points 60000 --test-size 0.2 --seed 42 \
  --nn-method faiss_hnsw --k-values 1,2,5,10,20,50,100
```

This fits the classical baselines on the training spectra and searches held-out spectra for every model. In a complete local rerun, all 35 model-by-k keys and valid-neighbor counts matched the saved output. Against its numeric values, UltraMS/DreaMS formula-hit differences were at most 0.00025 in proportion. Across all baselines, maximum differences were 0.00416 for formula hit, 0.00549 for fragment-family hit, 0.00195 for fingerprint similarity, 0.656 Da for mean m/z error, and 0.967 Da for median m/z error; the largest differences came from random forest neighbors. FAISS-HNSW and library versions can change individual neighbor choices.

## Inputs by experiment

| Task | Download with `--asset` | Starting path or generated input |
| --- | --- | --- |
| Masked peak reconstruction | `massspecgym gems masked_peak_reconstruction dreams_ssl` | `datasets/MassSpecGym/MassSpecGym.csv`, `gems/data/spectra/GeMS_A10.hdf5`, `benchmark_assets/spectral_properties/masked_peak_reconstruction.pt` |
| Ion mode | `msnlib_spectra ultrams_unsupervised dreams_ssl` | `datasets/MSnLib/MSnLib.csv`, `ultrams_unsupervised/model.pt` |
| Negative-ion structure | `spectraverse_spectra mona_exclusion negative_ion_structure ultrams_mona ultrams_unsupervised dreams_ssl dreams_embedding` | `benchmark_assets/spectral_properties/negative_ion_structure/`, `ultrams_mona/model.pt`, `dreams_embedding/embedding_model.ckpt` |
| Neutral loss and heteroatom count | `massspecgym_labels ultrams_unsupervised dreams_ssl` | `benchmark_assets/spectral_properties/massspecgym_labels.parquet` and the two backbone files |
| Element-bearing peak localization | `massspecgym massspecgym_labels ultrams_unsupervised dreams_ssl atom_query_heads` | `datasets/MassSpecGym/MassSpecGym.csv`, selected heads under `benchmark_assets/spectral_properties/heteroatom_count/`, included `magma_support/` |
| Peak-neighbor chemical fidelity | `peak_neighbors` | `benchmark_assets/spectral_properties/peak_neighbors/`; the saved representations run complete probe training and held-out testing |

The exact training and testing commands for each task are in [`EXPERIMENTS.md`](EXPERIMENTS.md). The archived ion-mode percentage identifies `rt_only_d` but does not record the checkpoint epoch; the public command explicitly uses the released Unsupervised checkpoint.
