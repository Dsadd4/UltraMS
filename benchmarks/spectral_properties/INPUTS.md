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
  --source "$INPUTS/msnlib_spectra/SpecBridge_MSnLib_dataset.mgf" \
  --output "$INPUTS/datasets/MSnLib/MSnLib.csv"
python benchmarks/spectral_properties/prepare_library_spectra.py \
  --dataset spectraverse \
  --source "$INPUTS/spectraverse_spectra/SpecBridge_Spectraverse_dataset.mgf" \
  --output "$INPUTS/datasets/Spectraverse/Spectraverse.csv"
```

This converts the original [MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) table to the column and fold layout consumed by the benchmark programs. It preserves all 231,104 source spectra, their train/validation/test labels, and the float32 numerical values used by the models. The frozen `massspecgym_labels.parquet` is the input for the reported neutral-loss and heteroatom training and tests. `original/prepare_massspecgym.py` can regenerate a task table from the CSV; in our full local regeneration, 3 of 221,010 H2O labels differed because peak selection differed. The MSnLib and SpectraVerse converters use the CSV-producing path of the original SpecBridge MGF converter, preserving its spectrum fields, fold labels, and order.

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
| MSnLib spectrum source | [SpecBridge datasets, Zenodo 18357418](https://zenodo.org/records/18357418), MIT; 1.17 GB MGF | Ion mode, peak neighbors | `msnlib_spectra/SpecBridge_MSnLib_dataset.mgf` → `prepare_library_spectra.py --dataset msnlib` → `datasets/MSnLib/MSnLib.csv` |
| SpectraVerse spectrum source | [SpecBridge datasets, Zenodo 18357418](https://zenodo.org/records/18357418), MIT; 751 MB MGF | Negative-ion structure | `spectraverse_spectra/SpecBridge_Spectraverse_dataset.mgf` → `prepare_library_spectra.py --dataset spectraverse` → `datasets/Spectraverse/Spectraverse.csv` |
| DreaMS implementation | [DreaMS source](https://github.com/pluskal-lab/DreaMS), MIT | All DreaMS comparisons | Checkout commit `dbec3a0b514a99e5056cfccde4559fda8cfe8129` and pass its root via `--dreams-root` |
| MAGMa implementation | [Included benchmark modules](magma_support/NOTICE.md), with MIST's MIT license | Element-bearing peaks, peak neighbors | `magma_support/fragments/magma/`; `run_experiment.py` supplies this path by default |

The MSnLib spectra originate from [Brungs, Schmid and Pluskal's MSnLib release](https://doi.org/10.5281/zenodo.11163380) under CC BY 4.0; the SpecBridge distribution above supplies the exact MGF layout used in this benchmark. Retain both source attributions when redistributing MSnLib-derived files.

The complete public-MGF conversions produce 560,084 MSnLib rows (train 446,667; validation 55,980; test 57,437) and 464,346 SpectraVerse rows (train 370,362; validation 47,825; test 46,159). Both CSVs are byte-identical to the research inputs: SHA-256 `be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07` and `06e084501e91f92cc642c3d60de2863266ff3d0db958354a011e32851bef3976`, respectively. The full peak-neighbor run also produced a 21 MB selected-spectrum table, a 126 MB MAGMa annotation file, a 42 MB peak metadata table and two 688 MB embedding arrays. These are distributed as separate benchmark assets.

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
