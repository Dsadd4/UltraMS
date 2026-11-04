# Figure 2 — plotting code and source data

This directory contains the plotting scripts, the plotted source data and the selection-time
rendered panel images for **Figure 2** of the UltraMS manuscript (panels **a–j**).

The data-generation code for these panels lives in
[`benchmarks/spectral_properties/`](../../benchmarks/spectral_properties/README.md); this package
contains the *figure-facing* layer, i.e. the code and the small numeric tables that turn the
benchmark outputs into the published panels.

```
paper_figures/figure2/
├── README.md                 (this file)
├── panel_manifest.json       per-file ledger: panel, source path, target path, bytes, sha256, note
├── .gitignore                ignores large regenerated inputs
├── scripts/<panel>/          plotting scripts, file names preserved
├── data/<panel>/             small source-data files read by the scripts (CSV / JSON / SVG)
└── outputs/<panel>/          rendered panel images from the selected figure snapshot (PNG / PDF / SVG)
```

Panel directories use the **manuscript panel letter** (`a` … `j`). The selection identifiers used
by the internal figure-selection manifest are listed per panel in the table and in
`panel_manifest.json`.

## Panel guide

| Manuscript panel | Content | Plotting scripts (this package) | Source data (this package) | Data-generation code in `benchmarks/spectral_properties/` | Re-render / reproduce |
| --- | --- | --- | --- | --- | --- |
| **a** | Masked-peak m/z reconstruction accuracy (UltraMS vs DreaMS) on GeMS-A10 and MassSpecGym; hit = \|Δ*m/z*\| ≤ 0.05 Da. Also includes the registered mean-absolute-error variant (`fig2a_mae`). | `scripts/a/plot_task1.py` | — (the reported numbers are set in the script) | `original/masked_peak_reconstruction/evaluate_masked_peaks.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py` (accuracy rows); `python scripts/a/plot_task1.py` to redraw |
| **b** | Ionization-polarity recognition: test F1 for [M−H]⁻ and [M+H]⁺ with a frozen-embedding MLP probe. | `scripts/b/plot_task4.py`, `scripts/b/plot_ion_polarity_embedding_umap.py` | `data/b/ion_polarity/polarity_linear_probe_combined.json`; `data/b/ion_polarity_embedding/*.csv`, `*.json` | `original/ion_mode/train_test_ion_mode.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/b/plot_task4.py` (reads `data/b/ion_polarity/`) |
| **c** | Negative-ion structure-family organization: DreaMS vs UltraMS UMAP of six model-independent RDKit Butina families (661 SpectraVerse test molecules), NMI / ARI / silhouette bars, and the database-index panel. | `scripts/c/render_compact_v2.py`, `scripts/c/render_db.py`, `scripts/c/compute_db.py`, `scripts/c/original_feature_definition.py` | `data/c/source_data/` — frozen cluster metrics, DB summary and per-family tables, paired bootstrap, frozen UMAP coordinates, molecule SVGs | `original/negative_ion_structure/prepare_spectraverse_structure_clusters.py`, `extract_supervised_cls_only.py`, `evaluate_butina_structure_clusters.py`, `formalize_butina_metrics.py` | `python scripts/c/render_compact_v2.py` re-renders the compact panel from `data/c/source_data/`; `reproduce.py` recomputes the NMI/ARI/silhouette rows |
| **d** | Neutral-loss detection: average-precision scatter (UltraMS vs DreaMS) for five losses. | `scripts/d/plot_task2.py` | `data/d/nl_ultra_0414_1826/config.json`, `data/d/nl_dreams_0414_1828/config.json` | `original/prepare_massspecgym.py`; `original/neutral_loss/train_test_neutral_loss.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/d/plot_task2.py` (reads the `nl_*` run configurations) |
| **e** | Neutral-loss detection: ROC small multiples for H₂O (−18 Da), NH₃ (−17 Da), CO₂ (−44 Da), CO (−28 Da) and CH₃ (−15 Da). | `scripts/e/plot_task2_roc.py` | `data/e/nl_ultra_0414_1826/roc_data.json`, `data/e/nl_dreams_0414_1828/roc_data.json` | `original/prepare_massspecgym.py`; `original/neutral_loss/train_test_neutral_loss.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/e/plot_task2_roc.py` |
| **f** | MAGMa fragment-attention evidence cases for S-, F-, Cl- and Br-bearing molecules: attention spectrum, highlighted structure and the top-4 annotated fragments. | `scripts/f/atom_attention_viz_magma_v16_mixed_top4_row4.py`, `scripts/f/atom_attention_viz_magma_v15_mixed_top4_grid_full_fragments.py`, `scripts/f/select_br_candidates_v12.py` | `data/f/*.plot_data.json`, `*.summary.json`, `*.peak_table.csv` | `magma_support/` (MAGMa annotation modules); atom-query heads from `original/heteroatom_count/`; source table `data/attention_examples.csv` | `python scripts/f/atom_attention_viz_magma_v16_mixed_top4_row4.py` re-renders the case grid from `data/f/` |
| **g** | Element-bearing peak hit rate versus Top-*k* peaks for S, F, Cl and Br (four methods, five repeats). | `scripts/g/plot_hit_recall_curves_v1.py`, `scripts/g/run_attn_subspectrum_eval_v1.py`, `scripts/g/plot_precision_curves_v1.py`, `scripts/g/postprocess_repeat_summary_v1.py` | `data/g/attn_subspectrum_*` (hit-rate and precision curves, per-peak scores, summary metrics) | `original/element_peak_localization/evaluate_element_peaks.py`, `summarize_element_peaks.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/g/plot_hit_recall_curves_v1.py` |
| **h** | Heteroatom-count regression: test *R*² for S, Cl, F and Br (UltraMS, DreaMS, Linear SVM, Random forest). | `scripts/h/fig_atomq_bar.py` | `data/h/atomq_ultra_S_Cl_F_Br_0413_1448/config.json`, `data/h/atomq_dreams_S_Cl_F_Br_0413_2123/config.json` | `original/heteroatom_count/train_atom_query.py`, `evaluate_atom_query_on_test.py`, `run_linear_spectrum_baseline.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/h/fig_atomq_bar.py` |
| **i** | Peak-token embedding UMAP property maps: fragment peak *m/z*, heavy atoms, hetero atoms, O/C ratio, parent cLogP and fragment chemotype. | `scripts/i/plot_ultrams_peak_umap.py`, `scripts/i/plot_peak_embedding_umap.py`, `scripts/i/prepare_msnlib_magma_subset.py`, `scripts/i/run_magma_msnlib_subset.py`, `scripts/i/extract_peak_embeddings.py` | `data/i/*.summary.json`; `data/i/x10/*.summary.json` (plotted summaries; the large arrays are not redistributed — see below) | `original/peak_neighbors/prepare_msnlib_magma_subset.py`, `run_magma_msnlib_subset.py`, `extract_peak_embeddings.py` | `python scripts/i/plot_ultrams_peak_umap.py` re-renders from the peak-embedding metadata and arrays obtained as described below |
| **j** | Cross-spectrum peak-neighbor fidelity: formula agreement, fragment-family agreement, fragment-structure similarity and mass fidelity versus Top-*k* neighbours. | `scripts/j/render_peak_neighbor_curves_from_data.py`, `scripts/j/plot_peak_neighbor_curves.py`, `scripts/j/plot_peak_neighbor_curves_classical.py`, `scripts/j/plot_peak_neighbor_fidelity.py` | `data/j/peak_neighbor_fidelity_metrics_x10.json`, `data/j/*curves*.csv`, `data/j/*.json` | `original/peak_neighbors/peak_neighbor_evaluation.py`; `reproduce.py` | `python benchmarks/spectral_properties/reproduce.py`; `python scripts/j/render_peak_neighbor_curves_from_data.py --metrics-json data/j/peak_neighbor_fidelity_metrics_x10.json --output-dir <out>` |

Selection identifiers (from the internal Figure 2 selection manifest) map to the manuscript panels as
follows: `a ← fig2a_accuracy` (+ `fig2a_mae` variant), `b ← fig2b`, `c ← fig2c`, `d ← fig2d`,
`e ← fig2e`, `f ← fig2g`, `g ← fig2h`, `h ← fig2f`, `i ← fig2j`, `j ← fig2k`. The selection
identifiers are **not** identical to the manuscript letters for panels f–j; use the manuscript
letter directories.

## Reproducing the reported numbers

The numeric values behind every panel are recalculated from the saved test outputs by the
benchmark package:

```bash
python -m pip install numpy pandas scikit-learn
python benchmarks/spectral_properties/reproduce.py
```

This writes `benchmarks/spectral_properties/results.csv`. Re-training or re-running the original
producers (reconstruction, ion mode, structure families, neutral loss, heteroatom counts, element
peaks, peak neighbours) is documented in
[`benchmarks/spectral_properties/EXPERIMENTS.md`](../../benchmarks/spectral_properties/EXPERIMENTS.md);
the public inputs and checkpoint downloads are listed in
[`benchmarks/spectral_properties/INPUTS.md`](../../benchmarks/spectral_properties/INPUTS.md).

## Re-rendering the panels

The registered plotting scripts are the producers used for the manuscript panels. Two conventions
apply when re-running them from this repository:

1. Most scripts derive their input roots from their own file location and expect the original
   research workspace layout (`main_figure/fig2/...` for figure data and `train/output/showcase/...`
   for run directories). They are included unchanged for provenance. To re-render from this
   package, either place the script into that layout or edit the few path constants
   (`FIG2_DIR`, `RUNS_BASE`, `DATA_PATH`, …) so that they point at the copied `data/<panel>/`
   directory. No absolute or guessed paths were substituted.
2. Scripts that take arguments (for example `render_peak_neighbor_curves_from_data.py`,
   `plot_ultrams_peak_umap.py`, `prepare_msnlib_magma_subset.py`) can be pointed at the copied data
   directories directly via `--data-dir`, `--metrics-json`, `--output-dir`, etc.

Two `fig2c` rendering scripts resolve `source_data/` as a sibling of the directory above the script
(`scripts/` and `source_data/` side by side). Copy `data/c/source_data/` next to `scripts/c/` (or run
from a copy of the original package layout) when re-rendering that panel.

`scripts/i/run_magma_msnlib_subset.py` originally contained a hard-coded MAGMa checkout path; the
default is now the relative `UltraGo`, and the checkout must be supplied explicitly with
`--ultrago-root` (see the sanitisation note below).

## Large inputs that are not redistributed

Following the repository size policy (single file ≤ 20 MB, single directory ≤ 50 MB, no binary
model checkpoints or large embedding arrays), the following inputs are **not** included. They can be
downloaded from the released benchmark assets or regenerated with the scripts listed above.

| Panel | Omitted inputs | How to obtain or regenerate |
| --- | --- | --- |
| b | Ion-polarity UMAP embedding arrays (`ion_polarity_mzmatched_{dreams,rt_only_d}_embeddings.npy`, 4.1 MB each) | Regenerate with `scripts/b/plot_ion_polarity_embedding_umap.py` from the MSnLib CSV and the released UltraMS / DreaMS checkpoints (`benchmarks/spectral_properties/INPUTS.md`). The plotted UMAP coordinates and sample tables are included. |
| c | Normalized embeddings and the 10,000-resample paired-bootstrap indices (`*_normalized_661.npy`, `original_inputs/test_*_supervised.npy`, `paired_bootstrap_indices.npy`, `*.npz`) | Regenerate with `scripts/c/original_feature_definition.py` and `scripts/c/compute_db.py` from the frozen SpectraVerse split and the released supervised checkpoints. The frozen cluster metrics, DB summaries, UMAP coordinates and molecule SVGs are included. |
| d, e | Neutral-loss test predictions (`*.npy`) and the trained task heads (`best_*.pt`, 25 MB each) | Predictions can be recalculated with `benchmarks/spectral_properties/original/neutral_loss/train_test_neutral_loss.py` from the public MassSpecGym table and the released checkpoints. The exact ROC/AP records used by the figure (`config.json`, `roc_data.json`) are included. |
| h | Atom-query probe checkpoints for S/Cl/F/Br (`best_*.pt`, `last_*.pt`, 42 MB each) | The selected heads (UltraMS epoch 13, DreaMS epoch 1) are distributed as public benchmark assets: `python benchmarks/spectral_properties/download_public_inputs.py --asset atom_query_heads`. The run configurations are included. |
| i | Peak-neighbour benchmark inputs: `peak_embeddings_ultra.npy` / `peak_embeddings_dreams.npy` (721 MB each), `peak_embedding_metadata.csv` (42 MB), `msnlib_magma_subset.csv` (21 MB), `msnlib_magma_annotations.json` (126 MB), and the UMAP coordinate tables `ultrams_peak_umap_coords*.csv` (24.8 MB each) | Download with `python benchmarks/spectral_properties/download_public_inputs.py --asset peak_neighbors`, or rebuild from the public MSnLib CSV with `scripts/i/prepare_msnlib_magma_subset.py`, `scripts/i/run_magma_msnlib_subset.py` and `scripts/i/extract_peak_embeddings.py`. The 24.8 MB UMAP coordinate tables are regenerated by `scripts/i/plot_ultrams_peak_umap.py`. |

The public assets are hosted with the UltraMS benchmark release on Hugging Face
(`dsadd4/UltraMS-Benchmark-Assets`); upstream spectra, model checkpoints and their licences are
listed in `benchmarks/spectral_properties/INPUTS.md` (MassSpecGym MIT; MSnLib / public dataset CC BY 4.0,
Zenodo 10.5281/zenodo.23963489; DreaMS MIT; UltraMS checkpoints Apache-2.0; MAGMa modules under the
MIST MIT licence).

## Rendered panel images

`outputs/<panel>/` holds the rendered panel images from the selected figure snapshot (450-dpi PNG
and vector PDF; SVG for panel c). The file names keep the original snapshot labelling
(`fig2a_accuracy`, `fig2b`, …). Two notes:

* Panel **a** also carries the registered mean-absolute-error variant (`fig2a_mae`), which is not
  part of the current manuscript panel a.
* Several panels were re-styled after this snapshot was produced; the manuscript figure
  (`Ultramsp/sec/figs/fig2.pdf`) is authoritative for the final appearance. The plotted values are
  the ones reproduced by `benchmarks/spectral_properties/reproduce.py`.

## Path hygiene

No internal absolute paths (user home directories, cluster mount points, internal account names)
appear in any file of this package. Eleven files whose *content* recorded internal absolute paths
were copied with those prefixes replaced by explicit placeholders: `<WORKSPACE_ROOT>/`,
`<REMOTE_WORKSPACE_ROOT>/`, and the relative `UltraGo` default. Each affected file is flagged in
`panel_manifest.json` with `"reused_or_copied": "copied_sanitized"` and a note. `source_path` values
in `panel_manifest.json` are relative to the original research workspace; `target_path` values are
relative to the root of this repository.

## QC note

The figure-selection manifest recorded 8,111 bytes for `scripts/a/plot_task1.py`; the file on disk
(the later version, 8,128 bytes) was copied. Every other manifest byte count matched the file on
disk. See the `warnings` array of `panel_manifest.json` for the machine-readable form.
