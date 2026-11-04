# Figure 3 — manuscript plotting code and source data

This directory packages the **plotting scripts, source data, and final outputs** for **Figure 3 (printed panels a–f in the manuscript, plus internal slots a–i in the figure workspace)** and for the Figure 3 supplementary performance-comparison figures (NPLIB1, MassSpecGym, MSnLib, source fusion, label-blind consensus, FDR threshold transfer, etc.). Files correspond to the code version used to draw the paper figures and are packaged into the UltraMS public repository (`Dsadd4/UltraMS`).

- Upstream figure workspace (**not redistributed**): `main_figure/figure3/Main_select/` (authoritative mappings: `fig3_main_select_manifest.json`, `selection_map.json`, `supp/mapping_manifest.csv`), combined with `main_figure/paper/Ultramsp/sup_figure/overall/`, `main_figure/Suppment_figure/promising/`, and the final artwork decisions of 2026-10-30 / 2026-11-04. Only the files copied into this package are distributed here.
- Manuscript: `Ultramsp/sec/figs/fig3.pdf`; text: `Ultramsp/sec/2_results.tex`; supplementary figures: `Ultramsp/sec/5_supp.tex` and `Ultramsp/sec/spfig/`.
- Existing data-generation and metric-recalculation code in the repository: `benchmarks/molecular_identification/`.
- This package only adds files under `ultrams_public_release/paper_figures/figure3/`; it does **not** modify other repository files and **does not run git add / commit / push**.

> Scope note: the current manuscript PDF (2026-09-06 snapshot) prints the six Figure 3 panels **a–f**; `fig3a`–`fig3i` are **internal slot identifiers** from the figure workspace and are not one-to-one with printed panels (see §2 and §3). `fig3_main_select_manifest.json` was generated on 2026-08-22; the author decisions of 2026-10-30 later reassigned Figure 3b, 3e, and 3f. This package keeps both the "authoritative manifest record" and the "final printed artwork" file sets, and itemizes the differences in `version_audit.md`.

## 1. Directory layout

```text
paper_figures/figure3/
├── README.md                    # this file: panel tables, provenance, reproduction, scope
├── panel_manifest.json          # per-file inventory (source_path/target_path/bytes/sha256/reused_or_copied/note)
├── version_audit.md             # itemized version audit against benchmarks/molecular_identification
├── scripts/<panel>/             # plotting scripts, copied at their original package-relative paths (byte-identical)
├── data/<panel>/                # small source data (CSV/JSON/small .gz/.npz), original package-relative paths preserved
└── outputs/                     # final panel PDF/PNG (for comparison only, all <5 MB)
```

- Covers **9 main-figure internal slots + 8 supplementary-figure slots**; the supplementary slot `fig3e_sp_1` was promoted to manuscript Figure 3c, so it is not duplicated and is registered as an alias of `fig3c`. The final package has 16 independent panel directories and 235 file entries.
- In `panel_manifest.json`, `sha256` is the SHA-256 of **the actual file in this package**; when a file was sanitized through internal-path placeholders, `source_sha256` is also given.
- `reused_or_copied` definition: `reused` = **byte-identical (same SHA-256)** to a file under `benchmarks/molecular_identification/`; `copied` = present only in the upstream figure workspace (including files that have identical values but different column structure/float serialization; see `version_audit.md`).
- `path_sanitization` records every internal-absolute-path substitution (see §6).

## 2. Main-figure panels (internal slot → current manuscript artwork)

| Internal slot | Current manuscript location | Paper content | Plotting script (this package) | Source data (this package) | Repository counterpart code/data | Reproduce / redraw |
|---|---|---|---|---|---|---|
| `fig3a` | Fig. 3a | Hit@1/5/20 for single-spectrum identification (57,437 spectra) and same-molecule multiple-spectrum voting (5,002 molecules), six methods | `scripts/fig3a/plot_fig3a_refined.py` | `data/fig3a/source_data/formal_final_six_v1/` + `data/fig3a/output/fig3a_plot_data.csv` | Generation: `source_code/original/evaluate_molecule_identification.py`, `evaluate_chemberta_readout.py`, `evaluate_multiple_spectrum_voting.py`, `extract_single_spectrum_ranks.py`; results: `data/single_method_summary.csv` (**same SHA**), `data/multiple_spectrum_voting_summary.csv` (same values, different table structure) | See §5 example 2 |
| `fig3b` | Fig. 3b | True-minus-top-ranked score margin density + ground-truth cumulative rank curve | `scripts/fig3b/export_panel.py` (vector export from the manuscript `fig3.pdf`, with value checks) | `data/fig3b/score_margin_data.csv`, `rank_curve_data.csv` | `data/score_margin_density.csv` (**same values, different float serialization**), `data/single_spectrum_rank_cdf.csv.gz` (**same SHA** as the `fig3g` source data) | Requires the manuscript PDF for re-export; value checks in `version_audit.md` |
| `fig3c` | Fig. 3c | 8-methoxyquercetin 7-O-β-D-glucopyranoside isomer case (4 peaks, candidate ranks, peak-level evidence) | `scripts/fig3c/plot_four_peaks_molecular_tie_nature.py` + 4 blind-scoring/audit scripts + `extract_from_paper.py` | `data/fig3c/source_data/` (all four_peaks tables, MAGMa evidence, counterfactual) + `BLIND_PEAK_EVIDENCE_AUDIT.md` | `data/four_peaks_break_molecular_tie_nature_fragment_source.csv`, `..._peak_source.csv`, and `data/chemical_significance_audit.json` all have the **same SHA**; `data/identification_case_ranks.csv` | See §5 example 1 |
| `fig3d` | Fig. 3d | Label-blind multiple-spectrum grouping + voting Top-1, paired gains, grouping purity | `scripts/fig3d/plot_fig3b_formal_baselines.py` (historical internal ID `fig3b`) | `data/fig3d/source_data/formal_final_six_v1/` | `data/model_performance.csv` (values identical item by item), `data/grouping_summary.csv` (**same SHA**), label-blind results `data/label_blind_*.csv.gz` | See §5 (output is written to `<panel>.parents[1]/output`) |
| `fig3e` | Fig. 3e | Same-adduct FDR–recall curves (six methods) + 0–5% FDR mean micro recall for additional adducts | `scripts/fig3e/plot_fig3e.py` | `data/fig3e/source_data/` (fdr_envelopes, working_points, low_fdr, additional bars, DONE) | Generation: `build_library_manifest.py`, `prepare_fig3h_readout_inputs.py`, `train_fig3h_s2s_readout.py`, `embed_search_fig3h_s2s_readout.py`, `evaluate_exact_fdr.py`; repository `data/strict_adduct_*`, `data/working_points_primary_seed0.csv`, `data/low_fdr_mean_recall_primary_seed0.csv`, `data/other_strict_adduct_low_fdr_summary.csv` (**same SHA**) | `python plot_fig3e.py` (data in `source_data/`) |
| `fig3f` | Fig. 3f (author decision of 2026-10-30; the 2026-09-06 `fig3.pdf` snapshot still shows the older chemical-subset panel) | Four-panel cross-polarity (positive/negative mode) retrieval figure (bidirectional Recall + paired gains/losses); the old chemical-subset concentric-ring + bar artwork moves to the supplement | `scripts/fig3f/plot_cross_polarity.py`, `prepare_source_data.py` | `data/fig3f/source_data/` (recall/paired/mrr + per-query results + PROTOCOL/verification JSON), `data/fig3f/figure3f_values.csv` (the 60 values of the chemical-subset panel in the printed snapshot) | **Missing from repository**: no cross-polarity retrieval code/data; the old chemical-subset artwork corresponds to the collision-energy part of repository `data/collision_energy_queries.csv.gz`/`results.json` (supplementary `fig3c_sp_1/2`) | `python plot_cross_polarity.py` |
| `fig3g` | Archived (Fig. 3b now includes the rank curve) | Standalone export of the discrimination-margin distribution + rank CDF | `scripts/fig3g/fig_sim_discriminability.py` | `data/fig3g/` (plot data, rank cdf, raw-score npz) | Rank CDF: `data/single_spectrum_rank_cdf.csv.gz` (**same SHA**); raw-score npz missing from repository | See §5 |
| `fig3h` | Unused (relabelled on 2026-10-30; the six-method core values are carried by Fig. 3e/repository) | Seven-method (including Raw cosine) same-adduct FDR–recall + additional adducts | `scripts/fig3h/plot_fig3h_formal_baselines.py` | `data/fig3h/` (seven-method envelopes/working points, raw matchms, additional summary) | Six-method core values: `data/working_points_primary_seed0.csv` etc. (**same values**); Raw-cosine seven-method data missing from repository | See §5 |
| `fig3i` | Registered but not included in the manuscript Figure 3 | Sample-level spectrum-to-spectrum UMAP + proline betaine case | `scripts/fig3i/plot_s2s_sample_umap_case.py` | `data/fig3i/` (UMAP points, rank table, overlay, plot_data.json) | **Missing from repository** | Requires upstream project `train/output` inputs; bundled data can be compared against the output |

> The manuscript artwork of Fig. 3c includes manual author layout refinement (`supp/fig3c_main_without_properties_20261030/`, extracted from the manuscript PDF). This package keeps both `outputs/fig3c_plotting_script_output.*` (the script-redrawable version) and `outputs/fig3c_printed_manuscript_panel.*` (the actual manuscript artwork). They are not interchangeable.
>
> The upstream figure package root also contains an older duplicate `four_peaks_break_molecular_tie_nature_peak_source.csv` (different `support_order`; sha `f4a886e7…`). It is not the canonical run input; this package copies only the `source_data/` version whose SHA matches the repository.

## 3. Supplementary performance-comparison figures (`supp/mapping_manifest.csv`)

| Supplementary figure | Content | Plotting script (this package) | Source data (this package) | Manuscript/front-end status | Repository counterpart |
|---|---|---|---|---|---|
| `fig3a_sp1` | Top-1/5/20 comparison across three tasks: NPLIB1 + MassSpecGym mass/formula (including published external values such as MIST/JESTR/ESP) | `scripts/fig3a_sp1/plot_benchmark_topn_comparison.py` | `data/fig3a_sp1/` | **Current Supplementary Fig. (`5_supp.tex` actually includes `sec/spfig/fig3a_sp1.pdf`)** | Values agree with `data/additional_molecule_identification.csv`, `nplib1_metrics.csv`, `massspecgym_*_metrics.csv` |
| `fig3a_sp_3` | Contribution of global/peak/source fusion to molecular-retrieval decisions (NPLIB1 + MassSpecGym) | `scripts/fig3a_sp_3/plot_s7.py`, `build_source_data.py`, `verify_s7.py` | `data/fig3a_sp_3/source_data/` | Front-end supplementary record | Values agree with `data/nplib1_metrics.csv`, `massspecgym_*_metrics.csv` (1 0.018 pp difference in DreaMS formula Top-5; see audit) |
| `fig3a_sp_4` | Retrieval gain from label-blind replicate-spectrum consensus | `scripts/fig3a_sp_4/plot_s9.py`, `run_analysis.py`, `audit_s9.py` | `data/fig3a_sp_4/` | Front-end supplementary record | `data/model_performance.csv`, `grouping_summary.csv`, `paired_group_bootstrap.csv` have the **same SHA** as the repository |
| `fig3a_sp_5` | Migration of the global+peak promoted pipeline to **MSnLib** UltraMS/DreaMS molecular-fingerprint retrieval | `scripts/fig3a_sp_5/plot_msnlib_promoted_fingerprint.py` | `data/fig3a_sp_5/` | Front-end supplementary record | **Missing from repository: MSnLib fingerprint task** (repository only has NPLIB1/MassSpecGym fingerprint tasks) |
| `fig3_baselines_sp_1` | Five-seed stability of Fingerprint FFN and Spec2Vec (main figure uses seed 0 only) | `scripts/fig3_baselines_sp_1/plot_fig3_baseline_seed_stability.py` | `data/fig3_baselines_sp_1/` | Front-end supplementary record | Repository has only seed 0 / single full-training results, **missing the five-seed table** |
| `fig3h_sp_1` | Transfer of a validation-locked 5% FDR threshold to unseen test molecules | `scripts/fig3h_sp_1/plot_fig3h_threshold_transfer_formal.py` | `data/fig3h_sp_1/` | Front-end supplementary record | **Missing from repository: threshold-transfer code/data** |
| `fig3e_sp_1` | Same as `fig3c` (this package was promoted to manuscript Fig. 3c; not duplicated) | — | — | See `fig3c` | See `fig3c` |
| `fig3e_sp_2` | Stress test over 256 combinations of eight peaks (candidate-molecule decision) | `scripts/fig3e_sp_2/plot_peak_addback_story_nature.py` | `data/fig3e_sp_2/` | Front-end main-figure evidence record | **Missing from repository: eight-peak add-back data** |

**Not included in this package** (outside the Figure 3 mapping requested here; the repository already has corresponding data): the cross collision-energy seven-method supplementary packages (`supp/fig3c_sp_1`, `supp/fig3c_sp_2`, corresponding to the collision-energy part of repository `data/collision_energy_queries.csv.gz` and `results/results.json`), and `fig3a_u1` (the unused Top-1-only predecessor superseded by `fig3a_sp1`).

> The current `5_supp.tex` typesets only one Figure 3 supplementary figure, `fig3a_sp1` (`sec/spfig/fig3a_sp1.pdf`); `fig3a_sp_3/4/5`, `fig3_baselines_sp_1`, `fig3h_sp_1`, and `fig3e_sp_2` are supplementary evidence packages from the figure workspace/front-end and have not yet been added one by one to the current LaTeX supplementary material.

## 4. NPLIB / MassSpecGym / MSnLib comparisons in Figure 3: which code produced them, and are they the same version as the repository?

This is the section most relevant to Nature open-source compliance review. Bottom line first: **the UltraMS/DreaMS NPLIB1 and MassSpecGym values agree item by item with the existing `benchmarks/molecular_identification/`; however, the figure "plotting code" that produced these numbers never entered the repository, which only contains data-generation and metric-recalculation code. The MSnLib promoted-fingerprint comparison (`fig3a_sp_5`) and the cross-polarity, threshold-transfer, and related figures have no corresponding code/data in the repository. The four simple encoder curves in Figure 3e use an earlier readout version, which differs from the s2s-readout version currently stored in the repository (UltraMS/DreaMS are the same).**

### 4.1 NPLIB1

- **Figure sources**: `fig3a_sp1`, `fig3a_sp_3` (plus the migration experiment in `fig3a_sp_5`).
- **Generating code (paper experiment version)**: repository `source_code/original/34_nplib1_ms2fp_contrastive_model_compare.py`, `22_nplib1_token_summary_ms2fp.py`, `36_nplib1_ms2fp_triplet_token_summary.py` train the four member sources; `38_nplib1_ms2fp_member_ensemble.py` selects the rank fusion on validation and evaluates the three test folds. Entry points: `run_nplib1_candidate_ranking.sh` / `run_public_nplib1_candidate_ranking.sh`.
- **Version relationship**: in the figure source data `data/fig3a_sp1/data/benchmark_top1_source_data.csv`, NPLIB1 `UltraMS fair fusion` / `DreaMS fair fusion` (3,699 queries) agree **bit for bit** with repository `data/nplib1_metrics.csv` aggregate `u5_cls_projmean_rawproj` / `dreams_global_iw`: UltraMS 32.981886996485535 / 57.907542579075425 / 80.5352798053528; DreaMS 27.926466612598 / 54.63638821303055 / 79.10246012435793. The global/projmean/final-fusion views in `fig3a_sp_3` also agree with the per-view rows in `nplib1_metrics.csv` (32.60340632603406 etc.).
- **Verdict**: **same-version data**. The external MIST/FFN values shown additionally in the figure come from the official MIST notebook / Supplementary Table 8 and are unrelated to the repository.

### 4.2 MassSpecGym (mass-restricted and formula-restricted)

- **Figure sources**: `fig3a_sp1` (main comparison) and `fig3a_sp_3` (source-fusion decomposition).
- **Generating code**: `39_massspecgym_ms2fp_sourcefusion.py` trains the four fingerprint sources and completes the mass task; `prepare_massspecgym_formula_candidates.py` builds the formula-candidate interface from the trained heads; `finish_massspecgym_candidate_ranking.py` selects two fusions on validation and evaluates mass/formula separately. Entry points: `run_massspecgym_candidate_ranking.sh` / `run_public_massspecgym_candidate_ranking.sh`.
- **Version relationship**: in the figure, UltraMS mass 17.092457420924575 / 33.345498783454985 / 54.89051094890512, DreaMS 13.242092457420926 / 26.271289537712896 / 45.675182481751825; formula UltraMS 14.07542579075426 / 29.03284671532847 / 51.68491484184915, DreaMS 11.411192214111923 / 23.953771289537713 / 43.400243309002434. These agree **bit for bit** with repository `data/massspecgym_mass_metrics.csv`, `data/massspecgym_formula_metrics.csv`, and `data/additional_molecule_identification.csv`.
- **Only difference**: the MassSpecGym formula view in `fig3a_sp_3` records DreaMS `global + peak fusion` Top-5 as 23.93552311435523, whereas the same fusion in the repository is 23.953771289537713 (0.018 pp difference; this is an older view table of that supplementary figure). Top-1/Top-20 and the other rows agree. See `version_audit.md`.
- **Verdict**: **same-version data (1 difference in a supplementary view table)**.

### 4.3 MSnLib (including promoted fingerprint and label-blind)

- **Single-spectrum identification / multiple-spectrum voting (Fig. 3a)**: produced by `evaluate_molecule_identification.py` + `evaluate_chemberta_readout.py` + `evaluate_multiple_spectrum_voting.py`; the results are repository `data/single_spectrum_query_ranks.csv.gz`, `single_method_summary.csv`, `multiple_spectrum_voting_*`. Figure source `data/fig3a/source_data/formal_final_six_v1/single_method_summary.csv` has the **same SHA** as the repository file → same version.
- **Label-blind (Fig. 3d, `fig3a_sp_4`)**: produced by `source_code/original/label_blind_grouping/scripts/run_analysis.py` (select a threshold on validation, then freeze test groups) and `evaluate_frozen_groups.py`. `fig3a_sp_4/statistics/{model_performance,grouping_summary,paired_group_bootstrap}.csv` have the **same SHA** as the repository; the scientific parameters of the config agree (only `project_root` and added documentation fields differ) → same version.
- **Promoted fingerprint (`fig3a_sp_5`)**: migrates the UltraMS/DreaMS global+peak fusion to the 57,437 MSnLib test spectra and ranks candidates with 2,048-bit Morgan fingerprints. **The repository has no code or data for this task**: repository `reproduce_fingerprint_prediction.py` covers only NPLIB1 and MassSpecGym, and `data/fingerprint_prediction/` contains only the test-query tables for those two tasks. This package keeps only the summary metrics and validation/test metrics JSON (`data/fig3a_sp_5/`); the **training/inference scripts are not in the repository** and need to be added.
- **Same-adduct FDR (Fig. 3e / former Fig. 3h)**: produced by `build_library_manifest.py` (70/15/15 molecule-level split), `prepare_fig3h_readout_inputs.py`, `train_fig3h_s2s_readout.py`, `embed_search_fig3h_s2s_readout.py`, and `evaluate_exact_fdr.py`.
  - **UltraMS/DreaMS fully agree** (97.13%/95.70%, 95.44%/93.73%; identical working-point values).
  - **The four simple methods (Linear/DeepSets/Fourier/Codebook) are a different version**: paper Fig. 3e uses the earlier readout (`formal_six_readout_reuse/`), whereas the repository currently stores the s2s-readout version (`formal_six_s2s_readouts/`). For example, the [M+H]+ Linear 5% FDR working point is 0.3138203515562006 (2,048 accepted) in the paper-figure version and 0.3559103370424125 (2,323 accepted) in the repository. The additional-adduct bar values (`other_strict_adduct_low_fdr_summary.csv`) agree between the two versions, and the repository file has the **same SHA** as the figure source.
  - Verdict: **same task, different readout version**; when reading the paper Fig. 3e curves for the simple methods, you must use `data/fig3e/` from this package and cannot redraw directly from repository `working_points_primary_seed0.csv`.
- **Cross-polarity (`fig3f`)**: MSnLib positive/negative cross-mode retrieval; **the repository has neither the code nor the data**.
- **Threshold transfer (`fig3h_sp_1`)**: **the repository has neither the code nor the data**.

### 4.4 Summary

| Comparison source | Figure packages | Same version as repository? | Repository gap |
|---|---|---|---|
| NPLIB1 | `fig3a_sp1`, `fig3a_sp_3` | Same-version values (bit-for-bit) | Plotting scripts |
| MassSpecGym mass/formula | `fig3a_sp1`, `fig3a_sp_3` | Same-version values (1 0.018 pp difference in a supplementary view table) | Plotting scripts |
| MSnLib single-spectrum/voting | `fig3a`, `fig3d` (internal `fig3b`) | Same version (same SHA) | Plotting scripts |
| MSnLib label-blind | `fig3d`, `fig3a_sp_4` | Same version (statistics tables same SHA) | Plotting scripts |
| MSnLib promoted fingerprint | `fig3a_sp_5` | **Different task, missing from repository** | All code + per-query results |
| MSnLib same-adduct FDR | `fig3e`, `fig3h` | Six-method core values agree; four simple methods are a **different readout version**; `fig3h` adds Raw-cosine seven-method results | Raw-cosine/seven-method data, paper-version readout metadata |
| MSnLib cross-polarity | `fig3f` | **Missing from repository** | All code + per-query results |
| MSnLib threshold transfer | `fig3h_sp_1` | **Missing from repository** | All code + repeated-split data |
| Five-seed stability | `fig3_baselines_sp_1` | **Repository missing multi-seed table** | Seed 1–4 metrics |
| eight-peak add-back | `fig3e_sp_2` | **Missing from repository** | Combination enumeration and decision-path data |
| sample-level s2s case | `fig3i` | **Missing from repository** | UMAP/retrieval inputs for this case |

## 5. Reproduction / redrawing

In the original packages, plotting scripts load inputs via `Path(__file__).resolve().parents[1]` or a sibling `source_data/`. Therefore, **restoring this package's scripts and data to their original package-relative positions allows them to run unchanged** (the scripts were copied byte for byte). The directory hierarchy of `scripts/<panel>/` and `data/<panel>/` follows the original package layout as closely as possible (e.g. `data/fig3a/source_data/`, `data/fig3a_sp1/data/`). Examples:

```bash
# Example 1: fig3c (scripts at package root; data in source_data/)
P=/tmp/ultrams_fig3_rerun/fig3c
mkdir -p "$P/out"
cp scripts/fig3c/*.py "$P"/
cp -r data/fig3c/source_data "$P"/source_data
cp data/fig3c/BLIND_PEAK_EVIDENCE_AUDIT.md "$P"/
python "$P/plot_four_peaks_molecular_tie_nature.py" --data-dir "$P/source_data" --output-dir "$P/out"

# Example 2: fig3a (scripts in scripts/; data in source_data/ and output/)
P=/tmp/ultrams_fig3_rerun/fig3a
mkdir -p "$P/scripts"
cp scripts/fig3a/plot_fig3a_refined.py "$P/scripts/"
cp -r data/fig3a/source_data data/fig3a/output "$P"/
python "$P/scripts/plot_fig3a_refined.py"

# Example 3: fig3a_sp1 (scripts in scripts/, data in data/, outputs back to package root)
P=/tmp/ultrams_fig3_rerun/fig3a_sp1
mkdir -p "$P/scripts"
cp scripts/fig3a_sp1/plot_benchmark_topn_comparison.py "$P/scripts/"
cp -r data/fig3a_sp1/data "$P"/data
cp data/fig3a_sp1/fig3a_sp1_plot_data.csv data/fig3a_sp1/fig3a_sp1_chart_contract.json "$P"/
python "$P/scripts/plot_benchmark_topn_comparison.py"
```

| Panel | Script restore location (relative to panel root) | Data restore location | Run / notes |
|---|---|---|---|
| `fig3a` | `scripts/` | `source_data/`, `output/` (both in `data/fig3a/`) | `python scripts/plot_fig3a_refined.py` |
| `fig3b` | package root `export_panel.py` | two CSVs at package root | Reads the manuscript `Ultramsp/sec/figs/fig3.pdf`; without that PDF only the bundled `outputs/` and the two CSVs can be used |
| `fig3c` | 6 scripts at package root | `source_data/` + `BLIND_PEAK_EVIDENCE_AUDIT.md` | `python plot_four_peaks_molecular_tie_nature.py --data-dir <root>/source_data --output-dir <out>`; the 4 helper scripts load data by original package-relative paths |
| `fig3d` | `scripts/` | `source_data/formal_final_six_v1/` (see `data/fig3d/`) | `python scripts/plot_fig3b_formal_baselines.py`; output is written to `<panel>.parents[1]/output`; keeping the `refinement/<panel>/` hierarchy is recommended |
| `fig3e` | package root `plot_fig3e.py` | `source_data/` (see `data/fig3e/`) | `python plot_fig3e.py`; output is written back to the panel root |
| `fig3f` | 2 scripts at package root | `source_data/` (see `data/fig3f/`) | `python plot_cross_polarity.py` (optionally `prepare_source_data.py` first) |
| `fig3g` | `scripts/` | `source_data/`, `output/` | Uses the original npz as fallback (bundled at `data/fig3g/source_data/`); the complete upstream experiment path is still determined by the original script |
| `fig3h` | `scripts/` | `source_data/formal_seven_raw_matchms/`, `source_data/other_strict_adduct_v5_s2s_raw_seven_methods/`, `output/` | `python scripts/plot_fig3h_formal_baselines.py`; output is written to `<panel>.parents[1]/output` |
| `fig3i` | `scripts/panel_sources/task8_s2s_sample_umap/` | upstream project data directory + panel `data/` | The complete case pipeline needs upstream `train/output`; bundled data can be compared against the final figure |
| `fig3a_sp1` | `scripts/` | `data/` + package-root plot data | See example 3 above |
| `fig3a_sp_3` | `scripts/` | `source_data/`, `config/`, `output/` | `python scripts/plot_s7.py`; `build_source_data.py` can regenerate the omitted large table |
| `fig3a_sp_4` | `scripts/` | `source_data/`, `statistics/`, `config/`, `audit/`; output `outputs/` | `python scripts/plot_s9.py`; **the script raises an error when the omitted `group_candidate_vote_ledger_test.csv.gz` (>20 MB) is absent**, so first regenerate that large table via `run_analysis.py` or the repository `run_public_grouping_evaluation.sh` route; the final figure is available in `outputs/fig3a_sp_4.*` |
| `fig3a_sp_5` | `scripts/` | `source_data/`, `output/` | `python scripts/plot_msnlib_promoted_fingerprint.py`; per-query results are not in this package |
| `fig3_baselines_sp_1` | package-root script | 3 `seed_stability_*_source.csv` files at package root | The bundled aggregate CSVs can be used to check values directly; rerunning the script needs three upstream experiment directories (`per_seed_metrics.csv`, `per_seed_performance.csv`, `low_fdr_mean_recall_per_seed.csv`), which are not bundled |
| `fig3h_sp_1` | package-root script | `source_data/`, package-root summary tables | `--aggregate-dir <dir> --output-dir <dir>`; the script also validates the upstream `DONE.json` or `threshold_transfer_summary.json` completion marker (requiring `n_repeats=200`, `target_fdr=0.05`, table sha match), and that upstream aggregate directory is not bundled. The bundled `threshold_transfer_repeats.csv.gz` + `PLOT_AUDIT.json` can be used for checking but are not sufficient for a standalone rerun |
| `fig3e_sp_2` | package-root script | `source_data/` | `--data-dir <root>/source_data --output-dir <out>` |

Environment: Python 3 + `numpy`, `pandas`, `matplotlib` (`fig3c` additionally needs `rdkit`, `scipy`). The font sizes/canvas constants in the scripts are the final paper dimensions; when comparing redraws against `outputs/`, compare the PDF vectors first.

## 6. Internal paths and data compliance

- **Internal absolute paths**: internal HPC / local-workspace absolute paths (and internal usernames) that appeared in the original figure source tables were replaced with placeholders in **the 14 copied data files (15 substitution records)**: `<INTERNAL_HPC_ROOT>`, `<LOCAL_WORKSPACE_ROOT>`, `<INTERNAL_HOME>`, `<INTERNAL_CODE_ROOT>`; each is recorded in `panel_manifest.json` under `path_sanitization` (file, kind, count). **All 27 scripts were copied byte for byte and contain no internal paths** (verified by scanning). The final package (including PDF/SVG/PNG/CSV/JSON) **contains no internal absolute path or internal username**.
- **Effect of placeholders**: only provenance/source-note columns are affected (for example, the original server path of a data row); no value, filtering step, or figure geometry is changed.
- **Size limits**: files >20 MB or panel data directories >50 MB are not copied. The largest single file in this package is 5.1 MB (`data/fig3d/source_data/formal_final_six_v1/per_group_metrics_all_runs.csv.gz`); the largest panel data directories are `data/fig3a_sp_4` (11.6 MB) and `data/fig3g` (9.2 MB), both below 50 MB.
- **Skipped large files** (see `panel_manifest.json.skipped_large_files` for details):
  1. `fig3a_sp_3`: `source_data/massspecgym_query_decomposition.csv` (37.3 MB) — per-query MassSpecGym decomposition table; regenerate with `scripts/fig3a_sp_3/build_source_data.py` from the original experiment outputs; aggregate statistics (`massspecgym_view_metrics.csv`, `massspecgym_outcome_stats.csv`) are bundled.
  2. `fig3a_sp_4`: `source_data/group_candidate_vote_ledger_test.csv.gz` (37.1 MB) — full per-group candidate vote ledger; can be rebuilt by `run_analysis.py` from the original MSnLib top-50 test lists (the repository `run_public_grouping_evaluation.sh` provides the public route); small tables such as purity/rank trajectories are bundled.
  3. `fig3h_sp_1`: `source_data/molecule_split_assignments.csv.gz` (21.9 MB) — molecule split assignment table; can be rebuilt from the 70/15/15 canonical isomeric SMILES split of the strict-adduct manifest; repeated-split summary tables are bundled.
- `.tar.gz` archives, model weights, per-query raw large tables, and rendering caches from the upstream packages were not copied.

## 7. Relationship to repository `benchmarks/molecular_identification/`

- The repository provides **data generation + metric recalculation** (`reproduce.py`, `recompute_library_search.py`, `reproduce_fingerprint_prediction.py`, `source_code/original/*`, `data/*`), and **contains no matplotlib plotting script** (no `import matplotlib` anywhere under the benchmarks); this package supplies exactly the paper-figure plotting code and figure source data.
- See `version_audit.md` for the per-file version comparison (identical / possibly different version / missing from repository).
- Recommended follow-up additions to the repository (by priority): paper-version Fig. 3e four-simple-method readout metadata with per-query scores, Raw-cosine seven-method same-adduct scores, the full MSnLib promoted-fingerprint pipeline, cross-polarity retrieval and threshold-transfer code, the five-seed stability table, eight-peak add-back enumeration data, and sample-level s2s case inputs.

## 8. Integrity

- `panel_manifest.json` contains **235 file entries**: 27 scripts + 173 data + 35 outputs, totaling **52,085,407 bytes**.
- Verification: every `sha256` and `bytes` value is checked item by item; the 8 `reused` entries can be compared directly with the same-named files under `benchmarks/molecular_identification/`.
- This package was generated from the 2026-11-04 figure-workspace snapshot; no network downloads, no uploads, no credentials written, and no git operations were performed.
