# Figure 3 artwork version audit (vs `benchmarks/molecular_identification/`)

Audit date: 2026-11-04. Audited objects: the plotting scripts, source data, and final outputs under `paper_figures/figure3/`. Control: the public repository `ultrams_public_release/benchmarks/molecular_identification/` (main branch; the workspace already had two unrelated uncommitted modifications, `.gitignore` and `data/label_blind_grouping/cluster_consensus_summary.json`, which this task did not touch). Paths under `main_figure/...` and `refinement/...` refer to the **upstream figure workspace, not redistributed**; only files copied into this package are distributed.

## 0. Method and criteria

1. **Byte-for-byte comparison**: for each figure source data file, compute SHA-256 against the same-named and semantically corresponding files under repository `data/`. Identical SHA → `identical (sha256 given)`.
2. **Value comparison**: when SHA differs but files are semantically corresponding, compare values row by row / column by column (direct float comparison, tolerance 0; tolerance is used only to describe float-printing differences). All values identical, with only column structure/row order/float serialization differing → `same-version values, different serialization`.
3. **Code comparison**: `benchmarks/molecular_identification/` **contains no matplotlib plotting script** (no `import matplotlib` anywhere under the benchmarks), only data-generation, evaluation, and metric-recalculation programs; all 27 plotting scripts in this package are missing from the repository. "Same script version" is therefore judged at the "same experiment-version plotting/consumption layer", and the corresponding repository data-generation program is identified.
4. **Missing determination**: no corresponding implementation/result in repository `data/` or `source_code/original/` → `missing from repository`.
5. Internal absolute paths were replaced as described in README §6; the SHA-256 values below are for the actual files in this package (target files). When placeholders were substituted, `source_sha256` is recorded (see the corresponding entries in `panel_manifest.json`).

## 1. Identical: byte-for-byte same SHA-256 (`reused`, 8 entries)

| File in this package | Repository counterpart | SHA-256 | bytes |
|---|---|---|---|
| `data/fig3a/source_data/formal_final_six_v1/single_method_summary.csv` | `data/single_method_summary.csv` | `35436465dac896e5d25167c4063c64481aa3e353f8a4e19fd136b0ff9f7d62ff` | 2045 |
| `data/fig3a_sp_4/statistics/model_performance.csv` | `data/model_performance.csv` | `96be66cb4f4263ea4a73d16914bca3f7f2a56fed8b3395917224a3d737de12ee` | 2760 |
| `data/fig3a_sp_4/statistics/grouping_summary.csv` | `data/grouping_summary.csv` | `327b5a58eed09a57a08e68c848720e3d64936add830be6e2d9f5f8de6c1df043` | 443 |
| `data/fig3a_sp_4/statistics/paired_group_bootstrap.csv` | `data/paired_group_bootstrap.csv` | `34210335f9c119ced488220c9f708ebd1c601642ca6bed6f021eab5c11aa5ccb` | 565 |
| `data/fig3c/source_data/chemical_significance_audit.json` | `data/chemical_significance_audit.json` | `791580a7395c620c23500842151ac92e2cb654dbf35d5ff004d6164aff11e4f1` | — |
| `data/fig3c/source_data/four_peaks_break_molecular_tie_nature_fragment_source.csv` | `data/four_peaks_break_molecular_tie_nature_fragment_source.csv` | `312b4437b46ebda9ee6deb755fa8500a5404fb1688e0931f484add20a5872591` | 841 |
| `data/fig3c/source_data/four_peaks_break_molecular_tie_nature_peak_source.csv` | `data/four_peaks_break_molecular_tie_nature_peak_source.csv` | `ee80f3c0820b6c75b32d84f9e9ee4fdb9cd3d2dd847b18b8f003567eb830d9a9` | 415 |
| `data/fig3e/source_data/other_strict_adduct_low_fdr_summary.csv` | `data/other_strict_adduct_low_fdr_summary.csv` | `8d8d6d025163edf74986622a5096d1c271c65ef290c4216543c913fe8a65de18` | 4341 |

Two additional items must be registered (not same-named `data/` files, but produced by the same repository implementation):

- `data/fig3g/source_data/formal_final_six_v1/rank_cdf_source_data.csv.gz` has the **same SHA** as repository `data/single_spectrum_rank_cdf.csv.gz` (`7baabe0926fa…`, 68,384 bytes); the Fig. 3b rank curve and `fig3g` are both derived from it.
- `data/fig3a_sp_4/config/analysis_config.json` has the same scientific parameters as repository `source_code/original/label_blind_grouping/config/analysis_config.json` (differences in §3.3).

**Conclusion**: the six-method single-spectrum summary of Fig. 3a, the three label-blind statistics, the additional-adduct summary of Fig. 3e, and the Fig. 3c fragment/peak source tables plus the chemical-significance audit are all **the same version** as the repository and can be reused/compared directly.

## 2. Identical values, different serialization or table structure (same-version data)

| File in this package | Repository counterpart | Comparison result |
|---|---|---|
| `data/fig3b/score_margin_data.csv` (sha `514afa8b…`) | `data/score_margin_density.csv` (sha `053bceba…`) | 6 models × 300 rows, 1,800 rows in total; `score_margin / scaled_density / n_queries / n_wrong / wrong_fraction / kde_bandwidth` are **all identical value by value (max abs diff = 0)**. Byte differences come only from float-printing format. |
| `data/fig3b/rank_curve_data.csv` | repository `data/single_spectrum_rank_cdf.csv.gz` | The curve is derived from the latter; the `fig3g` source data has the same SHA as the repository file (see §1). |
| `data/fig3a/source_data/formal_final_six_v1/oracle_method_summary.csv` | `data/multiple_spectrum_voting_summary.csv` | Top-1/5/20 of the `oracle_top50_consensus_cluster_majority` rows for UltraMS/DreaMS/Linear/DeepSets/Fourier/Codebook are **bit-for-bit identical**; this package is a wide table (with seed statistics), the repository is a narrow table. |
| `data/fig3d/source_data/formal_final_six_v1/method_summary.csv` (sha `02a59bb2…`) | `data/model_performance.csv` (sha `96be66cb…`) | For the six methods, `all_grouped` single Top-1 / consensus Top-1 / group-majority Top-1 are **bit-for-bit identical** (e.g. UltraMS 0.2736316497023238 / 0.29245246783176493 / 0.2851167878514896). This package has more seed columns and fewer stratum rows. |
| `data/fig3d/source_data/formal_final_six_v1/paired_group_bootstrap.csv` (sha `83cf575c…`) | `data/paired_group_bootstrap.csv` (sha `34210335…`) | estimates and CIs are identical; extra columns `comparison_policy / metric / bootstrap_seed`; DreaMS MRR CI last digits `…93061` vs `…93063` (~2e-17, printing rounding). The repository file holds label-blind statistics; `fig3d` holds single-vs-grouped statistics — **not the same table**; do not substitute one for the other. |
| `data/fig3h/source_data/formal_seven_raw_matchms/working_points_primary_seed0.csv` | `data/working_points_primary_seed0.csv` | The 12 rows (six methods × two polarities) for `threshold / empirical_fdr / identification_recall / accepted / correct` are **bit-for-bit identical**; this package adds two `Raw cosine` rows and prints `seed` as `0.0` (repository: `0`). |
| `data/fig3a_sp1/data/benchmark_top1_source_data.csv` | `data/additional_molecule_identification.csv` | Intersection by (benchmark, model, metric): 51 keys, **51/51 values identical**; the repository has 21 extra leaderboard rows (ESP/DeepSets/DeepSets+Fourier for MassSpecGym mass/formula). This package file was path-sanitized; target sha is `91f46a22…`. |
| `data/fig3a_sp_3/source_data/nplib1_view_metrics.csv` | `data/nplib1_metrics.csv` | UltraMS final fusion 32.981886996485535, global+projected 32.60340632603406, DreaMS final fusion 27.926466612598 etc. are **bit-for-bit identical** to repository aggregate rows; the view decomposition is finer. |
| `data/fig3a_sp_3/source_data/massspecgym_view_metrics.csv` | `data/massspecgym_mass_metrics.csv` / `massspecgym_formula_metrics.csv` | UltraMS/DreaMS mass/formula fusion values are **bit-for-bit identical**; the only difference: the formula view records DreaMS `global + peak fusion` Top-5 as 23.93552311435523, while the same fusion in the repository is 23.953771289537713 (0.0182 pp difference; Top-1/Top-20 identical). |
| `data/fig3e/source_data/fig3e_additional_adduct_bars.csv` | repository `results/results.json` `additional_adduct_library_search` | The 12 mean micro recall values are **bit-for-bit identical** (e.g. UltraMS pos 17.57733351405461, neg 35.71214784131094; Fourier pos 14.250552463407857). |

**Conclusion**: the files above can be cited as "same-version data"; however, for publication redraws use the target files in this package, not the repository files (column structure/row order differ, and script parsing would fail or misalign).

## 3. Possibly different versions (not directly interchangeable; manual confirmation required)

### 3.1 Fig. 3e: the four simple-encoder same-adduct curves come from a different readout version (most important)

- Figure sources: `data/fig3e/source_data/{fdr_envelopes_primary_seed0.csv.gz, working_points_primary_seed0.csv, low_fdr_mean_recall_primary_seed0.csv}`.
- Upstream evidence: these three files have the **same SHA** as the same-named files under `main_figure/figure3/Main_select/refinement/fig3h/source_data/formal_six_readout_reuse/` (upstream figure workspace, not redistributed); whereas repository `data/working_points_primary_seed0.csv` and `data/low_fdr_mean_recall_primary_seed0.csv` have the **same SHA** as the same-named files under `formal_six_s2s_readouts/` (repository `working_points` sha `9ada1dc6d9b1…` vs figure source `12f0a1590b54…`).
- Value differences (5% empirical FDR working point, positive / negative):

| Method | Polarity | Paper Fig. 3e version (this package) | Repository current version |
|---|---:|---|---|
| Linear | + | recall 0.3138203515562006 (accepted 2048) | 0.3559103370424125 (accepted 2323) |
| Linear | − | 0.2439873417721519 (811) | 0.2610759493670886 (868) |
| DeepSets | + | 0.1167553620383809 (762) | 0.13707466537655216 (894) |
| DeepSets | − | **0 (no eligible point, accepted 0)** | 0.1512658227848101 (502) |
| Fourier | + | 0.3780035478148685 (2467) | 0.38800193517174647 (2532) |
| Fourier | − | 0.2892405063291139 (962) | 0.30253164556962026 (1006) |
| Codebook | + | 0.2591517497177874 (1691) | 0.297532656023222 (1942) |
| Codebook | − | 0.2012658227848101 (669) | 0.21518987341772153 (715) |

  UltraMS / DreaMS are completely identical between the two versions (97.13/95.70, 95.44/93.73; working-point values bit-for-bit identical).
- Version note (from `refinement/fig3e_fdr_v1_20261030/DESIGN_AND_VISUAL_QC.md` in the upstream figure workspace, not redistributed): the manuscript PDF panel is the **six-method version whose four simple encoders use the earlier readout**; the front-end `fig3h` record dated 2026-10-30 is the **seven-method version (with Raw cosine, spectrum-pair-trained)**, whose four simple-method values differ. The repository stores exactly the s2s-readout version.
- **Verdict**: **same task, different readout version**. Redrawing manuscript Fig. 3e requires `data/fig3e/` from this package; plotting directly from repository `data/working_points_primary_seed0.csv` produces a different curve version. If the repository claims to "reproduce Figure 3e", it must add the paper-version readout configuration/per-query scores.

### 3.2 Fig. 3c: canonical inputs match the repository SHA, but the figure package root has an older duplicate

- The canonical run input is `supp/fig3e_sp_1/source_data/` (because the script also needs `peak_embedding_cross_parent_20260823/`); this package copies only that directory. Therein, `four_peaks_break_molecular_tie_nature_fragment_source.csv`, `..._peak_source.csv`, and `chemical_significance_audit.json` have the **same SHA** as the repository (§1).
- **Caution**: the upstream figure package root contains another older `four_peaks_break_molecular_tie_nature_peak_source.csv` (sha `f4a886e7…`), whose `support_order`/`story_order` differ from the canonical input: the old copy is 18→1, 21→2, 11→3, 51→4, while the canonical input/repository is 18→1, 51→2, 11→3, 21→4 (same peaks and peak set). That old copy is not the input actually read by the plotting script and was not copied into this package. Taking that root file to redraw would produce artwork with a different peak order.

### 3.3 Fig. 3h: seven methods (including Raw cosine) vs the repository's six methods

- `data/fig3h/` in this package is the seven-method version: it adds `Raw cosine` (`formal_seven_raw_matchms/raw_matchms_*`) to the repository's six methods and keeps the additional-adduct summary from `other_strict_adduct_v5_s2s_raw_seven_methods`.
- The 12 core six-method rows match repository `data/working_points_primary_seed0.csv` (§2), but the repository has no Raw-cosine same-adduct scores or enumeration code.
- **Verdict**: the repository lacks Raw-cosine data; the current manuscript Figure 3e uses the paper version in §3.1, not this seven-method version. The `fig3h` slot has been unused since 2026-10-30.

### 3.4 Other registered formatting/minor differences

- `data/fig3a_sp1/data/benchmark_top1_source_data.csv`, `provenance.json`, `fig3a_sp1_plot_data.csv`: internal HPC paths were replaced with placeholders (original sha differs from target sha; `source_sha256` is given in the manifest). Values are unaffected.
- `data/fig3a_sp_4/audit/upstream_input_manifest.csv`, `config/analysis_config.json`, `data/fig3g/output/fig_sim_discriminability_plot_data.csv`, `data/fig3h/.../{DONE.json,raw_matchms_summary.csv,AUDIT.json}`, `data/fig3f/source_data/{SOURCE_BINDING.json,original/INDEPENDENT_RESULT_VERIFICATION.json}`, `data/fig3i/plot_data.json`, `data/fig3a_sp_3/source_data/provenance.json`, `data/fig3h/source_data/other_strict_adduct_v5_s2s_raw_seven_methods/raw_matchms/AUDIT.json`: as above, only source paths were substituted; values/artwork are unchanged.

## 4. Missing from the repository: scripts/data that need to be added

Repository `benchmarks/molecular_identification/` contains only data-generation/evaluation/recalculation code, **no plotting scripts** (no matplotlib anywhere). All 27 scripts in this package are additions. Grouped by gap:

| Gap | Affected panel | Repository status | Suggested addition |
|---|---|---|---|
| All Figure 3 plotting scripts | all panels | none | The `scripts/` here can serve as the public version; the repository README should reference this directory |
| Full MSnLib promoted-fingerprint pipeline | `fig3a_sp_5` | The repository fingerprint task covers only NPLIB1 and MassSpecGym mass/formula (`reproduce_fingerprint_prediction.py`, `data/fingerprint_prediction/`) | Training/fusion/inference scripts + 57,437 test-query results + validation weight-selection records |
| Cross-polarity cross-mode retrieval | `fig3f` | no code, no data | Positive/negative reciprocal-retrieval query/library lists, embedding extraction, paired-statistics scripts |
| FDR threshold transfer | `fig3h_sp_1` | none | Independent-molecule split repeated-split script + `threshold_transfer_repeats.csv.gz` generator |
| Five-seed baseline stability | `fig3_baselines_sp_1` | only single-run/seed-0 results (`data/single_spectrum_baselines_full_training/results.csv`) | Fingerprint FFN and Spec2Vec seed 1–4 metrics and run configurations |
| eight-peak add-back | `fig3e_sp_2` | none | 8-peak 256-combination enumeration, decision paths, candidate-molecule source tables (results already bundled here) |
| Sample-level s2s case | `fig3i` | none | Sample-level UMAP/retrieval inputs and case-selection records |
| Discrimination-margin raw scores | `fig3g` | only `data/score_margin_density.csv`, `data/single_spectrum_rank_cdf.csv.gz`; no raw npz | `10_raw_scores_msnlib_mass_{rt_only_d11,dreams}_test.npz` (already bundled here) |
| Raw-cosine / seven-method same-adduct | `fig3h` | six methods only (`evaluate_exact_fdr.py`, `strict_adduct_scores/`) | `raw_matchms` seven-method scores and full-library matching script |
| Paper-version Fig. 3e readout | `fig3e` | repository has the s2s-readout version | Paper-version four-simple-method readout configuration/per-query scores (needed for exact reproduction if archived with the paper) |
| Fig. 3c complete case data | `fig3c` | only `data/four_peaks_break_molecular_tie_nature_{fragment,peak}_source.csv`, `identification_case_ranks.csv`, `chemical_significance_audit.json` | Full rank/raw-spectrum/candidate/descriptor/MAGMa evidence tables (already bundled here) |
| Fig. 3d single-vs-grouped bootstrap | `fig3d` | `model_performance.csv` has deltas but not the full paired bootstrap table | `paired_group_bootstrap` generator and full table (already bundled here; note it differs from the label-blind `data/paired_group_bootstrap.csv`) |
| Large tables (>20 MB, not copied here) | `fig3a_sp_3`, `fig3a_sp_4`, `fig3h_sp_1` | none | See README §6 for regeneration methods |
| Upstream completion markers/aggregate directories | `fig3h_sp_1`, `fig3_baselines_sp_1` | none | `fig3h_sp_1` `DONE.json`/`threshold_transfer_summary.json`; the three per-seed upstream directories of `fig3_baselines_sp_1`; this package includes only the aggregate tables and PLOT_AUDIT |

## 5. Code-level correspondence (function/file differences)

| Plotting script in this package | Repository implementation | Key difference |
|---|---|---|
| `scripts/fig3a/plot_fig3a_refined.py` | `reproduce.py` (`single_spectrum`, `multiple_spectrum_voting`), `source_code/original/evaluate_molecule_identification.py`, `evaluate_chemberta_readout.py`, `evaluate_multiple_spectrum_voting.py` | The figure script only consumes `formal_final_six_v1/*` summary tables and typesets them; the repository recomputes metrics from per-query ranks and writes `results/results.json`, with no plotting layer. |
| `scripts/fig3b/export_panel.py` | `reproduce.py` | The figure script exports the panel as vectors from the manuscript PDF and checks displayed values; the repository only recomputes metrics. |
| `scripts/fig3c/plot_four_peaks_molecular_tie_nature.py` + 4 helper scripts | none | The repository has only the two peak source tables and the chemistry audit, with no case-plotting/blind-scoring/MAGMa evidence programs. |
| `scripts/fig3d/plot_fig3b_formal_baselines.py` | `reproduce.py`, `source_code/original/evaluate_frozen_groups.py` | The figure script consumes `formal_final_six_v1/{method_summary,paired_group_bootstrap,per_group_metrics_all_runs}`; the repository recomputes from `label_blind_per_query.csv.gz` etc. |
| `scripts/fig3e/plot_fig3e.py` | `evaluate_exact_fdr.py`, `recompute_library_search.py` | The figure script consumes the paper-version FDR envelopes/working points; the repository recomputes from `strict_adduct_scores/top1_*.npz` (s2s version). |
| `scripts/fig3f/plot_cross_polarity.py` | none | No cross-polarity retrieval implementation in the repository. |
| `scripts/fig3g/fig_sim_discriminability.py` | `reproduce.py` (score-margin side) | The figure script reads raw-score npz files to plot the margin distribution and CDF; the repository only outputs `score_margin_density.csv` / `single_spectrum_rank_cdf.csv.gz`. |
| `scripts/fig3h/plot_fig3h_formal_baselines.py` | `evaluate_exact_fdr.py`, `evaluate_other_strict_adduct_six_methods.py`, etc. | The figure script reads seven-method raw-matchms results; the repository evaluator is six-method. |
| `scripts/fig3i/plot_s2s_sample_umap_case.py` | none | The sample-level s2s case pipeline is not in the repository. |
| `scripts/fig3a_sp1/plot_benchmark_topn_comparison.py` | `reproduce_fingerprint_prediction.py` | The figure script reads a static wide table (including external MIST/JESTR/ESP/leaderboard values) and typesets it; the repository only recomputes Top-1/5/20 from UltraMS/DreaMS test queries. |
| `scripts/fig3a_sp_3/{plot_s7.py,build_source_data.py,verify_s7.py}` | `38_nplib1_ms2fp_member_ensemble.py`, `39_massspecgym_ms2fp_sourcefusion.py`, `finish_massspecgym_candidate_ranking.py` | The figure script performs global/peak/fusion view decomposition and decision ledgers; the repository handles training and fusion scoring. |
| `scripts/fig3a_sp_4/{plot_s9.py,run_analysis.py,audit_s9.py}` | `source_code/original/label_blind_grouping/scripts/run_analysis.py`, `evaluate_frozen_groups.py` | The figure script consumes frozen assignments/threshold/statistics; the repository version rebuilds groups from MSnLib spectra and retrieval results and selects the threshold (the public route reproduces 15,541 groups / 52,070 spectra / 98.10% purity). |
| `scripts/fig3a_sp_5/plot_msnlib_promoted_fingerprint.py` | none | The repository fingerprint task does not include MSnLib. |
| `scripts/fig3_baselines_sp_1/plot_fig3_baseline_seed_stability.py` | none | No five-seed aggregation layer in the repository. |
| `scripts/fig3h_sp_1/plot_fig3h_threshold_transfer_formal.py` | none | No such transfer experiment in the repository. |
| `scripts/fig3e_sp_2/plot_peak_addback_story_nature.py` | none | No add-back enumeration in the repository. |

## 6. Conclusions and recommendations

1. **It can be stated directly that "the figure data are the same version as the repository"**: Fig. 3a six-method summary (same SHA), Fig. 3d label-blind three statistics (same SHA), Fig. 3e additional-adduct summary (same SHA), Fig. 3c fragment/peak source tables and chemistry audit (same SHA), and the NPLIB1/MassSpecGym UltraMS/DreaMS Top-k values (bit-for-bit identical).
2. **It cannot be stated that "the repository fully reproduces Figure 3"**: the repository lacks all plotting scripts and several classes of figure source data (§4); in particular, the Fig. 3e four-simple-method curves (paper-version readout), Raw-cosine seven-method, MSnLib promoted fingerprint, cross-polarity, threshold transfer, five-seed, eight-peak, and sample-level case have no corresponding implementation in the repository.
3. **The two Fig. 3e versions must be cited separately**: the manuscript PDF panel (six methods, earlier readout) uses `data/fig3e/` from this package; the front-end `fig3h` record (seven methods, s2s + Raw cosine) uses `data/fig3h/` from this package. UltraMS/DreaMS are identical in both versions, while the four simple methods differ; do not mix them or cite their values interchangeably.
4. **There are two Fig. 3c peak-order versions**: the canonical input matches the repository (`51→2`, `21→4`); the old duplicate at the figure package root has `21→2`, `51→4`, has been confirmed not to be the actual plotting input, and should not be used as source data.
5. **Recommended repository commit order**: first add `paper_figures/figure3/` from this package (code + source data + figures), then add the training/evaluation code for MSnLib promoted fingerprint, cross-polarity, and threshold transfer in the §4 priority order; when adding code, follow the repository's existing "public inputs + workspace-parameterized" convention (do not hard-code internal absolute paths).
6. This audit is based on the 2026-11-04 filesystem snapshot; if the repository's `data/` files are updated later, re-run the SHA comparison in `panel_manifest.json` and the value comparisons in §1–§3 of this document.
