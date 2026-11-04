# Paper figures — plotting code and source data

This directory contains the **figure-facing layer** of the UltraMS manuscript: the
plotting programs, the numeric tables they read, the rendered panels, and a
file-level provenance ledger for each figure. It is packaged from the internal
figure-selection workspaces at the code versions used for the submitted manuscript
figures.

The data-generation, training and metric-recalculation code stays in
[`benchmarks/`](../benchmarks/README.md); `paper_figures/` only contains the code
that turns those benchmark outputs into the published panels.

| Directory | Manuscript figure | Main contents |
| --- | --- | --- |
| [`figure2/`](figure2/README.md) | Figure 2 (panels a–j) | 28 plotting/rendering scripts, 57 small source-data files, 23 rendered panel images |
| [`figure3/`](figure3/README.md) | Figure 3 (printed panels a–f; internal slots a–i) and the supplementary comparison packages (NPLIB1, MassSpecGym, MSnLib, source fusion, label-blind consensus, FDR threshold transfer, cases) | 27 plotting scripts, 173 source-data files, 35 rendered panels, plus `version_audit.md` |

## Conventions

Each figure directory follows the same layout and rules:

```text
figures/<figure>/
├── README.md            panel-by-panel map: content, plotting script, source data,
│                        data-generation code in benchmarks/, re-render command
├── panel_manifest.json  per-file ledger: panel, source path, target path, bytes,
│                        sha256, reused_or_copied, notes, and skipped large inputs
├── scripts/<panel>/     plotting scripts, original file names preserved
├── data/<panel>/        small source-data tables read by the scripts
└── outputs/<panel>/     rendered panel images (PNG/PDF, SVG where available)
```

- Panel directories are named with the **manuscript panel letter**, even where the
  internal selection identifiers differ (Figure 2 selection IDs `fig2g`/`fig2h`/
  `fig2f`/`fig2j`/`fig2k` correspond to manuscript panels f/g/h/i/j). The mapping
  is given in each README and in `panel_manifest.json`.
- Files are copied byte-for-byte from the selected figure snapshot, except that
  internal absolute paths were replaced by explicit placeholders
  (`<WORKSPACE_ROOT>/`, `<REMOTE_WORKSPACE_ROOT>/`); every such replacement is
  recorded in `panel_manifest.json`.
- Large evaluation arrays and model weights are **not redistributed**. When a
  script needs one, the README states where it comes from (public downloads in
  [`benchmarks/*/INPUTS.md`](../benchmarks/README.md) or the
  [UltraMS benchmark assets](https://huggingface.co/datasets/dsadd4/UltraMS-Benchmark-Assets)).
- No internal absolute paths, credentials or private machine names are included.

## Version fidelity

The packages are built from the authoritative figure-selection manifests
(`selection_map.json`, `*_main_select_manifest.json`, `supp/mapping_manifest.csv`),
not from re-drawn approximations.

- [`figure3/version_audit.md`](figure3/version_audit.md) compares every packaged
  script and table against the current public benchmark code and classifies it as
  byte-identical, numerically identical, a different experiment version, or
  missing from the public repository. Two points matter for reproduction:
  - The printed Figure 3e simple-encoder curves come from an **earlier readout
    version** than the s2s-readout currently saved in
    `benchmarks/molecular_identification/`; the UltraMS and DreaMS curves are the
    same. Redraw Figure 3e from this package's `data/fig3e/`.
  - The historical Figure 3h panel is a seven-method panel that includes a raw
    cosine baseline; the public benchmark data contain six methods. This is
    documented rather than silently reconciled.
- The Figure 2 package records any registered panel whose rendering in the
  selection workspace predates the printed figure (e.g. superseded R² values), so
  that the printed PDF remains the reference.

## Figure 3 library-search benchmark split

The Figure 3e/3h library-search benchmark is evaluated on a **molecule-level
disjoint split** of the strict `[M+H]+` / `[M-H]-` MSnLib subsets (70% / 15% / 15%
of compounds by canonical isomeric SMILES; 6,202 + 3,193 test queries against
295,434 + 137,000 library spectra). This split is distributed separately from this
repository together with its protocol, verification record and provenance;
a persistent identifier will be linked here once the dataset record is published.
It is **not** the formula-level public dataset `FOLD` split (446,667 / 55,980 / 57,437
spectra) that the repository-wide MSnLib tables use; the release documents the
distinction explicitly.

## Licensing

The UltraMS repository is released under Apache-2.0. Figure data inherit the
licenses of their public sources (for example MIT for MassSpecGym and MAGMa-derived
support code, CC BY 4.0 for MSnLib spectra); the notices are kept with the
corresponding benchmark and dataset releases.
