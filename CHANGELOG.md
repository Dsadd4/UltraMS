# Changelog

All notable changes to UltraMS are documented here. The package follows semantic
versioning; the GitHub releases correspond to the PyPI versions.

## Unreleased

- Benchmark packages for spectral properties and molecular identification, including saved
  outputs, metric recalculation scripts and the task documentation.
- Public benchmark releases: the UltraMS MSnLib benchmark (formula-level 80/10/10 and
  strict-adduct molecule-level 70/15/15 partitions; Zenodo DOI 10.5281/zenodo.23963489) and
  the UltraMS SpectraVerse dataset (molecule-level 8:1:1 split; Zenodo DOI
  10.5281/zenodo.23964105).
- Manuscript figure source data and rendered panels for Figures 2 and 3 with per-file
  provenance ledgers.
- Documentation: API/CLI reference, FAQ, contributing guide and repository layout.

## 0.3.0 — 2026-09-24

- Researcher tutorials: MGF embeddings, PyTorch fine-tuning and spectral search notebooks.
- A MassSpecGym contrastive tutorial that keeps the released folds and evaluates the held-out
  test fold after reloading the selected checkpoint.
- Verification of all three public checkpoints against the installable package.

## 0.2.0 — 2026-09-24

- MGF reading with the base install and mzML with `ultrams[io]`.
- Ordered embedding export with `ultrams-embed` and `model.encode_batch()`.
- Complete examples for batch embeddings, PyTorch fine-tuning and spectrum similarity search.
- Documented checkpoint dimensions and input limits for `unsupervised`, `mona` and `search`.

## 0.1.3 — 2026-09-24

- Clear installation, embedding and fine-tuning instructions; aligned model cards.

## 0.1.2 — 2026-09-23

- Runnable README examples with self-contained sample spectra.

## 0.1.1 — 2026-09-23

- `pip install ultrams` release with the Unsupervised, MoNA contrastive and Search models.
- PyTorch-module interface and one-call fine-tuning.

## 0.1.0 — 2026-09-23

- Initial UltraMS inference package and pretraining source.
