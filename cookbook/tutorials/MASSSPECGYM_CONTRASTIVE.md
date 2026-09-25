# Contrastive fine-tuning with MassSpecGym

This tutorial fine-tunes the **UltraMS Unsupervised** encoder on MassSpecGym. Two distinct MS/MS spectra with the same `inchikey` are a positive pair; spectra from different molecules in a batch are negatives. The dataset's `train`, `val`, and `test` folds are retained, and the script checks that no molecular key appears in more than one fold. It keeps spectra with at least three positive-intensity, positive-m/z peaks and records the source and selected row and molecule counts in `data_audit.json`.

Install UltraMS, then run the small teaching subset from the repository root:

```bash
python -m pip install -e .
python cookbook/tutorials/massspecgym_contrastive.py --output-dir /path/to/my_run
```

The default selects up to 64 train molecules and 24 molecules from each evaluation fold, with two distinct spectra per molecule. Selection is deterministic. To use every training molecule with at least two distinct spectra and all valid evaluation spectra, after removing exact spectral duplicates:

```bash
python cookbook/tutorials/massspecgym_contrastive.py --full --output-dir /path/to/full_run
```

The full run can take substantially longer. `--data-file /path/to/MassSpecGym.tsv` reuses a local copy; otherwise the script downloads the pinned file through Hugging Face Hub. In both cases it checks SHA-256 before reading. The source is [`roman-bushuiev/MassSpecGym`](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym), revision `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`, file `data/MassSpecGym.tsv`, SHA-256 `0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95`.

The script writes `best.pt`, `config.json`, `data_audit.json`, `history.json`, and `test.json`. Validation selects the checkpoint; the held-out test fold is evaluated once afterward. Top1 means nearest *other spectrum* within the same fold has the same `inchikey`; singleton molecules are candidates but cannot be queries. This is a **tutorial retrieval task**, not an official MassSpecGym benchmark or a reported UltraMS result. The label is molecular identity from the dataset, not measured retention time or logP.

For the original dataset and citation, see [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym).
