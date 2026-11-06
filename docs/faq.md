# FAQ and troubleshooting

## Installation

**`pip install ultrams` is slow or fails while building dependencies.** UltraMS depends on
`torch` and `transformers`. Install a matching PyTorch build for your platform first
(for example `pip install torch --index-url https://download.pytorch.org/whl/cu124`), then
run `pip install ultrams`.

**Which Python versions are supported?** Python 3.10 or newer; CI tests 3.10–3.13 on Linux.

## Checkpoints

**The first call hangs or fails.** `from_pretrained` downloads the checkpoint (~0.8–1.2 GB)
from Hugging Face. Check network access, `HF_HOME`/cache permissions, and
`hf auth whoami` if the repository is not public. The three repositories are public.

**Can I use a local checkpoint?** Yes: `UltraMS.from_checkpoint("model.pt")` loads any
released checkpoint (and compatible fine-tuned checkpoints) without network access.

## Input

**`ValueError: a spectrum needs at least three peaks with positive m/z`.** The model input
requires at least three peaks after removing non-positive `m/z` values. This is a model-input
requirement, not a file-format error.

**Peaks are silently dropped.** Spectra longer than the checkpoint limit
(150 peaks for `unsupervised`/`search`, 100 for `mona`) are truncated to the most intense
peaks. Check `model.input_max_peaks`.

**mzML is not supported.** Install the optional dependency: `pip install 'ultrams[io]'`.
MGF and `.mgf.gz` work with the base install.

**Intensities are not on the same scale as my file.** `encode()` normalizes intensities to
the base peak and clips them to `[0, 1]`; see [spectrum input](data-format.md).

## Devices

**CUDA out of memory during batch encoding.** Reduce `batch_size`
(`encode_batch(..., batch_size=8)`) or use `--batch-size` in `ultrams-embed`.

**Apple Silicon.** Pass `device="mps"`. If your PyTorch build has no MPS support, use CPU.

## Fine-tuning

**`ValueError: classification labels must be strings or integers`.** Classification labels
must be strings or integers (booleans are rejected); use `task="regression"` for numeric
targets.

**Where are the training artifacts?** `model.finetune(..., output_dir=...)` saves weights,
configuration and `history` in that directory; without `output_dir`, a timestamped
`ultrams_finetune/` directory is created.

## Reproducing benchmarks

The benchmark pages under [`benchmarks/`](../benchmarks/README.md) document public inputs,
training/test commands and the saved results. The `reproduce.py` scripts recalculate the
reported tables from the included saved test outputs without retraining. Full GPU training
runs are documented per task, including the parts that were verified.

## Citing and data

See [CITATION.cff](../CITATION.cff) for software citation. The released benchmark datasets
and their DOIs are listed in the repository README under "Data and models".
