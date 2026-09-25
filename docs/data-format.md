# Spectrum input

UltraMS accepts an MS/MS spectrum as two equal-length, one-dimensional arrays and the precursor ion's measured `m/z`:

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised")
result = model.encode(
    mz=[100.1, 121.1, 150.0],
    intensity=[20, 100, 35],
    precursor_mz=301.2,
)
print(result.embedding.shape)  # (1024,)
```

`mz` and `intensity` describe the same spectral peaks in the same order. `precursor_mz` is a positive, finite number. Every spectral peak `m/z` and intensity must be finite. The input must contain at least three spectral peaks with positive `m/z` after preprocessing.

## Preprocessing

`encode()` and `batch_converter()` apply the same preparation:

1. Discard spectral peaks with `m/z <= 0`.
2. Divide intensities by the largest retained intensity when it is positive, then clip intensities to `[0, 1]`.
3. If the spectrum exceeds the model's limit, keep the most intense spectral peaks.
4. Sort the retained spectral peaks by `m/z`.

The three released models accept up to **150** spectral peaks for `"unsupervised"`, **100** for `"mona"`, and **150** for `"search"`. You can inspect the loaded model's limit with `model.input_max_peaks`. A batch made with `model.batch_converter()` pads shorter spectra and supplies an `attention_mask`; padding does not count as a measured spectral peak.

For a PyTorch `Dataset`, return one mapping per spectrum:

```python
{
    "mz": [100.1, 121.1, 150.0],
    "intensity": [20, 100, 35],
    "precursor_mz": 301.2,
    "target": 0.5,
}
```

The `target` field is a numeric training value used by `model.batch_converter()` when present in every item of a batch. For `model.finetune(...)`, use `"label"` instead; see [PyTorch fine-tuning](../examples/pytorch_finetune.py) and the [README](../README.md).

`model.encode(...).embedding` is a NumPy vector for inference. `model(peaks, attention_mask, precursor_mz)` returns a `[batch, embedding_dim]` PyTorch tensor with gradients for training. See [model selection](model-selection.md) for the representation returned by each checkpoint.

To inspect representations for individual spectral peaks, pass `return_peaks=True`:

```python
result = model.encode(
    mz=[150.0, 100.1, 121.1],
    intensity=[35, 20, 100],
    precursor_mz=301.2,
    return_peaks=True,
)
print(result.peak_mz)                 # processed peak m/z, sorted ascending
print(result.peak_embeddings.shape)  # (3, 1024) for the Unsupervised model
```

`peak_embeddings[i]` is the encoder's final contextual representation for `peak_mz[i]` and `peak_intensity[i]`. These arrays reflect the peak selection, intensity normalization and sorting described above. They are returned only when requested.

## Spectrum files

For MGF or `.mgf.gz`, UltraMS reads each `BEGIN IONS` block, uses `PEPMASS` for precursor-ion `m/z`, and takes the spectrum ID from `TITLE`, `NAME`, or `SCANS` when available. Extra columns after a spectral peak's `m/z` and intensity are ignored. The optional `ultrams[io]` installation adds mzML input; only MS2 spectra are read.

```bash
ultrams-embed examples/data/example_5_spectra.mgf embeddings.npz --model search
```

The `.npz` output contains aligned `ids` and `embeddings` arrays. See the [batch embedding example](../examples/batch_embeddings.py) to use the file reader and model directly in Python.
