# Python API and command line

This page documents the public interface of the `ultrams` package. For input
requirements and preprocessing, see [spectrum input](data-format.md); for choosing a
checkpoint, see [model selection](model-selection.md).

## Load a model

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised", device="cpu")   # "unsupervised", "mona", "search"
model = UltraMS.from_checkpoint("model.pt", device="cpu")       # local checkpoint
model = UltraMS.from_hub("dsadd4/UltraMS-Search", revision="main")
```

`UltraMS` is a `torch.nn.Module`. Useful attributes:

| Attribute | Meaning |
| --- | --- |
| `embedding_dim` | Output dimension (1024 for `unsupervised`/`mona`, 512 for `search`) |
| `input_max_peaks` | Maximum retained peaks (150, 100, 150 respectively) |
| `model_family` | `unsupervised`, `mona` or `atlas` |
| `config` | Model configuration used to build the architecture |

## Encode a spectrum

```python
result = model.encode(
    mz=[100.1, 121.1, 150.0],
    intensity=[20, 100, 35],
    precursor_mz=301.2,
    return_peaks=False,
)
embedding = result.embedding          # model-specific representation (NumPy array)
```

`encode()` returns a `SpectrumEmbedding` with these fields:

| Field | Meaning |
| --- | --- |
| `cls` | Normalized encoder CLS embedding |
| `peak_intensity_weighted` | Intensity-weighted peak-embedding average |
| `fusion` | Concat/aggregate representation used by the released head |
| `projection` | Projection output for `mona`/`search`, otherwise `None` |
| `embedding` | `projection` when present, otherwise `cls` |
| `rt_seconds` | Predicted retention time when the checkpoint provides one |
| `peak_embeddings`, `peak_mz`, `peak_intensity` | Only with `return_peaks=True`; aligned peak-level outputs |

## Batch encoding

```python
spectra = [
    {"id": "a", "mz": [...], "intensity": [...], "precursor_mz": 301.2},
    {"id": "b", "mz": [...], "intensity": [...], "precursor_mz": 315.3},
]
vectors = model.encode_batch(spectra, batch_size=32)   # shape (N, embedding_dim)

loader = torch.utils.data.DataLoader(spectra, batch_size=32, collate_fn=model.batch_converter())
```

`encode_batch` validates inputs, pads batches, and returns vectors in input order. For a
custom training loop, use `model.batch_converter()` (or `model.collate(...)`) and the
differentiable `model(peaks, attention_mask, precursor_mz)` forward call.

## Fine-tuning

```python
predictor = model.finetune(
    records,                      # each record: mz, intensity, precursor_mz, label
    task="regression",            # or "classification"
    epochs=5,
    batch_size=8,
    lr=1e-5,
    validation_records=None,
    output_dir=None,
    device=None,
    seed=42,
)
prediction = predictor.predict(mz=[100.1, 121.1], intensity=[20, 100], precursor_mz=301.2)
predictor.save_pretrained("my_predictor")
```

`finetune()` returns an `UltraMSPredictor` (encoder + task head), records per-epoch losses
in `predictor.history`, and saves configuration, weights and history when `output_dir` is
given. Reload a saved predictor with `UltraMSPredictor.from_pretrained(path)`.

## Read spectra

```python
from ultrams import read_spectra, read_mgf, read_mzml

for record in read_spectra("spectra.mgf"):
    print(record["id"], record["mz"], record["intensity"], record["precursor_mz"])
```

`read_spectra` dispatches on the file suffix (`.mgf`, `.mgf.gz`, `.mzML`). mzML requires
`pip install 'ultrams[io]'`. `read_mgf` and `read_mzml` read one format explicitly. The
readers yield `id`, `mz`, `intensity` and `precursor_mz`; malformed records raise
`ValueError` with the spectrum ID in the message.

## Command line

```bash
ultrams-embed INPUT OUTPUT.npz [--model unsupervised|mona|search] [--checkpoint PATH]
              [--batch-size N] [--device auto|cpu|cuda|mps]
```

`ultrams-embed` writes an `.npz` with aligned `ids` and `embeddings` arrays. `--device auto`
selects CUDA, then Apple MPS, then CPU.

## Version

```python
import ultrams
print(ultrams.__version__)
```
