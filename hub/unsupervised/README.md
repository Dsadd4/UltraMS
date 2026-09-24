---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS Unsupervised

The general UltraMS MS/MS encoder, released from the completed retention-time-only epoch 11 (`rtonly11`) training checkpoint. UltraMS learned from UltraMSdata through masked peak reconstruction (MPR), followed by retention-time training. This model returns the encoder's normalized spectrum-level CLS embedding.

| Output | Value |
| --- | --- |
| Embedding dimension | 1024 |
| Maximum spectral peaks | 150 |
| Python model name | `"unsupervised"` |
| Weights SHA-256 | `6a4c6660999848c409303119f6caa54fbae9444d0b75c7fcd8bafd303cde9830` |

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
print(embedding.shape)  # (1024,)
```

Supply measured spectral peak `m/z` and intensity arrays plus precursor-ion `m/z`. UltraMS requires at least three spectral peaks with positive `m/z`, normalizes intensities by their maximum when positive, and retains the 150 most intense spectral peaks when needed. See the [input format](https://github.com/Dsadd4/UltraMS/blob/main/docs/data-format.md), [model selection](https://github.com/Dsadd4/UltraMS/blob/main/docs/model-selection.md), and [PyTorch fine-tuning example](https://github.com/Dsadd4/UltraMS/blob/main/examples/pytorch_finetune.py).

Use `return_peaks=True` in `model.encode(...)` to obtain final encoder embeddings aligned with the retained spectral peaks. The three released checkpoints are in the [UltraMS model family](https://huggingface.co/collections/dsadd4/ultrams-6ab4cfaa860cbbd1f9a6b9b6).
