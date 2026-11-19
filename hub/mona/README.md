---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS MoNA Contrastive

The UltraMS MS/MS encoder with a spectrum-level projection learned through contrastive training on MoNA spectra. `encode(...).embedding` returns the learned projection of the encoder's CLS embedding. Use cosine similarity to compare spectra in this space.

| Output | Value |
| --- | --- |
| Embedding dimension | 1024 |
| Maximum spectral peaks | 100 |
| Python model name | `"mona"` |

`model.pt` contains model weights and inference configuration and can be loaded with UltraMS 0.3.2.

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("mona")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
print(embedding.shape)  # (1024,)
```

Supply measured spectral peak `m/z` and intensity arrays plus precursor-ion `m/z`. UltraMS requires at least three spectral peaks with positive `m/z`, normalizes intensities by their maximum when positive, and retains the 100 most intense spectral peaks when needed. See the [input format](https://github.com/Dsadd4/UltraMS), [model selection](https://github.com/Dsadd4/UltraMS), and [spectrum search example](https://github.com/Dsadd4/UltraMS).

Use `return_peaks=True` in `model.encode(...)` to obtain final encoder embeddings aligned with the retained spectral peaks. The three released checkpoints are in the [UltraMS model family](https://huggingface.co/collections/dsadd4/ultrams-6ab4cfaa860cbbd1f9a6b9b6).
