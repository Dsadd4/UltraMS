---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS Search

The spectrum-to-spectrum contrastive model selected for the UltraAtlas application (`best.pt`). `encode(...).embedding` returns its normalized 512-dimensional projection of the UltraMS encoder's CLS embedding. Use this representation to rank spectra in a supplied reference library by cosine similarity.

| Output | Value |
| --- | --- |
| Embedding dimension | 512 |
| Maximum spectral peaks | 150 |
| Python model name | `"search"` |

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("search")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
print(embedding.shape)  # (512,)
```

Supply measured spectral peak `m/z` and intensity arrays plus precursor-ion `m/z`. UltraMS requires at least three spectral peaks with positive `m/z`, normalizes intensities by their maximum when positive, and retains the 150 most intense spectral peaks when needed. See the [input format](https://github.com/Dsadd4/UltraMS/blob/main/docs/data-format.md), [model selection](https://github.com/Dsadd4/UltraMS/blob/main/docs/model-selection.md), and [spectrum search example](https://github.com/Dsadd4/UltraMS/blob/main/examples/spectrum_search.py).
