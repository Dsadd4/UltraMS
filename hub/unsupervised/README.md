---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS Unsupervised

Use this model to obtain a general spectrum-level embedding of an MS/MS spectrum. `encode(...).embedding` returns the encoder's CLS embedding.

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
```

Fine-tuning examples are in the [UltraMS repository](https://github.com/Dsadd4/UltraMS).
