---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS MoNA Contrastive

Use this model, contrastively trained on MoNA, for MS/MS spectrum similarity. `encode(...).embedding` returns the embedding projection of the encoder's CLS embedding.

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("mona")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
```

Fine-tuning examples are in the [UltraMS repository](https://github.com/Dsadd4/UltraMS).
