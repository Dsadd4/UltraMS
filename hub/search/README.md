---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS Search

This spectrum-to-spectrum contrastive model provides the representation used to construct UltraAtlas. `encode(...).embedding` returns the embedding projection of the encoder's CLS embedding.

```bash
python -m pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("search")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
```

Fine-tuning examples are in the [UltraMS repository](https://github.com/Dsadd4/UltraMS).
