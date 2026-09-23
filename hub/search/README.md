---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS Search

The final UltraMS model for spectrum similarity search, used in UltraAtlas.

```bash
pip install ultrams
```

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("search")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding
```

Fine-tuning examples are in the [UltraMS repository](https://github.com/Dsadd4/UltraMS).
