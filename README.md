# UltraMS

UltraMS encodes tandem mass spectra into learned molecular representations.

```bash
pip install ultrams
```

## Models

| Model | Purpose |
| --- | --- |
| [RT-only D11](hub/rt-only/README.md) | General spectrum representation |
| [MoNA contrastive](hub/mona/README.md) | MoNA spectrum similarity |
| [UltraAtlas contrastive](hub/atlas/README.md) | Figure 5 Atlas application spectrum similarity |

```python
from ultrams import UltraMS

model = UltraMS.from_checkpoint("/path/to/checkpoint.pt")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0],
    intensity=[20, 100, 35],
    precursor_mz=301.2,
)
print(embedding.embedding.shape)
```

The pretraining source and its installation instructions are in [training/](training/).
