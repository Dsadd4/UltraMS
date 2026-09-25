# Choose a model

All three checkpoints use the UltraMS encoder. `UltraMS.from_pretrained(name)` downloads the selected weights from Hugging Face; `model.encode(...).embedding` returns the representation associated with that checkpoint.

| Model | Load with | Training | `embedding` | Dimension | Maximum spectral peaks |
| --- | --- | --- | --- | ---: | ---: |
| [Unsupervised](https://huggingface.co/dsadd4/UltraMS-Unsupervised) | `"unsupervised"` | Self-supervised pretraining on UltraMSdata, followed by retention-time training | Encoder spectrum-level CLS embedding | 1024 | 150 |
| [MoNA contrastive](https://huggingface.co/dsadd4/UltraMS-MoNA-Contrastive) | `"mona"` | Contrastive training on MoNA spectra | Projection of the encoder's CLS embedding | 1024 | 100 |
| [Search](https://huggingface.co/dsadd4/UltraMS-Search) | `"search"` | Spectrum-to-spectrum contrastive training for the UltraAtlas search model | Projection of the encoder's CLS embedding | 512 | 150 |

Use **Unsupervised** as the general starting point for embeddings or a new downstream task. Use **MoNA contrastive** for a representation trained on molecular-identity relationships in MoNA. Use **Search** when working in the spectrum similarity space used by the UltraAtlas application. A model alone does not include the UltraAtlas reference collection or its search index; the [spectrum search example](../examples/spectrum_search.py) shows how to rank a supplied spectral library.

The Unsupervised release comes from the completed retention-time-only epoch 11 checkpoint (`rtonly11`). The Search release is the `best.pt` selected for the UltraAtlas application. These source names identify the released weights; the public API uses the names in the table.

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("search")
vector = model.encode(
    mz=[100.1, 121.1, 150.0],
    intensity=[20, 100, 35],
    precursor_mz=301.2,
).embedding
print(vector.shape)  # (512,)
```

For every model, `result.cls` exposes the encoder's normalized CLS embedding. MoNA and Search also expose `result.projection`. The Unsupervised and Search default embeddings are unit-normalized; the MoNA default embedding is the learned projection before normalization. Use cosine similarity when comparing spectra. Input requirements and spectral peak preparation are in [Spectrum input](data-format.md).
