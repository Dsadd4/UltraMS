# Choose a model

All three checkpoints use the UltraMS encoder. `UltraMS.from_pretrained(name)` downloads the selected weights from Hugging Face; `model.encode(...).embedding` returns the representation associated with that checkpoint.

| Model | Load with | Training | `embedding` | Dimension | Maximum spectral peaks |
| --- | --- | --- | --- | ---: | ---: |
| [Unsupervised](https://huggingface.co/dsadd4/UltraMS-Unsupervised) | `"unsupervised"` | Self-supervised pretraining on UltraMSdata, followed by retention-time training | Encoder spectrum-level CLS embedding | 1024 | 150 |
| [MoNA contrastive](https://huggingface.co/dsadd4/UltraMS-MoNA-Contrastive) | `"mona"` | Contrastive training on MoNA spectra | Projection of the encoder's CLS embedding | 1024 | 100 |
| [Search](https://huggingface.co/dsadd4/UltraMS-Search) | `"search"` | Spectrum-to-spectrum contrastive training for the UltraAtlas search model | Projection of the encoder's CLS embedding | 512 | 150 |

Use **Unsupervised** as the general starting point for embeddings or a new downstream task. Use **MoNA contrastive** for a representation trained on molecular-identity relationships in MoNA. Use **Search** when working in the spectrum similarity space used by the UltraAtlas application. A model alone does not include the UltraAtlas reference collection or its search index; the [spectrum search example](../examples/spectrum_search.py) shows how to rank a supplied spectral library.

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

## Inference checkpoint files

The released `model.pt` files contain the encoder weights, model heads or projections, and the architecture and input configuration. They can be used directly with UltraMS 0.3.2. Training parameters, optimizer state and training history are stored separately from these inference files.

To export a training checkpoint while keeping the original file:

```bash
python tools/export_inference_checkpoint.py training_checkpoint.pt inference_model.pt
```

The export preserves the weight values and the existing input peak limits. The Unsupervised file includes both the full-model and backbone state dictionaries used by the public benchmark loaders.

Inference files also support downstream fine-tuning with the public PyTorch API. Exact pretraining continuation uses the original training checkpoint, which retains optimizer, scheduler, random-number-generator and progress state; see [checkpoint use](../training/README.md#checkpoint-use).
