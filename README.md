# UltraMS

**Pretrained models for tandem mass spectra.** Turn MS/MS spectra into embeddings, compare spectra, or fine-tune the encoder for a measured property.

```bash
python -m pip install ultrams
```

| I want to… | Start here |
| --- | --- |
| Embed one spectrum | [Python example](#embed-a-spectrum) |
| Embed an MGF or mzML file | [File command](#embed-a-spectrum-file) |
| Train on my labelled spectra | [PyTorch example](#fine-tune-with-pytorch) · [complete MGF example](https://github.com/Dsadd4/UltraMS/blob/main/examples/pytorch_finetune.py) |
| Compare spectra | [Search example](https://github.com/Dsadd4/UltraMS/blob/main/examples/spectrum_search.py) |
| Learn in a notebook | [UltraMS tutorials](https://github.com/Dsadd4/UltraMS/tree/main/cookbook/tutorials) |
| Inspect or run pretraining | [Training code](https://github.com/Dsadd4/UltraMS/blob/main/training/README.md) |

## Embed a spectrum

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0],
    intensity=[20, 100, 35],
    precursor_mz=301.2,
).embedding
print(embedding.shape)  # (1024,)
```

The first call downloads the selected checkpoint from Hugging Face. Use `device="cuda"` or `device="mps"` with `from_pretrained` to run on an available accelerator.

The three checkpoints are collected on [Hugging Face](https://huggingface.co/collections/dsadd4/ultrams-6ab4cfaa860cbbd1f9a6b9b6).

## Embed a spectrum file

```bash
ultrams-embed spectra.mgf embeddings.npz --model unsupervised
```

The output has aligned `ids` and `embeddings` arrays. To use mzML, first run `python -m pip install 'ultrams[io]'`; the base installation reads MGF and `.mgf.gz` without an extra file-format package. The [checked-in example](https://github.com/Dsadd4/UltraMS/tree/main/examples) includes five real MS/MS spectra and a complete Python batch workflow.

## Fine-tune with PyTorch

`UltraMS` is a `torch.nn.Module`. Each dataset item contains peak `mz`, peak `intensity`, `precursor_mz`, and a measured numeric `target`. The two rows below are runnable example data; substitute your measured spectra and targets for research.

```python
import torch
from torch.utils.data import DataLoader
from ultrams import UltraMS

dataset = [
    {"mz": [100.1, 121.1, 150.0], "intensity": [20, 100, 35], "precursor_mz": 301.2, "target": 0.5},
    {"mz": [102.1, 135.2, 167.3], "intensity": [40, 100, 25], "precursor_mz": 315.3, "target": 0.7},
]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UltraMS.from_pretrained("unsupervised", device=device).train()
head = torch.nn.Linear(model.embedding_dim, 1).to(device)
loader = DataLoader(dataset, batch_size=2, collate_fn=model.batch_converter())
optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-5)

for batch in loader:
    batch = {name: value.to(device) for name, value in batch.items()}
    prediction = head(model(batch["peaks"], batch["attention_mask"], batch["precursor_mz"]))
    loss = torch.nn.functional.mse_loss(prediction.squeeze(-1), batch["target"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss: {loss.item():.4f}")
```

The [complete fine-tuning example](https://github.com/Dsadd4/UltraMS/blob/main/examples/pytorch_finetune.py) reads an MGF file and saves training history. For a shorter route, use `model.finetune(labelled_spectra, task="regression")` with a `label` field; it saves weights, configuration, and losses. See [spectrum input](https://github.com/Dsadd4/UltraMS/blob/main/docs/data-format.md) for the input format.

For measured molecular labels, the [MassSpecGym contrastive tutorial](https://github.com/Dsadd4/UltraMS/blob/main/cookbook/tutorials/MASSSPECGYM_CONTRASTIVE.md) downloads the dataset, keeps its train/validation/test folds, fine-tunes the encoder, and reloads the selected checkpoint.

## Choose a model

| Model name | Representation | Dimension | Spectral peak limit | Weights |
| --- | --- | ---: | ---: | --- |
| `unsupervised` | Encoder spectrum embedding | 1024 | 150 | [Unsupervised](https://huggingface.co/dsadd4/UltraMS-Unsupervised) |
| `mona` | MoNA contrastive projection | 1024 | 100 | [MoNA contrastive](https://huggingface.co/dsadd4/UltraMS-MoNA-Contrastive) |
| `search` | UltraAtlas search projection | 512 | 150 | [Search](https://huggingface.co/dsadd4/UltraMS-Search) |

Load any of them with `UltraMS.from_pretrained("name")`, or load a downloaded `model.pt` with `UltraMS.from_checkpoint(path)`. [Model selection](https://github.com/Dsadd4/UltraMS/blob/main/docs/model-selection.md) explains the training purpose and output of each checkpoint. The Search checkpoint is the **UltraAtlas application `best.pt`**.

## Pretraining

The [UltraMSdata pretraining package](https://github.com/Dsadd4/UltraMS/blob/main/training/README.md) contains the current model architecture, training entry points, configuration and history outputs. It installs separately from the inference package.

## Cite

Citation metadata for this software is in [CITATION.cff](https://github.com/Dsadd4/UltraMS/blob/main/CITATION.cff).
