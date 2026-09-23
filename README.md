# UltraMS

Pretrained models for MS/MS spectra.

```bash
pip install ultrams
```

| Model | Use | Weights |
| --- | --- | --- |
| **Unsupervised** | General spectrum embeddings | [Hugging Face](https://huggingface.co/dsadd4/UltraMS-Unsupervised) |
| **MoNA Contrastive** | Spectrum similarity learned on MoNA | [Hugging Face](https://huggingface.co/dsadd4/UltraMS-MoNA-Contrastive) |
| **Search** | Final spectrum search model | [Hugging Face](https://huggingface.co/dsadd4/UltraMS-Search) |

## Get an embedding

```python
from ultrams import UltraMS

model = UltraMS.from_pretrained("unsupervised")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding  # NumPy vector
```

## Fine-tune with PyTorch

`UltraMS` is a `torch.nn.Module`. Each item in `dataset` contains `mz`, `intensity`, `precursor_mz`, and a numeric `target`. The batch converter prepares variable-length spectra for `DataLoader`.

```python
import torch
from torch.utils.data import DataLoader
from ultrams import UltraMS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UltraMS.from_pretrained("unsupervised", device=device).train()
head = torch.nn.Linear(model.embedding_dim, 1).to(device)
loader = DataLoader(dataset, batch_size=16, collate_fn=model.batch_converter())
optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-5)

for batch in loader:
    batch = {name: value.to(device) for name, value in batch.items()}
    prediction = head(model(batch["peaks"], batch["attention_mask"], batch["precursor_mz"]))
    loss = torch.nn.functional.mse_loss(prediction.squeeze(-1), batch["target"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Fine-tune in one call

Each item in `labelled_spectra` contains `mz`, `intensity`, `precursor_mz`, and `label`.

```python
model = UltraMS.from_pretrained("unsupervised")
predictor = model.finetune(labelled_spectra, task="regression", epochs=5)
prediction = predictor.predict([100.1, 121.1, 150.0], [20, 100, 35], precursor_mz=301.2)
```

Use `task="classification"` for class labels. Fine-tuning saves the model, training configuration, and loss history in `ultrams_finetune/`. A downloaded `model.pt` can be loaded with `UltraMS.from_checkpoint(path)`.

The current pretraining code is in [training](https://github.com/Dsadd4/UltraMS/tree/main/training).
