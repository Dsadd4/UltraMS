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

`UltraMS` is a `torch.nn.Module`. The two rows below show the required data format; replace them with your labelled spectra.

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

## Fine-tune in one call

For a shorter route, provide the same spectra with a `label` field.

```python
from ultrams import UltraMS

labelled_spectra = [
    {"mz": [100.1, 121.1, 150.0], "intensity": [20, 100, 35], "precursor_mz": 301.2, "label": 0.5},
    {"mz": [102.1, 135.2, 167.3], "intensity": [40, 100, 25], "precursor_mz": 315.3, "label": 0.7},
]

model = UltraMS.from_pretrained("unsupervised")
predictor = model.finetune(labelled_spectra, task="regression", epochs=1)
prediction = predictor.predict([100.1, 121.1, 150.0], [20, 100, 35], precursor_mz=301.2)
print(prediction)
```

Use `task="classification"` for class labels. Fine-tuning saves the model, training configuration, and loss history in `ultrams_finetune/`. A downloaded `model.pt` can be loaded with `UltraMS.from_checkpoint(path)`.

The current pretraining code is in [training](https://github.com/Dsadd4/UltraMS/tree/main/training).
