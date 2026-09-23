# UltraMS

Encode an MS/MS spectrum in three lines. Fine-tune the same encoder with your own labels.

```bash
pip install "ultrams[hub]"
```

```python
from ultrams import UltraMS

model = UltraMS.from_hub("dsadd4/UltraMS-RT-only")
embedding = model.encode(
    mz=[100.1, 121.1, 150.0], intensity=[20, 100, 35], precursor_mz=301.2
).embedding  # NumPy vector
```

## Fine-tune

`encode_tensor` keeps gradients. Add a task head and train the encoder and head together:

```python
import torch
from ultrams import UltraMS

model = UltraMS.from_hub("dsadd4/UltraMS-RT-only").train()
head = torch.nn.Linear(model.embedding_dim, 1)
optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-5)

for mz, intensity, precursor_mz, target in training_rows:
    prediction = head(model.encode_tensor(mz, intensity, precursor_mz=precursor_mz))
    loss = torch.nn.functional.mse_loss(prediction.flatten(), torch.tensor([target], dtype=torch.float32))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`training_rows` contains your labelled spectra. For MoNA or UltraAtlas fine-tuning, replace the model ID in the same code.

## Checkpoints

| Model | Hugging Face | Use |
| --- | --- | --- |
| RT-only D11 | [UltraMS-RT-only](https://huggingface.co/dsadd4/UltraMS-RT-only) | General MS/MS representation |
| MoNA contrastive | [UltraMS-MoNA](https://huggingface.co/dsadd4/UltraMS-MoNA) | MoNA spectrum similarity |
| UltraAtlas contrastive | [UltraMS-UltraAtlas](https://huggingface.co/dsadd4/UltraMS-UltraAtlas) | Figure 5 Atlas application |

All three can also be loaded from a downloaded `model.pt` with `UltraMS.from_checkpoint(path)`.

The current pure Ae3 pretraining and RT/ion-mode adaptation source is in [training/](training/).
