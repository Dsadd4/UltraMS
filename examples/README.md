# Run UltraMS on MS/MS spectra

These examples use five spectra from the [DreaMS project](data/NOTICE.md). Install UltraMS from the repository root first:

```bash
python -m pip install -e .
```

| Task | Run |
| --- | --- |
| Read an MGF file and save its embeddings | `python examples/batch_embeddings.py` |
| Fine-tune with a native PyTorch loop | `python examples/pytorch_finetune.py` |
| Rank spectra by cosine similarity | `python examples/spectrum_search.py` |

Each command downloads its pretrained checkpoint from Hugging Face on first use. `batch_embeddings.py` saves an `.npz` containing spectrum IDs and one embedding per input spectrum. `spectrum_search.py` saves a CSV of ranked matches; its scores illustrate spectrum similarity and are not a benchmark. `pytorch_finetune.py` uses **placeholder regression targets based on file order** solely to show how to train the encoder and a task head. Replace these targets with measured labels for research. By default, output files go to a newly created temporary directory printed by each command; use `--output` or `--output-dir` to choose a persistent destination.

The PyTorch example saves both the encoder and task head in `model.pt`. Load them for prediction with:

```python
import torch
from ultrams import UltraMS

weights = torch.load("/path/to/model.pt", map_location="cpu", weights_only=True)
model = UltraMS.from_pretrained("unsupervised").eval()
head = torch.nn.Linear(model.embedding_dim, 1).eval()
model.load_state_dict(weights["encoder"])
head.load_state_dict(weights["head"])
```
