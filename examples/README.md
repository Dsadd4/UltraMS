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
