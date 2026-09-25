# UltraMS tutorials

Each notebook runs from top to bottom on the [five-spectrum MGF example](../../examples/data/NOTICE.md). It installs UltraMS from PyPI and downloads the selected model from Hugging Face on first use.

| Tutorial | Notebook | Colab |
| --- | --- | --- |
| Read MGF and obtain embeddings | [Get embeddings](embed.ipynb) | [Open in Colab](https://colab.research.google.com/github/Dsadd4/UltraMS/blob/main/cookbook/tutorials/embed.ipynb) |
| Fine-tune in a PyTorch loop | [PyTorch fine-tuning](pytorch_finetune.ipynb) | [Open in Colab](https://colab.research.google.com/github/Dsadd4/UltraMS/blob/main/cookbook/tutorials/pytorch_finetune.ipynb) |
| Rank spectra with the Search model | [Spectrum search](spectrum_search.ipynb) | [Open in Colab](https://colab.research.google.com/github/Dsadd4/UltraMS/blob/main/cookbook/tutorials/spectrum_search.ipynb) |

The fine-tuning labels are placeholders for learning the API. They do not represent measured properties or a model performance result.

For a complete fine-tuning example with real molecular labels and the original dataset folds, follow the [MassSpecGym contrastive tutorial](MASSSPECGYM_CONTRASTIVE.md).
