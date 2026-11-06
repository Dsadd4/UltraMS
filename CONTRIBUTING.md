# Contributing to UltraMS

Thanks for helping improve UltraMS. This repository contains the inference package,
pretraining code, benchmark code, and the source data behind the manuscript figures.

## Development setup

```bash
git clone https://github.com/Dsadd4/UltraMS.git
cd UltraMS
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The checkpoint verification test (`tests/check_released_models.py`) downloads the three
released checkpoints from Hugging Face and is also run by the PyPI publish workflow.

## What to work on

- **Bug reports** — use the bug report issue template. Include the package version
  (`python -c "import ultrams; print(ultrams.__version__)"`), Python version, OS, device,
  and a minimal reproduction (spectrum input or file excerpt).
- **Features** — open an issue describing the use case before a large pull request.
- **Benchmarks** — benchmark results belong in `benchmarks/` with their saved outputs and a
  `reproduce.py` path; see `benchmarks/README.md` for the conventions and the input
  licensing requirements.
- **Documentation and tutorials** — `docs/`, `examples/`, and `cookbook/` are good starting
  points.

## Data and repository policy

- Large spectra, embeddings, model weights and evaluation arrays are not committed to Git.
  Use the released Hugging Face datasets and checkpoint repositories and document the
  source, byte size and SHA-256 in the relevant README.
- Do not add credentials, private hosts or absolute paths.
- Third-party code keeps its original license; add the license text next to the code.

## Pull requests

1. Keep changes focused and include or update tests where behavior changes.
2. Run the test suite locally (`python -m unittest discover -s tests -v`).
3. Update documentation and `CHANGELOG.md` for user-visible changes.
4. Describe the motivation, the change and how it was verified.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
