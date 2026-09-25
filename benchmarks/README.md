# UltraMS benchmarks

The benchmark inputs are publicly available. From a fresh checkout, the task commands download spectra, candidate tables, and starting weights, then build or download feature caches in a workspace you choose. You do not need files from our internal servers. Each task page provides its training, test, and result-recalculation commands and states which full runs have been verified.

| Benchmark | Public inputs | Train and evaluate | Results and reproduction |
| --- | --- | --- | --- |
| Spectral properties | [Download and prepare](spectral_properties/INPUTS.md) | [Task commands](spectral_properties/EXPERIMENTS.md) | [Results](spectral_properties/README.md) |
| Molecular identification | [Download and prepare](molecular_identification/INPUTS.md) | [Task commands](molecular_identification/EXPERIMENTS.md) | [Results](molecular_identification/README.md) |

NPLIB1 and MassSpecGym use predicted 2,048-bit Morgan molecular fingerprints to rank candidate molecules. Their [training and download commands](molecular_identification/README.md#fingerprint-based-molecular-identification), [saved test-query results](molecular_identification/data/fingerprint_prediction/), and [result calculator](molecular_identification/reproduce_fingerprint_prediction.py) are included.

To recalculate the reported tables from saved test outputs without training:

```bash
python -m pip install numpy scikit-learn
python benchmarks/spectral_properties/reproduce.py
python benchmarks/molecular_identification/reproduce.py
```
