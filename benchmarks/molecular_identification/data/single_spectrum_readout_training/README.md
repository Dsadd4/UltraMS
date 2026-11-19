# Figure 3a readout training records

These are the four manuscript ChemBERTa readouts: Linear, DeepSets, Fourier and Codebook. Each uses seed 0, 30 training epochs, and the checkpoint with the lowest full-validation paired cosine loss. The original DeepSets run selected epoch 28.

All methods evaluate the same 57,437 test spectra and frozen candidate lists. The selected weights are downloaded by `run_public_single_spectrum_reported_six_methods.sh`.

| Method | Test Top-1 (%) |
| --- | ---: |
| Linear | 17.915281 |
| DeepSets | 5.686230 |
| Fourier | 14.992078 |
| Codebook | 15.911346 |
