---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS RT-only

UltraMS spectrum encoder after pretraining and RT/ion-mode adaptation. Use it to obtain a general representation of an MS/MS spectrum.

- **Input:** fragment `m/z` and intensity arrays, plus precursor `m/z`.
- **Output:** a spectrum embedding (`embedding.embedding`); the `ultrams` package also exposes CLS, peak-weighted, and fused embeddings.
- **Package:** `pip install ultrams`; load a downloaded checkpoint with `UltraMS.from_checkpoint(path)`.
- **Source checkpoint:** `Light_ultra/train/output/phase2_rt_only/stage_d_epoch_11.pt`.
- **Expected source SHA256:** `6a4c6660999848c409303119f6caa54fbae9444d0b75c7fcd8bafd303cde9830`.
- **License:** Apache-2.0.
