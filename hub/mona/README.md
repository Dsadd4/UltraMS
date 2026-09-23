---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS MoNA

UltraMS spectrum encoder with a contrastive projection trained on MoNA spectra. Use its projected embedding for spectrum similarity.

- **Input:** fragment `m/z` and intensity arrays, plus precursor `m/z`.
- **Output:** a projected spectrum embedding (`embedding.embedding`); the `ultrams` package also exposes CLS, peak-weighted, and fused embeddings.
- **Package:** `pip install ultrams`; load a downloaded checkpoint with `UltraMS.from_checkpoint(path)`.
- **Source checkpoint:** `Light_ultra/train/output/comparison/dreams_contrastive_ultrams_finetune/u4_e3_top100_seed3407_20260702/best.pt`.
- **Expected source SHA256:** `3b8ae5bd85ff8f6991f8a78b6d70531aab74f32914878730e7296aaddb79afb0`.
- **License:** Apache-2.0.
