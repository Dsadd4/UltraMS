---
license: apache-2.0
tags:
  - mass-spectrometry
  - metabolomics
  - spectrum-embedding
---

# UltraMS UltraAtlas

The contrastive UltraMS checkpoint used by the Figure 5 UltraAtlas application. Use its projected embedding for Atlas spectrum similarity.

- **Input:** fragment `m/z` and intensity arrays, plus precursor `m/z`.
- **Output:** a projected spectrum embedding (`embedding.embedding`); the `ultrams` package also exposes CLS, peak-weighted, and fused embeddings.
- **Load:** `pip install "ultrams[hub]"`; `UltraMS.from_hub("dsadd4/UltraMS-UltraAtlas")`.
- **Source checkpoint:** `Light_ultra/output/appliedmodel/spectrum_search_supcon/ultra_u2_p512_v4_seed42/best.pt`.
- **Expected source SHA256:** `447e41e99e5f6ee4155f5dd9938bd7b5fe3f09239b6d5eb0f4973b44c963e383`.
- **License:** Apache-2.0.
