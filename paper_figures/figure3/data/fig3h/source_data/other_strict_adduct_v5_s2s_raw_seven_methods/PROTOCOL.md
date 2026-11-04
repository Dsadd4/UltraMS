# Figure 3h additional-adduct spectrum-to-spectrum evaluation

The four simple learned controls (Linear, DeepSets, Fourier and Codebook) were fitted separately by ion mode using two distinct spectra from the same training molecule as positives under a symmetric in-batch InfoNCE objective. They were not trained against fingerprints, molecular embeddings or ChemBERTa. DreaMS and UltraMS use their frozen spectrum encoders. Query and library spectra are compared directly by cosine similarity between spectrum embeddings.

Raw cosine uses MatchMS `CosineGreedy` directly on the frozen top-150, base-peak-normalized peak lists (0.02 Da tolerance, `mz_power=0`, `intensity_power=1`, no precursor shift). It has no learned parameters. Every method uses the same exact-adduct query and full-library rows, and a retrieval is correct only when the top-ranked library spectrum has the same canonical isomeric SMILES as the query.
