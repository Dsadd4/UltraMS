# MAGMa annotation code

The `fragments/magma` source here is the MAGMa implementation used to produce the spectral-property benchmark annotations. Its fragmentation code descends from the MIT-licensed MAGMa implementation in [MIST](https://github.com/samgoldman97/mist/tree/main_v2/src/mist/magma), Copyright (c) 2022 Samuel Goldman. The original MIT terms are in [MIST_LICENSE.md](MIST_LICENSE.md).

UltraMS added the optimized `fragmentation_op_v8.py` path and `Magma4MassSpecGYm.py` interface for the benchmark spectra. The files are included so the annotation step can run from this repository.

This release removes an unused plotting import and optional parallel helper, replaces an old machine-specific import path, and keeps the fragment matching code used by the experiment.
