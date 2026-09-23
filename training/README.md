# UltraMS pretraining

The `ultrams_training` package contains the current pure Ae3 pretraining source. The model curriculum is MPR 5 epochs, supervised RT 5 epochs, then ion mode 11 epochs. The four and six GPU profiles preserve the configured global batch sizes.

```bash
pip install -e ./training
ultrams-pretrain --config training/ultrams_training/configs/pretraining_pure_ae3_mpr5.json --project-root training/ultrams_training/project_snapshot --data-root /path/to/data --validate-only
```

The MPR config is the exact frozen `pure_ae3_20260920` configuration (SHA256 `fbb5aede99cfd4cc8931e44b3e660641995f10ed283761ed6e21ab1c1a86de4e`). `pretraining_pure_ae3_full_adaptation.json` contains the RT and ion mode program; its parent checkpoint and data fingerprints are bound when the MPR stage completes.
