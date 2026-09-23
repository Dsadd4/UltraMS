# Train UltraMS

The current UltraMS pretraining code and configurations are provided here for researchers training on their own MS/MS corpus.

```bash
pip install -e ./training
ultrams-pretrain \
  --config training/ultrams_training/configs/pretraining_pure_ae3_mpr5.json \
  --project-root training/ultrams_training/project_snapshot \
  --data-root /path/to/data \
  --end-stage peak_reconstruction
```

The RT and ion-mode training configuration is in `configs/pretraining_pure_ae3_full_adaptation.json`. Fine-tuning runs save their model, configuration, and loss history; `python training/plot_finetune_history.py <run_dir>` plots the saved history.
