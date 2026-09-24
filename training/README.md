# UltraMS pretraining

The bundled configuration runs masked peak reconstruction (MPR) on UltraMSdata (160,641,162 spectra in this configuration).

```bash
pip install -e ./training
ultrams-pretrain \
  --config training/ultrams_training/configs/ultrams_pretraining.json \
  --project-root training/ultrams_training/model \
  --data-root /path/to/data \
  --end-stage peak_reconstruction
```
