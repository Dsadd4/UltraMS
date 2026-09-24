# UltraMS pretraining

The bundled configuration trains masked peak reconstruction (MPR) on 160,641,162 spectra from UltraMSdata.

## Install

Use Python 3.11 on a Linux machine with four or six GPUs.

```bash
git clone https://github.com/Dsadd4/UltraMS.git
cd UltraMS
python -m pip install -e ./training
```

## Prepare the data

The configuration requires the prepared UltraMSdata Parquet shards and their manifest. Set `--data-root` to the parent of the `derived/` directory specified in [`ultrams_pretraining.json`](ultrams_training/configs/ultrams_pretraining.json). The training entry point verifies the dataset identity before starting.

## Train MPR

With the prepared data at `/path/to/ultramsdata`:

```bash
torchrun --standalone --nproc_per_node=4 -m ultrams_training.train_pretraining \
  --config training/ultrams_training/configs/ultrams_pretraining.json \
  --project-root training/ultrams_training/model \
  --data-root /path/to/ultramsdata \
  --end-stage peak_reconstruction
```

For six GPUs, change `--nproc_per_node=4` to `6`. The run writes checkpoints, the resolved configuration, input validation, and training metrics to its output directory.
