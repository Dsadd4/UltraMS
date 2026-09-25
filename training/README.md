# UltraMS pretraining

UltraMS trains on UltraMSdata in three stages: **masked peak reconstruction (5 epochs) → retention time (5 epochs) → joint retention-time and ion-mode training (11 epochs)**. The latter two stages update all effective model parameters.

## Install

Use Python 3.11 on Linux with four GPUs.

```bash
git clone https://github.com/Dsadd4/UltraMS.git
cd UltraMS
python -m pip install -e ./training
```

## Prepare UltraMSdata

Set `DATA_ROOT` to the directory containing the two prepared UltraMSdata shard sets. Link an existing data bundle by its dataset fingerprints:

```bash
export DATA_ROOT=/path/to/ultramsdata
python -m ultrams_training.data_tools.link_ultramsdata \
  --data-root "$DATA_ROOT" \
  --config training/ultrams_training/configs/ultrams_mpr.json
```

The training paths are:

```text
$DATA_ROOT/derived/ultramsdata_clean/shards/
$DATA_ROOT/derived/ultramsdata_polarity/shards/
```

Each shard directory contains its `manifest.json`. The [data preparation code](ultrams_training/data_tools) produces these shard sets from the frozen source bundle. Run the data audit once; its output is also used to prepare the next training stage:

```bash
python -m ultrams_training.data_tools.validate_pretraining_data \
  --clean-dir "$DATA_ROOT/derived/ultramsdata_clean/shards" \
  --polarity-dir "$DATA_ROOT/derived/ultramsdata_polarity/shards" \
  --output "$DATA_ROOT/ultramsdata_audit.json"
```

## Masked peak reconstruction

[`train_pretraining.py`](ultrams_training/train_pretraining.py) and [`ultrams_mpr.json`](ultrams_training/configs/ultrams_mpr.json) run the five-epoch MPR stage on UltraMSdata. Check inputs, then train:

```bash
python -m ultrams_training.train_pretraining \
  --config training/ultrams_training/configs/ultrams_mpr.json \
  --project-root training/ultrams_training/model \
  --data-root "$DATA_ROOT" \
  --end-stage peak_reconstruction --validate-only

export MPR_RUN=/path/to/mpr_run
torchrun --standalone --nproc_per_node=4 -m ultrams_training.train_pretraining \
  --config training/ultrams_training/configs/ultrams_mpr.json \
  --project-root training/ultrams_training/model \
  --data-root "$DATA_ROOT" \
  --output-dir "$MPR_RUN" \
  --end-stage peak_reconstruction
```

To resume with the same code, data and GPU count, add `--resume "$MPR_RUN/checkpoints/latest.pt"`.

## Retention time and ion mode

The [configuration producer](ultrams_training/prepare_ultramsdata_config.py) reads the audited UltraMSdata manifests and completed MPR checkpoint. It fills the data identities and calculates the retention-time training schedule from the actual shard layout. Generate the configuration, then train:

```bash
export MPR_CHECKPOINT="$MPR_RUN/checkpoints/stage_peak_reconstruction_epoch_5.pt"
python -m ultrams_training.prepare_ultramsdata_config \
  --mode adaptation \
  --template training/ultrams_training/configs/ultrams_full_adaptation.json \
  --clean-dir "$DATA_ROOT/derived/ultramsdata_clean/shards" \
  --polarity-dir "$DATA_ROOT/derived/ultramsdata_polarity/shards" \
  --audit-path "$DATA_ROOT/ultramsdata_audit.json" \
  --phase1-config "$MPR_RUN/resolved_config.json" \
  --parent-checkpoint "$MPR_CHECKPOINT" \
  --parent-sidecar "$MPR_CHECKPOINT.json" \
  --run-name ultramsdata_full_adaptation \
  --output "$DATA_ROOT/ultrams_full_adaptation.json"

torchrun --standalone --nproc_per_node=4 -m ultrams_training.train_full_adaptation \
  --config "$DATA_ROOT/ultrams_full_adaptation.json" \
  --project-root training/ultrams_training/model \
  --data-root "$DATA_ROOT" \
  --initialize-from "$MPR_CHECKPOINT" \
  --output-dir /path/to/full_adaptation_run
```

The training entries save checkpoints, resolved configurations, input validation and training metrics in their output directories.
