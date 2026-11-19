# UltraMS pretraining

The from-scratch recipe in this package trains on UltraMSdata in three stages: **masked peak reconstruction (5 epochs) → retention time (5 epochs) → joint retention-time and ion-mode training (11 epochs)**. The latter two stages update all effective model parameters.

All three stages use the processed GNPS/MassIVE UltraMSdata corpus or its ion-mode-balanced subset. MSnLib, SpectraVerse, MassSpecGym and MoNA are used in the separate downstream benchmark, contrastive-training and reference-library workflows.

## Install

Use Python 3.11 on Linux with four GPUs.

```bash
git clone https://github.com/Dsadd4/UltraMS.git
cd UltraMS
python -m pip install -e ./training
```

## Prepare UltraMSdata

Use the prepared UltraMSdata bundle with the following two shard sets:

| Set | Spectra | Parquet files | Role |
| --- | ---: | ---: | --- |
| UltraMSdata | 160,641,162 | 816 | Masked peak reconstruction and retention-time training |
| Ion-mode-balanced subset of UltraMSdata | 21,586,258 | 48 | Joint retention-time and ion-mode training; 10,793,129 spectra of each polarity drawn from UltraMSdata |

The prepared bundle is available at [Hugging Face](https://huggingface.co/datasets/dsadd4/UltraMSdata): download `derived/ultramsdata/shards/` (816 clean shards) and `derived/ultramsdata_polarity/shards/` (48 polarity shards), each with its `manifest.json`. The [public source-file manifest](https://doi.org/10.5281/zenodo.23966050) identifies the original MassIVE files. The commands below require the prepared bundle.

Set `DATA_ROOT` to the bundle directory. Link the two prepared sets to the training paths:

```bash
export DATA_ROOT=/path/to/ultramsdata
python -m ultrams_training.data_tools.link_ultramsdata \
  --data-root "$DATA_ROOT" \
  --config training/ultrams_training/configs/ultrams_mpr.json
```

The training paths are:

```text
$DATA_ROOT/derived/ultramsdata/shards/
$DATA_ROOT/derived/ultramsdata_polarity/shards/
```

Prepare the data manifests; their output is also used to configure the next training stage:

```bash
python -m ultrams_training.data_tools.validate_pretraining_data \
  --ultramsdata-dir "$DATA_ROOT/derived/ultramsdata/shards" \
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

The [configuration producer](ultrams_training/prepare_ultramsdata_config.py) reads the UltraMSdata manifests and completed MPR checkpoint. It fills the data identities and calculates the retention-time training schedule from the actual shard layout. Generate the configuration, then train:

```bash
export MPR_CHECKPOINT="$MPR_RUN/checkpoints/stage_peak_reconstruction_epoch_5.pt"
python -m ultrams_training.prepare_ultramsdata_config \
  --mode adaptation \
  --template training/ultrams_training/configs/ultrams_full_adaptation.json \
  --ultramsdata-dir "$DATA_ROOT/derived/ultramsdata/shards" \
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

## Checkpoint use

| Task | File and command |
| --- | --- |
| Inference or downstream fine-tuning | Load one of the three Hugging Face models with `UltraMS.from_pretrained(...)`. These contain weights and model configuration. |
| Resume a pretraining run | Pass that run's `checkpoints/latest.pt` with `--resume`, using the same resolved configuration and output directory. The training file restores optimizer, scheduler, RNG and progress. |
| Start RT training after MPR | Use the completed `stage_peak_reconstruction_epoch_5.pt` and its sidecar/configuration in the commands above. |
| Share a newly trained model | Export a separate inference file with the command below, preserving the original training file. |

From the repository root, with the inference package installed:

```bash
python -m pip install -e .
python tools/export_inference_checkpoint.py /path/to/run/checkpoints/latest.pt /path/to/inference_model.pt
```

The exporter reads the full checkpoint produced by your training run, including its RNG state, and writes only the weights and necessary model configuration. The public Hugging Face inference files do not contain optimizer/RNG state and are not pretraining resume checkpoints. To start a new run with `--initialize-from`, use a new output directory; to continue an existing run, use `--resume` without `--initialize-from`.
