"""Run the original spectral-property experiment programs by task name.

This launcher sets source-module and input paths. Each task program retains its
own train/validation/test logic and CLI arguments; saved-output metric replay is
the separate ``reproduce.py`` entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ORIGINAL = HERE / "original"
SUPPORT = ORIGINAL / "support"
MAGMA_SUPPORT = HERE / "magma_support"
PUBLIC_MODEL = REPOSITORY / "training/ultrams_training/model"
PUBLIC_TRAIN = PUBLIC_MODEL / "train"

PROGRAMS = {
    "prepare_massspecgym": "prepare_massspecgym.py",
    "masked_peak_reconstruction": "masked_peak_reconstruction/evaluate_masked_peaks.py",
    "ion_mode": "ion_mode/train_test_ion_mode.py",
    "negative_ion_structure_prepare": "negative_ion_structure/prepare_spectraverse_structure_clusters.py",
    "negative_ion_structure_extract": "negative_ion_structure/extract_supervised_cls_only.py",
    "negative_ion_structure_select": "negative_ion_structure/evaluate_butina_structure_clusters.py",
    "negative_ion_structure_test": "negative_ion_structure/formalize_butina_metrics.py",
    "neutral_loss_train_and_test": "neutral_loss/train_test_neutral_loss.py",
    "heteroatom_count_train": "heteroatom_count/train_atom_query.py",
    "heteroatom_count_test": "heteroatom_count/evaluate_atom_query_on_test.py",
    "heteroatom_count_classical": "heteroatom_count/run_linear_spectrum_baseline.py",
    "element_peak_localization_test": "element_peak_localization/evaluate_element_peaks.py",
    "element_peak_localization_summary": "element_peak_localization/summarize_element_peaks.py",
    "peak_neighbors_prepare": "peak_neighbors/prepare_msnlib_magma_subset.py",
    "peak_neighbors_annotate": "peak_neighbors/run_magma_msnlib_subset.py",
    "peak_neighbors_extract": "peak_neighbors/extract_peak_embeddings.py",
    "peak_neighbors_train_and_test": "peak_neighbors/peak_neighbor_evaluation.py",
}

INPUT_ENV = {
    "stage_d_checkpoint": "ULTRAMS_STAGE_D_CKPT",
    "reconstruction_checkpoint": "ULTRAMS_RECONSTRUCTION_CKPT",
    "dreams_checkpoint": "ULTRAMS_DREAMS_CKPT",
    "dreams_root": "ULTRAMS_DREAMS_ROOT",
    "massgym_csv": "ULTRAMS_MASSGYM_CSV",
    "massgym_parquet": "ULTRAMS_MASSGYM_PARQUET",
    "msnlib_csv": "ULTRAMS_MSNLIB_CSV",
    "spectraverse_csv": "ULTRAMS_SPECTRAVERSE_CSV",
    "gems_hdf5": "ULTRAMS_GEMS_HDF5",
    "ultrago_root": "ULTRAGO_DIR",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path,
                        default=Path(os.environ.get("ULTRAMS_EXPERIMENT_ROOT", REPOSITORY)))
    parser.add_argument("--source-train-root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    for option in INPUT_ENV:
        parser.add_argument("--" + option.replace("_", "-"), type=Path)
    parser.add_argument("task", choices=sorted(PROGRAMS))
    parser.add_argument("task_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    source_train = args.source_train_root or PUBLIC_TRAIN
    source_train = source_train.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ULTRAMS_EXPERIMENT_ROOT"] = str(experiment_root)
    env["ULTRAMS_SOURCE_TRAIN_ROOT"] = str(source_train)
    env["ULTRAMS_SOURCE_SUPPLEMENT_ROOT"] = str(PUBLIC_MODEL)
    env.setdefault("ULTRAGO_DIR", str(MAGMA_SUPPORT))
    for option, name in INPUT_ENV.items():
        value = getattr(args, option)
        if value is not None:
            env[name] = str(value.resolve())
    imports = [
        MAGMA_SUPPORT,
        SUPPORT,
        SUPPORT / "comparison",
        source_train,
        PUBLIC_TRAIN,
        PUBLIC_MODEL,
        experiment_root,
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in imports] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    task_args = args.task_args[1:] if args.task_args[:1] == ["--"] else args.task_args
    command = [sys.executable, str(ORIGINAL / PROGRAMS[args.task]), *task_args]
    print("Running", args.task, "in", run_dir, flush=True)
    subprocess.run(command, cwd=run_dir, env=env, check=True)


if __name__ == "__main__":
    main()
