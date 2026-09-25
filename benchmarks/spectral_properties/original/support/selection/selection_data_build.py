"""Expose the published ion-mode experiment's Stage-D loader to atom-query tasks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "ion_mode" / "train_test_ion_mode.py"
_spec = importlib.util.spec_from_file_location("_ultrams_ion_mode_source", _SOURCE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot import ion-mode experiment: {_SOURCE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
load_phase2_stage_d_for_fusion = _module.load_phase2_stage_d_for_fusion
