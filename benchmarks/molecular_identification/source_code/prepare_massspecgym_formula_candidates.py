#!/usr/bin/env python3
"""Reuse mass-trained fingerprint heads with MassSpecGym formula candidates."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "original"


def load_source_module():
    path = ORIGINAL / "39_massspecgym_ms2fp_sourcefusion.py"
    spec = importlib.util.spec_from_file_location("massspecgym_source_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Mass-trained source run name")
    parser.add_argument("--run-name", required=True, help="Formula candidate view name")
    parser.add_argument("--formula-fingerprint-cache", type=Path, help="Existing formula/all Morgan fingerprint cache")
    args = parser.parse_args()

    root = Path(os.environ["LIGHT_ULTRA_ROOT"]).resolve()
    base = root / "train/output/comparison/massspecgym_ms2fp_sourcefusion"
    source = json.loads((base / args.source_run / "summary.json").read_text())
    mass = source["tasks"]["mass"]["scenarios"]["all"]
    output = base / args.run_name
    output.mkdir(parents=True, exist_ok=True)

    m39 = load_source_module()
    samples, train_idx, val_idx, test_idx, _ = m39.load_massspecgym(0)
    candidates, candidate_audit = m39.load_candidates("formula", samples)
    fingerprint_cache = args.formula_fingerprint_cache
    if fingerprint_cache is None:
        previous_cache = (base / "sourcefusion_all_formula_mass_20260703a" / "fingerprints"
                          / "fpdict_formula_all_bits2048_r2_chir0_traincand0_35ea97d3f7e3a133.pkl")
        if previous_cache.is_file():
            fingerprint_cache = previous_cache
    if fingerprint_cache is None:
        scenario = m39.SCENARIOS["all"]
        splits = [m39.scenario_mask(samples, idx, scenario) for idx in (train_idx, val_idx, test_idx)]
        smiles = m39.collect_smiles_for_protocol(samples, *splits, candidates, include_train_candidates=False)
        protocol = argparse.Namespace(fp_bits=2048, radius=2, fp_use_chirality=False,
                                      include_train_candidates_in_fp=False, force_fp=False)
        fp_dict, fingerprint_cache = m39.build_or_load_fp_dict(output, "formula", "all", smiles, protocol)
        n_fingerprints = len(fp_dict)
    else:
        if not fingerprint_cache.is_file():
            raise FileNotFoundError(fingerprint_cache)
        import pickle
        with fingerprint_cache.open("rb") as stream:
            n_fingerprints = len(pickle.load(stream))

    view = copy.deepcopy(source)
    view["config"]["tasks"] = ["formula", "mass"]
    view["config"]["formula_view_of"] = args.source_run
    view["config"]["formula_view_note"] = (
        "Formula candidate evaluation view: reuse source heads/features from mass source run; "
        "replace fingerprint cache with formula/all cache."
    )
    view["candidate_audit"]["formula"] = candidate_audit
    formula = copy.deepcopy(mass)
    formula["fingerprint_cache"] = str(Path(fingerprint_cache).resolve())
    formula["n_fingerprints"] = n_fingerprints
    formula["sources_from_formula_view_of"] = args.source_run
    formula["ensembles"] = {}
    view["tasks"]["formula"] = {"scenarios": {"all": formula}}

    (output / "summary.json").write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")
    (output / "config.json").write_text(json.dumps({
        "run_name": args.run_name,
        "formula_view_of": args.source_run,
        "fingerprint_cache_formula": str(Path(fingerprint_cache).resolve()),
        "note": "Formula candidate evaluation view; source heads are reused without retraining.",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_name": args.run_name, "formula_fingerprints": n_fingerprints,
                      "fingerprint_cache": str(fingerprint_cache)}, indent=2))


if __name__ == "__main__":
    main()
