"""CPU regressions for the published UltraMS training entry points."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ultrams_training import prepare_ultramsdata_config as preparation
from ultrams_training import train_pretraining as pretraining
from ultrams_training.train_full_adaptation import (
    ContinuousAdamWFactory,
    NormalizedRTLoader,
)


class CurrentTrainingTests(unittest.TestCase):
    def test_rt_loader_normalizes_seconds_without_mutating_source(self):
        source_rt = torch.tensor([0.0, 300.0, 600.0, 1500.0])
        source_batch = {"rt": source_rt, "orig_spectra": torch.ones(4, 2, 2)}

        class BatchLoader:
            dataset = SimpleNamespace(epoch=1)

            def __iter__(self):
                yield source_batch

        loader = NormalizedRTLoader(
            BatchLoader(),
            {"1": {"microbatches_per_rank": [1]}},
            scale=600.0,
        )
        for _ in range(2):
            normalized = list(loader)[0]
            torch.testing.assert_close(
                normalized["rt"], torch.tensor([0.0, 0.5, 1.0, 2.5])
            )
            self.assertIs(normalized["orig_spectra"], source_batch["orig_spectra"])
        torch.testing.assert_close(
            source_rt, torch.tensor([0.0, 300.0, 600.0, 1500.0])
        )

    def test_adamw_keeps_moments_across_rt_to_ion_transition(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        factory = ContinuousAdamWFactory(torch.optim.AdamW, fused=False)

        def parameter_group(stage, learning_rate, weight_decay):
            return [
                {
                    "params": [parameter],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "group_name": "shared_early_decay",
                    "parameter_names_sha256": "shared_parameter",
                    "protocol_stage": stage,
                }
            ]

        optimizer = factory(parameter_group("supervised_rt", 0.01, 0.01))
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rt_state = copy.deepcopy(optimizer.state[parameter])
        rt_parameter = parameter.detach().clone()
        self.assertTrue(torch.any(rt_state["exp_avg"] != 0))
        self.assertTrue(torch.any(rt_state["exp_avg_sq"] != 0))

        continued = factory(parameter_group("ion_mode", 0.005, 0.0))
        self.assertIs(continued, optimizer)
        self.assertEqual(continued.param_groups[0]["lr"], 0.005)
        self.assertEqual(continued.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(continued.param_groups[0]["protocol_stage"], "ion_mode")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                continued.state[parameter][key], rt_state[key], rtol=0, atol=0
            )

        (parameter - 3.0).square().sum().backward()
        continued.step()
        torch.testing.assert_close(
            continued.state[parameter]["step"], rt_state["step"] + 1
        )
        self.assertFalse(torch.equal(parameter.detach(), rt_parameter))

    def test_phase1_config_generation_loads_current_training_config(self):
        template = Path(pretraining.__file__).parent / "configs" / "ultrams_mpr.json"
        template_config = json.loads(template.read_text(encoding="utf-8"))
        assets = copy.deepcopy(template_config["expected_assets"])
        audits = copy.deepcopy(template_config["data_audits"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            audit_path.write_text(
                json.dumps({"status": "passed", "errors": []}), encoding="utf-8"
            )
            output = root / "configs" / "mpr.json"
            # Dataset validation is exercised separately on the full data release.
            # Keep config generation, serialization and loading real here.
            with patch.object(
                preparation, "read_verified_assets", return_value=({}, assets, audits)
            ):
                frozen = preparation.freeze_config(
                    mode="phase1",
                    template=template,
                    ultramsdata_dir=root / "clean",
                    polarity_dir=root / "polarity",
                    audit_path=audit_path,
                    output=output,
                    run_name="ultramsdata-mpr-regression",
                )
            loaded = pretraining.load_config(output, root)

            self.assertEqual(loaded["run_name"], "ultramsdata-mpr-regression")
            self.assertEqual(loaded["data_mode"], "ultramsdata_parquet")
            self.assertEqual(loaded["expected_assets"], assets)
            self.assertEqual(loaded["data_audits"], audits)
            self.assertEqual(loaded["ultramsdata_freeze"]["mode"], "phase1")
            self.assertEqual(
                loaded["phase1_contract_sha256"], preparation.phase1_identity(frozen)
            )
            self.assertEqual(
                loaded["paths"]["clean_shard_dir"],
                str((root / frozen["paths"]["clean_shard_dir"]).resolve()),
            )


if __name__ == "__main__":
    unittest.main()
