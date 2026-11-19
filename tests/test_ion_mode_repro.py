"""Regression tests for probe initialization and selected-head replay."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'benchmarks/spectral_properties/original/ion_mode'
sys.path.insert(0, str(SOURCE))
from train_test_ion_mode import train_mlp_probe
from evaluate_mlp import evaluate_head


class IonModeReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.RandomState(3)
        self.emb = rng.normal(size=(96, 12)).astype('float32')
        self.labels = (self.emb[:, 0] > 0).astype('int64')
        self.splits = dict(labels=self.labels, train_idx=np.arange(64),
                           val_idx=np.arange(64, 80), test_idx=np.arange(80, 96))

    def train(self, directory, labels=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_mlp_probe(
                self.emb, self.labels if labels is None else labels,
                self.splits['train_idx'], self.splits['val_idx'], self.splits['test_idx'],
                torch.device('cpu'), hidden1=16, hidden2=8, max_epochs=4,
                bs=16, forward_bs=8, patience=2, rng_seed=42, artifact_dir=directory)

    def test_seed_controls_initialization_independently_of_previous_rng(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            torch.manual_seed(17)
            first = self.train(a)
            torch.manual_seed(986)
            torch.rand(173)
            second = self.train(b)
            self.assertEqual(first, second)
            x = torch.load(a/'mlp.pt', weights_only=True)
            y = torch.load(b/'mlp.pt', weights_only=True)
            for key in x['state_dict']:
                self.assertTrue(torch.equal(x['state_dict'][key], y['state_dict'][key]))

    def test_selected_checkpoint_reproduces_every_saved_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            metrics = self.train(d)
            replay, predictions = evaluate_head(Path(d)/'mlp.pt', self.emb,
                                                 self.splits, torch.device('cpu'))
            self.assertEqual(replay, {k: metrics[k] for k in ('mlp_val', 'mlp_test')})
            with np.load(Path(d)/'mlp_predictions.npz') as saved:
                for key, value in predictions.items():
                    np.testing.assert_array_equal(value, saved[key])

    def test_test_labels_do_not_select_head_or_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            self.train(a)
            changed = self.labels.copy()
            changed[self.splits['test_idx']] = 1-changed[self.splits['test_idx']]
            self.train(b, changed)
            x = torch.load(a/'mlp.pt', weights_only=True)
            y = torch.load(b/'mlp.pt', weights_only=True)
            self.assertEqual(x['config'], y['config'])
            for key in x['state_dict']:
                self.assertTrue(torch.equal(x['state_dict'][key], y['state_dict'][key]))


if __name__ == '__main__':
    unittest.main()
