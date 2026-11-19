"""File input and command-line embedding checks without model downloads."""

from __future__ import annotations

import gzip
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ultrams.cli import main
from ultrams.spectra import read_mgf, read_mzml, read_spectra


MGF = """BEGIN IONS
TITLE=first spectrum
PEPMASS=301.2 42.0
100.1 20
121.1 100
150.0 35
END IONS
BEGIN IONS
SCANS=42
PEPMASS=315.3
102.1 40
135.2 100
167.3 25
END IONS
"""


class SpectrumFileTests(unittest.TestCase):
    def test_mgf_preserves_order_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectra.mgf"
            path.write_text(MGF)
            spectra = list(read_mgf(path))
            self.assertEqual([row["id"] for row in spectra], ["first spectrum", "42"])
            self.assertEqual([row["precursor_mz"] for row in spectra], [301.2, 315.3])
            self.assertEqual(spectra[0]["mz"], [100.1, 121.1, 150.0])

    def test_gzipped_mgf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectra.mgf.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(MGF)
            self.assertEqual(len(list(read_spectra(path))), 2)

    def test_name_precedes_placeholder_scan_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "named.mgf"
            path.write_text(
                "BEGIN IONS\nNAME=example molecule\nSCANS=-1\n"
                "PEPMASS=301.2,42\n100 20\n120 40\n150 10\nEND IONS\n"
            )
            spectrum = next(read_mgf(path))
            self.assertEqual(spectrum["id"], "example molecule")
            self.assertEqual(spectrum["precursor_mz"], 301.2)

    def test_missing_precursor_names_spectrum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.mgf"
            path.write_text("BEGIN IONS\nTITLE=missing mz\n100 10\nEND IONS\n")
            with self.assertRaisesRegex(ValueError, "missing mz.*precursor"):
                list(read_mgf(path))

    def test_unclosed_mgf_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.mgf"
            path.write_text("BEGIN IONS\nPEPMASS=300\n100 10\n")
            with self.assertRaisesRegex(ValueError, "unfinished BEGIN IONS"):
                list(read_mgf(path))

    def test_mzml_reads_ms2_precursor_and_skips_ms1(self) -> None:
        @contextmanager
        def fake_read(_path):
            yield iter(
                [
                    {"id": "scan=1", "ms level": 1},
                    {
                        "id": "scan=2",
                        "ms level": 2,
                        "m/z array": np.asarray([100.1, 121.1, 150.0]),
                        "intensity array": np.asarray([20, 100, 35]),
                        "precursorList": {
                            "precursor": [
                                {"selectedIonList": {"selectedIon": [{"selected ion m/z": 301.2}]}}
                            ]
                        },
                    },
                ]
            )

        package = types.ModuleType("pyteomics")
        module = types.ModuleType("pyteomics.mzml")
        module.read = fake_read
        package.mzml = module
        with patch.dict(sys.modules, {"pyteomics": package, "pyteomics.mzml": module}):
            spectra = list(read_mzml("unused.mzML"))
        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0]["id"], "scan=2")
        self.assertEqual(spectra[0]["precursor_mz"], 301.2)

    def test_cli_saves_ordered_embeddings(self) -> None:
        class FakeModel:
            def encode_batch(self, records, *, batch_size):
                return np.asarray(
                    [[record["precursor_mz"], len(record["mz"])] for record in records],
                    dtype=np.float32,
                )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "spectra.mgf"
            output = Path(directory) / "result.npz"
            source.write_text(MGF)
            with patch("ultrams.cli.UltraMS.from_pretrained", return_value=FakeModel()):
                main([str(source), str(output), "--batch-size", "1", "--device", "cpu"])
            with np.load(output) as result:
                self.assertEqual(result["ids"].tolist(), ["first spectrum", "42"])
                np.testing.assert_array_equal(
                    result["embeddings"], np.asarray([[301.2, 3], [315.3, 3]], dtype=np.float32)
                )


if __name__ == "__main__":
    unittest.main()
