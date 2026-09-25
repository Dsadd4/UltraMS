#!/usr/bin/env python3
"""Convert the official MassSpecGym TSV to the CSV consumed by the benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SOURCE_SHA256 = '0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95'
EXPECTED_FLOAT32 = {
    'parent_mass': '01f1262700296d5ca4f4e7d8be7f17955b9c23b18a48a566909d021d9d6b88d2',
    'precursor_mz': 'cf3b238bec5452851fd3ee1ae881416aa53f428a3515a7901e0dce730dcb5194',
    'collision_energy': 'a3f6e73ad0245bdddfaf38ff51c301ce9bcdaafd1f6fe62d16754aa3c1752b1c',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Benchmark workspace root')
    args = parser.parse_args()
    data = args.root / 'datasets' / 'MassSpecGym'
    source = data / 'MassSpecGym.tsv'
    target = data / 'MassSpecGym.csv'
    if not source.is_file():
        raise FileNotFoundError(source)
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError('MassSpecGym.tsv does not match the official benchmark input')
    frame = pd.read_csv(source, sep='\t')
    expected = {'identifier', 'mzs', 'intensities', 'smiles', 'fold'}
    if not expected.issubset(frame.columns):
        raise ValueError(f'Missing official columns: {sorted(expected - set(frame.columns))}')
    frame.to_csv(target, index=True)
    numeric_input_hashes = {
        column: hashlib.sha256(np.asarray(frame[column], dtype='<f4').tobytes()).hexdigest()
        for column in ('parent_mass', 'precursor_mz', 'collision_energy')
    }
    if numeric_input_hashes != EXPECTED_FLOAT32:
        raise ValueError('MassSpecGym model input floats differ from the published benchmark input')
    audit = {
        'source_tsv_sha256': sha256(source),
        'csv_sha256': sha256(target),
        'spectra': len(frame),
        'fold_counts': {str(key): int(value) for key, value in frame['fold'].value_counts().items()},
        'float32_input_sha256': numeric_input_hashes,
    }
    (data / 'MassSpecGym_conversion.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    print(f'{target}: {len(frame)} spectra')


if __name__ == '__main__':
    main()
