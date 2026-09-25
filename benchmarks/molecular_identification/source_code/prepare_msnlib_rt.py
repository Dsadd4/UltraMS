"""Add SpecBridge retention times to the MSnLib benchmark CSV."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgf_rt(mgf_path):
    """Parse MGF line-by-line, extract (mzs_hash, precursor_mz, smiles, rt) per spectrum."""
    records = []
    current = {}
    peaks_mz = []
    in_peaks = False

    with open(mgf_path) as f:
        for line in tqdm(f, desc="Parsing MGF"):
            line = line.strip()
            if line == 'BEGIN IONS':
                current = {}
                peaks_mz = []
                in_peaks = False
            elif line == 'END IONS':
                if peaks_mz and 'rt' in current:
                    first3 = sorted(peaks_mz)[:3]
                    current['mz_key'] = ','.join(f'{m:.5f}' for m in first3)
                    current['n_peaks'] = len(peaks_mz)
                    records.append(current)
                current = {}
                peaks_mz = []
                in_peaks = False
            elif '=' in line and not in_peaks:
                key, _, val = line.partition('=')
                key = key.strip().upper()
                if key == 'RTINSECONDS':
                    try:
                        current['rt'] = float(val)
                    except ValueError:
                        pass
                elif key == 'SMILES':
                    current['smiles'] = val.strip()
                elif key in ('PEPMASS', 'PRECURSOR_MZ'):
                    try:
                        current['precursor_mz'] = float(val.split()[0])
                    except ValueError:
                        pass
            elif line and line[0].isdigit():
                in_peaks = True
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks_mz.append(float(parts[0]))
                    except ValueError:
                        pass

    print(f"Parsed {len(records)} spectra with RT from MGF")
    return records


def build_csv_key(row):
    """Build matching key from CSV row's mzs field."""
    mzs_str = str(row['mzs'])
    mzs = sorted(float(x) for x in mzs_str.split(',') if x.strip())[:3]
    return ','.join(f'{m:.5f}' for m in mzs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Benchmark workspace root')
    args = parser.parse_args()
    source = args.root.resolve() / 'datasets' / 'MSnLib'
    mgf_path = source / 'SpecBridge_MSnLib_dataset.mgf'
    csv_path = source / 'MSnLib.csv'
    output_path = source / 'MSnLib_with_rt.csv'
    mgf_records = parse_mgf_rt(mgf_path)

    rt_lookup = {}
    for r in mgf_records:
        key = (r.get('smiles', ''), r.get('mz_key', ''))
        rt_lookup[key] = r['rt']
    print(f"RT lookup table: {len(rt_lookup)} entries")

    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    rt_vals = []
    matched = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching RT"):
        smiles = str(row.get('smiles', ''))
        mz_key = build_csv_key(row)
        key = (smiles, mz_key)
        rt = rt_lookup.get(key, 0.0)
        if rt > 0:
            matched += 1
        rt_vals.append(rt)

    df['rt'] = rt_vals
    print(f"Matched: {matched}/{len(df)} ({100*matched/len(df):.1f}%)")

    valid_rt = df[df['rt'] > 0]['rt']
    if len(valid_rt) > 0:
        print(f"RT stats: mean={valid_rt.mean():.1f}s, std={valid_rt.std():.1f}s, "
              f"min={valid_rt.min():.1f}s, max={valid_rt.max():.1f}s")

    df.to_csv(output_path, index=False)
    audit = {
        'source_mgf_sha256': sha256(mgf_path),
        'source_csv_sha256': sha256(csv_path),
        'output_csv_sha256': sha256(output_path),
        'n_spectra': len(df),
        'n_with_rt': matched,
        'n_without_rt': len(df) - matched,
        'n_mgf_rt_records': len(mgf_records),
        'n_rt_lookup_keys': len(rt_lookup),
    }
    if len(df) != 560084 or matched != 446032:
        raise ValueError(f'MSnLib spectrum or retention-time counts differ: {len(df)}, {matched}')
    (source / 'MSnLib_rt_conversion.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f"Saved to: {output_path}")
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
