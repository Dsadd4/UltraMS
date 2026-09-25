
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(os.environ.get('LIGHT_ULTRA_ROOT', '.')).resolve()
DATA = ROOT / 'datasets' / 'NPLIB1'
OUT_AUDIT = ROOT / 'train/output/comparison/nplib1_data_audit'

MAX_PEAKS = 150
MS2_BLOCK_TYPES = {'ms2peaks', 'collision'}
EXPECTED_SIDECARS = {
    'splits_ms_pred/split_1.tsv': (246418, '24c66ff1c16c0dc55a06645e20dbf41d8f77572731b19bf06bdb6a7c5ec4ff66'),
    'splits_ms_pred/split_2.tsv': (246505, '567b1e789fa22cc7b0c0431000b5227b9ce0b551c4a1a016e37c6145367eea68'),
    'splits_ms_pred/split_3.tsv': (246400, 'd576fe464ba0ba5405d79acd9ea978e73cf232dfb0b7ae13dbf7296d7c78dfac'),
    'retrieval_candidates/cands_df.tsv': (49873428, '671cfba8c8f5b5f9bb619ccca09bd6f7485be564bd87a3ab31a555e272c2e641'),
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels() -> Dict[str, dict]:
    with (DATA / 'labels.tsv').open(newline='', errors='replace') as f:
        return {row['spec']: row for row in csv.DictReader(f, delimiter='\t')}


def load_split(split_id: int) -> Dict[str, str]:
    with (DATA / 'splits_ms_pred' / f'split_{split_id}.tsv').open(newline='', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fold_col = [c for c in reader.fieldnames if c != 'spec'][0]
        return {row['spec']: row[fold_col] for row in reader}


def parse_ms2_blocks(path: Path):
    """Return metadata and every numeric block; filtering to MS2 happens downstream."""
    metadata = {}
    parentmass = None
    header = None
    rows: List[Tuple[float, float]] = []
    blocks = []
    header_index = -1
    metadata_phase = True

    def flush():
        nonlocal rows, header
        if header is not None and rows:
            blocks.append({'header': header, 'peaks': np.asarray(rows, dtype=np.float32)})
        rows = []

    for raw in path.read_text(errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>'):
            flush()
            header_index += 1
            header = line[1:].strip()
            parts = header.split(None, 1)
            key = parts[0] if parts else ''
            val = parts[1] if len(parts) == 2 else ''
            if key.lower() == 'parentmass':
                try:
                    parentmass = float(val)
                except ValueError:
                    pass
            if metadata_phase and key:
                # Preserve repeated metadata keys with a suffix.
                mkey = key
                while mkey in metadata:
                    mkey += "'"
                metadata[mkey] = val
            continue
        if line.startswith('#'):
            if metadata_phase:
                body = line[1:].strip()
                if ' ' in body:
                    key, val = body.split(' ', 1)
                    mkey = key
                    while mkey in metadata:
                        mkey += "'"
                    metadata[mkey] = val
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            mz = float(parts[0])
            inten = float(parts[1])
        except ValueError:
            continue
        if mz > 0 and inten >= 0:
            metadata_phase = False
            rows.append((mz, inten))
    flush()
    return metadata, parentmass, blocks


def block_type_and_collision(header: str):
    h = header.strip()
    low = h.lower()
    block_type = low.split()[0] if low else ''
    collision = ''
    m = re.search(r'collision\s+([0-9.]+)\s*([a-zA-Z]*)', low)
    if m:
        collision = (m.group(1) + ((' ' + m.group(2)) if m.group(2) else '')).strip()
    return block_type, collision


def prep_top150(peaks: np.ndarray):
    """Model-facing representation: valid peaks, top 150 by intensity, sorted by m/z."""
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz = peaks[:, 0].astype(np.float32)
    inten = peaks[:, 1].astype(np.float32)
    mask = (mz > 0) & (inten > 0)
    mz, inten = mz[mask], inten[mask]
    if len(mz) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(mz) > MAX_PEAKS:
        idx = np.argsort(inten)[-MAX_PEAKS:]
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


def arr_to_text(arr):
    return ','.join(f'{float(x):.8g}' for x in arr)


def build_blocks(labels):
    manifest_rows = []
    skipped = []
    header_mismatch = Counter()
    non_ms2_block_counter = Counter()
    for spec_id in sorted(labels):
        label = labels[spec_id]
        path = DATA / 'spec_files' / f'{spec_id}.ms'
        if not path.exists():
            skipped.append({'parent_spec_id': spec_id, 'reason': 'missing_ms_file'})
            continue
        try:
            meta, parentmass, blocks = parse_ms2_blocks(path)
        except Exception as exc:
            skipped.append({'parent_spec_id': spec_id, 'reason': type(exc).__name__})
            continue
        if parentmass is None:
            skipped.append({'parent_spec_id': spec_id, 'reason': 'missing_parentmass'})
            continue
        header_ik = str(meta.get('InChIKey', '') or meta.get('InChIKey\'', ''))
        if header_ik and header_ik != str(label.get('inchikey', '')):
            header_mismatch['inchikey'] += 1
        if not blocks:
            skipped.append({'parent_spec_id': spec_id, 'reason': 'no_numeric_ms2_blocks'})
            continue
        for bi, block in enumerate(blocks):
            btype, collision = block_type_and_collision(block['header'])
            if btype not in MS2_BLOCK_TYPES:
                non_ms2_block_counter[btype] += 1
                continue
            peaks = block['peaks']
            top = prep_top150(peaks)
            row = {
                'block_id': f'{spec_id}::ms2block{bi:03d}',
                'parent_spec_id': spec_id,
                'block_index': bi,
                'block_header': block['header'],
                'block_type': btype,
                'collision_energy': collision,
                'n_peaks_raw': int(len(peaks)),
                'n_peaks_model': int(len(top)),
                'valid_for_model': int(len(top) >= 3),
                'precursor_mz': f'{float(parentmass):.8f}',
                'smiles': label.get('smiles', ''),
                'inchikey': label.get('inchikey', ''),
                'formula': label.get('formula', ''),
                'adduct': label.get('ionization', ''),
                'instrument': label.get('instrument', ''),
                'molecule_source': 'labels.tsv',
                'mzs': arr_to_text(top[:, 0]) if len(top) else '',
                'intensities': arr_to_text(top[:, 1]) if len(top) else '',
            }
            manifest_rows.append(row)
    return manifest_rows, skipped, header_mismatch, non_ms2_block_counter


def write_tsv(path: Path, rows: List[dict], fields: List[str]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})


def write_csv(path: Path, rows: List[dict], fields: List[str]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})


def main():
    global ROOT, DATA, OUT_AUDIT
    parser = argparse.ArgumentParser(description='Prepare NPLIB1 MS2 blocks and candidate maps')
    parser.add_argument('--root', type=Path, required=True, help='Benchmark workspace root')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    DATA = ROOT / 'datasets' / 'NPLIB1'
    OUT_AUDIT = ROOT / 'train/output/comparison/nplib1_data_audit'
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    sidecars = [*(DATA / 'splits_ms_pred' / f'split_{i}.tsv' for i in (1, 2, 3)),
                DATA / 'retrieval_candidates/cands_df.tsv']
    missing_sidecars = [str(path) for path in sidecars if not path.exists()]
    if missing_sidecars:
        raise FileNotFoundError('NPLIB1 benchmark split/candidate files missing: '
                                + ', '.join(missing_sidecars))
    for relative, (size, expected) in EXPECTED_SIDECARS.items():
        path = DATA / relative
        if path.stat().st_size != size or sha256(path) != expected:
            raise ValueError(f'NPLIB1 benchmark input differs from the reported experiment: {path}')
    archive = DATA / 'canopus_train.zip'
    if archive.is_file() and (not (DATA / 'labels.tsv').exists()
                              or not any((DATA / 'spec_files').glob('*.ms'))):
        with zipfile.ZipFile(archive) as zf:
            entries = [entry for entry in zf.infolist()
                       if entry.filename == 'canopus_train/labels.tsv'
                       or entry.filename.startswith('canopus_train/spec_files/')]
            if not any(entry.filename == 'canopus_train/labels.tsv' for entry in entries):
                raise ValueError(f'{archive} does not contain canopus_train/labels.tsv')
            for entry in entries:
                if entry.is_dir():
                    continue
                destination = DATA / entry.filename.removeprefix('canopus_train/')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or destination.stat().st_size != entry.file_size:
                    partial = destination.with_name(destination.name + '.part')
                    with zf.open(entry) as source, partial.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    partial.replace(destination)
    required = [DATA / 'labels.tsv', DATA / 'spec_files']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('NPLIB1 inputs missing: ' + ', '.join(missing))
    labels = load_labels()
    rows, skipped, header_mismatch, non_ms2_block_counter = build_blocks(labels)

    manifest_fields = [
        'block_id', 'parent_spec_id', 'block_index', 'block_header', 'block_type',
        'collision_energy', 'n_peaks_raw', 'n_peaks_model', 'valid_for_model',
        'precursor_mz', 'smiles', 'inchikey', 'formula', 'adduct', 'instrument',
        'molecule_source', 'mzs', 'intensities',
    ]
    manifest_path = DATA / 'NPLIB1_ms2block_manifest.tsv'
    write_tsv(manifest_path, rows, manifest_fields)

    valid_rows = [r for r in rows if int(r['valid_for_model']) == 1]
    model_fields = [
        'spec_id', 'parent_spec_id', 'block_index', 'block_header', 'collision_energy',
        'mzs', 'intensities', 'smiles', 'inchikey', 'formula', 'precursor_mz', 'adduct', 'fold'
    ]
    split_stats = {}
    for split_id in [1, 2, 3]:
        split = load_split(split_id)
        split_rows = []
        for r in valid_rows:
            fold = split.get(r['parent_spec_id'])
            if fold not in {'train', 'val', 'test'}:
                continue
            split_rows.append({
                'spec_id': r['block_id'],
                'parent_spec_id': r['parent_spec_id'],
                'block_index': r['block_index'],
                'block_header': r['block_header'],
                'collision_energy': r['collision_energy'],
                'mzs': r['mzs'],
                'intensities': r['intensities'],
                'smiles': r['smiles'],
                'inchikey': r['inchikey'],
                'formula': r['formula'],
                'precursor_mz': r['precursor_mz'],
                'adduct': r['adduct'],
                'fold': fold,
            })
        out_csv = DATA / f'NPLIB1_ms2blocks_split{split_id}.csv'
        write_csv(out_csv, split_rows, model_fields)
        fold_counts = Counter(r['fold'] for r in split_rows)
        parent_counts = Counter(r['parent_spec_id'] for r in split_rows)
        split_stats[str(split_id)] = {
            'csv': str(out_csv),
            'rows': len(split_rows),
            'fold_counts': dict(fold_counts),
            'unique_parent_specs': len(parent_counts),
            'multi_block_parent_specs': sum(1 for v in parent_counts.values() if v > 1),
        }

    # Candidate map by block id for spec-level evaluator, plus smiles map for current evaluator.
    for split_id in [1, 2, 3]:
        full_cand_path = DATA / 'retrieval_candidates' / 'cands_df.tsv'
        cand_by_parent = defaultdict(list)
        with full_cand_path.open(newline='', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                cand_by_parent[row['spec']].append(row['smiles'])
        block_map = {r['block_id']: cand_by_parent.get(r['parent_spec_id'], []) for r in valid_rows}
        (DATA / f'NPLIB1_ms2blocks_candidates_by_block_split{split_id}.json').write_text(json.dumps(block_map, indent=2))
        # smiles-level map remains valid if candidate list is consistent; this is convenient for existing code.
        smiles_map = {}
        inconsistent = []
        for r in valid_rows:
            sm = r['smiles']
            lst = cand_by_parent.get(r['parent_spec_id'], [])
            prev = smiles_map.setdefault(sm, lst)
            if prev != lst:
                inconsistent.append(r['block_id'])
        (DATA / f'NPLIB1_ms2blocks_candidates_by_smiles_split{split_id}.json').write_text(json.dumps(smiles_map, indent=2))
        split_stats[str(split_id)]['candidate_block_keys'] = len(block_map)
        split_stats[str(split_id)]['candidate_smiles_keys'] = len(smiles_map)
        split_stats[str(split_id)]['candidate_smiles_inconsistent_blocks'] = len(inconsistent)

    stats = {
        'labels': len(labels),
        'ms2_blocks_total': len(rows),
        'ms2_blocks_valid_for_model': len(valid_rows),
        'parent_specs_with_blocks': len(set(r['parent_spec_id'] for r in rows)),
        'parent_specs_with_valid_blocks': len(set(r['parent_spec_id'] for r in valid_rows)),
        'parent_specs_multi_valid_blocks': sum(1 for v in Counter(r['parent_spec_id'] for r in valid_rows).values() if v > 1),
        'skipped_parent_specs': skipped,
        'header_label_mismatch': dict(header_mismatch),
        'block_type_counts': dict(Counter(r['block_type'] for r in rows)),
        'non_ms2_blocks_excluded': dict(non_ms2_block_counter),
        'raw_peak_count_summary': {
            'min': int(min(int(r['n_peaks_raw']) for r in rows)),
            'median': float(np.median([int(r['n_peaks_raw']) for r in rows])),
            'mean': float(np.mean([int(r['n_peaks_raw']) for r in rows])),
            'max': int(max(int(r['n_peaks_raw']) for r in rows)),
        },
        'model_peak_count_summary': {
            'min': int(min(int(r['n_peaks_model']) for r in rows)),
            'median': float(np.median([int(r['n_peaks_model']) for r in rows])),
            'mean': float(np.mean([int(r['n_peaks_model']) for r in rows])),
            'max': int(max(int(r['n_peaks_model']) for r in rows)),
        },
        'splits': split_stats,
        'note': 'Each numeric MS2 block is a separate sample. Molecule annotation is inherited from labels.tsv by parent_spec_id. No MS2 blocks are merged.',
    }
    if (stats['labels'], stats['ms2_blocks_total'], stats['ms2_blocks_valid_for_model']) != (10709, 19687, 18979):
        raise ValueError('NPLIB1 spectrum or MS2-block counts differ from the benchmark')
    expected_folds = {
        '1': {'train': 15256, 'val': 1765, 'test': 1958},
        '2': {'train': 15475, 'val': 1583, 'test': 1921},
        '3': {'train': 15248, 'val': 1750, 'test': 1981},
    }
    for split_id, fold_counts in expected_folds.items():
        if split_stats[split_id]['rows'] != 18979 or split_stats[split_id]['fold_counts'] != fold_counts:
            raise ValueError(f'NPLIB1 split {split_id} differs from the benchmark')
        if split_stats[split_id]['candidate_block_keys'] != 18979:
            raise ValueError(f'NPLIB1 split {split_id} has missing candidate keys')
        if split_stats[split_id]['candidate_smiles_inconsistent_blocks']:
            raise ValueError(f'NPLIB1 split {split_id} has inconsistent molecule candidate lists')
    stats_path = OUT_AUDIT / 'nplib1_ms2block_manifest_summary.json'
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    md_path = OUT_AUDIT / 'nplib1_ms2block_manifest_summary.md'
    lines = ['# NPLIB1 MS2-block manifest summary', '', stats['note'], '']
    lines.append(f"labels={stats['labels']} blocks_total={stats['ms2_blocks_total']} valid_blocks={stats['ms2_blocks_valid_for_model']}")
    lines.append(f"parent_specs_with_valid_blocks={stats['parent_specs_with_valid_blocks']} multi_valid_parent_specs={stats['parent_specs_multi_valid_blocks']}")
    lines.append(f"block_type_counts={stats['block_type_counts']}")
    lines.append(f"non_ms2_blocks_excluded={stats['non_ms2_blocks_excluded']}")
    for sid, st in split_stats.items():
        lines.append(f"split {sid}: rows={st['rows']} folds={st['fold_counts']} unique_parent_specs={st['unique_parent_specs']} multi_block_parent_specs={st['multi_block_parent_specs']} candidate_block_keys={st['candidate_block_keys']} inconsistent_smiles_candidates={st['candidate_smiles_inconsistent_blocks']}")
    md_path.write_text('\n'.join(lines) + '\n')
    print(manifest_path)
    print(stats_path)
    print(md_path)
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
