"""Recalculate Figure 3f from all saved test-query molecule ranks."""
from pathlib import Path
import csv
import gzip
import json
import math

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'cross_polarity'


def reproduce():
    reference = {}
    with (DATA / 'retrieval_summary.csv').open() as handle:
        for row in csv.DictReader(handle):
            reference[row['model'], row['direction']] = row
    frames, summary, paired = {}, [], []
    for model in ('ultra', 'dreams'):
        for direction in ('pos_to_neg', 'neg_to_pos'):
            with gzip.open(DATA / f'{model}_{direction}_per_query.csv.gz', 'rt') as handle:
                rows = list(csv.DictReader(handle))
            frames[model, direction] = rows
            expected = reference[model, direction]
            supported = [r for r in rows if r['has_library_positive'].lower() == 'true']
            assert len(rows) == int(expected['n_query'])
            assert len(supported) == int(expected['n_supported'])
            for k in (1, 5, 10):
                hits = sum(0 < int(r['gt_molecule_rank']) <= k for r in rows)
                assert hits == int(expected[f'n_hit_at_{k}'])
                recall = hits / len(supported)
                assert math.isclose(recall, float(expected[f'recall_at_{k}_supported']), abs_tol=1e-12)
                summary.append(dict(model=model, direction=direction, k=k,
                                    n_query=len(rows), n_supported=len(supported),
                                    n_hit=hits, recall_percent=100 * recall))
    with (DATA / 'paired_improvement.csv').open() as handle:
        expected_pairs = {(r['direction'], int(r['k'])): r for r in csv.DictReader(handle)}
    for direction in ('pos_to_neg', 'neg_to_pos'):
        u, d = frames['ultra', direction], frames['dreams', direction]
        for a, b in zip(u, d):
            assert all(a[key] == b[key] for key in ('query_h5_row', 'query_source_row', 'query_smiles', 'has_library_positive'))
        for k in (1, 5, 10):
            gained = lost = 0
            for a, b in zip(u, d):
                ua, da = 0 < int(a['gt_molecule_rank']) <= k, 0 < int(b['gt_molecule_rank']) <= k
                gained += ua and not da
                lost += da and not ua
            ref = expected_pairs[direction, k]
            assert gained == int(ref['dreams_wrong_ultra_correct'])
            assert lost == int(ref['dreams_correct_ultra_wrong'])
            assert gained-lost == int(ref['net_new_correct'])
            paired.append(dict(direction=direction, k=k, gained=gained, lost=lost, net=gained-lost))
    return dict(recall=summary, paired=paired)


if __name__ == '__main__':
    result = reproduce()
    output = HERE / 'results' / 'cross_polarity.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Recomputed 12 recall values and 6 paired comparisons from 18,790 model-query records: {output}')
