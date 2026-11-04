"""Add standard Euclidean DB on frozen Fig. 2c structural families."""
from pathlib import Path
import hashlib
import json
import sys
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'fig2c_supervised_structure_families'
INPUT = ROOT / 'source_data/original_inputs'
OUT = ROOT / 'source_data'
N_BOOT = 10000
SEED = 3407
EXPECTED = {
    'test_dreams_supervised.npy': 'bdb5567c167f757c6d32ccc743629b004b40a329ebac799a6e439924f3ba9f2c',
    'test_ultrams_supervised.npy': 'f9bd8d21cef3bbb2366e55eefe3e014bc7007e11a20bfde86f6c6f94963782e8',
    'test_metadata.csv': 'ba2ed918b745a8a2402c4aef913b72efe30bd44f1536df53872da523c9935ae7',
    'test_top150_peaks.npz': 'f262073c8c462667c2368d9ed659aa596cd4645af1eea912b4dcc84b89747223',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def db_from_gram(gram, groups, weights):
    """Exact bootstrap DB via multiplicities; recompute every centroid/spread."""
    batch = len(weights[0])
    k = len(groups)
    center_norms = np.zeros((batch, k))
    spreads = np.zeros((batch, k))
    for j, indices in enumerate(groups):
        block = gram[np.ix_(indices, indices)]
        products = weights[j] @ block
        center_norms[:, j] = np.sum(products * weights[j], axis=1)
        squared = np.maximum(np.diag(block)[None, :] - 2 * products + center_norms[:, j, None], 0)
        spreads[:, j] = np.sum(weights[j] * np.sqrt(squared), axis=1)
    distances = np.zeros((batch, k, k))
    for a in range(k):
        for b in range(a + 1, k):
            cross = np.sum((weights[a] @ gram[np.ix_(groups[a], groups[b])]) * weights[b], axis=1)
            d = np.sqrt(np.maximum(center_norms[:, a] + center_norms[:, b] - 2 * cross, 0))
            distances[:, a, b] = distances[:, b, a] = d
    np.fill_diagonal(distances[0], 0)  # all diagonals were initialized to zero
    if np.any(distances[:, ~np.eye(k, dtype=bool)] <= 1e-12):
        raise RuntimeError('Coincident family centroids: inspect DB definition')
    ratio = (spreads[:, :, None] + spreads[:, None, :]) / np.where(distances > 0, distances, np.inf)
    scores = ratio.max(axis=2).mean(axis=1)
    return scores, spreads, distances, ratio


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        assert sha(INPUT / name) == expected, name
    selected_path = BASE / 'audit/locked_families/test_selected_metadata.csv'
    assert sha(selected_path) == '19af6c063cb35ce98f5732ca30c622f97f49f6c4754e4b0eb8bf0e72a1eb260a'
    selected = pd.read_csv(selected_path)
    full = pd.read_csv(INPUT / 'test_metadata.csv')
    assert len(full) == 1592 and full.connectivity.is_unique
    assert len(selected) == 661 and selected.connectivity.is_unique
    indices = full.set_index('connectivity').index.get_indexer(selected.connectivity)
    assert (indices >= 0).all()
    for field in ['spectrum_id', 'canonical_smiles', 'precursor_mz']:
        assert np.array_equal(full.iloc[indices][field].to_numpy(), selected[field].to_numpy()), field
    labels = selected.structure_cluster.to_numpy(int)
    groups = [np.flatnonzero(labels == j) for j in range(6)]
    assert [len(g) for g in groups] == [327, 100, 89, 62, 48, 35]
    selected.to_csv(OUT / 'test_selected_metadata.csv', index=False)

    helper = ROOT.parent / 'fig2c_isomer/scripts'
    sys.path.insert(0, str(helper))
    from evaluate_cross_ce_isomer_retrieval import raw_features, unit_norm
    features = {
        'Raw spectrum': unit_norm(raw_features(INPUT / 'test_top150_peaks.npz')[indices]),
        'DreaMS': unit_norm(np.load(INPUT / 'test_dreams_supervised.npy')[indices]),
        'UltraMS': unit_norm(np.load(INPUT / 'test_ultrams_supervised.npy')[indices]),
    }
    # Match the existing resampling stream: each replicate draws each family in turn.
    rng = np.random.default_rng(SEED)
    resamples = np.vstack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                           for _ in range(N_BOOT)])
    np.save(OUT / 'paired_bootstrap_indices.npy', resamples.astype(np.int32))
    counts = np.vstack([np.bincount(row, minlength=len(labels)) for row in resamples])
    weights = [counts[:, g].astype(np.float64) / len(g) for g in groups]
    original = pd.read_csv(BASE / 'source_data/figure2c_cluster_metrics.csv')
    rows, boot_all, verification, family_rows, pair_rows = [], {}, {}, [], []
    for name, original_values in features.items():
        safe = name.lower().replace(' ', '_')
        if sparse.issparse(original_values):
            sparse.save_npz(OUT / f'{safe}_normalized_661.npz', original_values)
            x = original_values.toarray().astype(np.float64)
        else:
            np.save(OUT / f'{safe}_normalized_661.npy', original_values)
            x = original_values.astype(np.float64)
        assert np.isfinite(x).all() and np.all(np.linalg.norm(x, axis=1) > 0)
        old_sil = original[(original.display_name == name) & (original.metric == 'silhouette')].iloc[0]
        per_sil = silhouette_samples(original_values, labels, metric='cosine')
        assert abs(per_sil.mean() - old_sil.value) < 2e-7, (name, per_sil.mean(), old_sil.value)
        gram = x @ x.T
        point = davies_bouldin_score(x, labels)
        uniform = [np.ones((1, len(g))) / len(g) for g in groups]
        point_gram, spread, distances, ratio = db_from_gram(gram, groups, uniform)
        assert abs(point - point_gram[0]) < 1e-10
        scores, _, _, _ = db_from_gram(gram, groups, weights)
        # Independently recompute selected real bootstrap resamples in feature space.
        checks = []
        for b in [0, 1, 42, 999, 9999]:
            direct = davies_bouldin_score(x[resamples[b]], labels[resamples[b]])
            assert abs(direct - scores[b]) < 1e-10, (name, b, direct, scores[b])
            checks.append({'replicate': b, 'direct': direct, 'gram': float(scores[b])})
        boot_all[name] = scores
        lo, hi = np.quantile(scores, [0.025, 0.975])
        rows.append({'display_name': name, 'metric': 'davies_bouldin', 'value': point,
                     'diagnostic_bootstrap_q025': lo, 'diagnostic_bootstrap_q975': hi,
                     'n_molecules': len(labels), 'n_families': 6,
                     'direction': 'lower_is_better', 'space': 'L2_normalized_original_features',
                     'distance': 'euclidean', 'labels': 'frozen_Butina_structure_families'})
        for j, g in enumerate(groups):
            family_rows.append({'display_name': name, 'structure_cluster': j, 'n': len(g),
                                'mean_distance_to_centroid': spread[0, j],
                                'worst_pair_ratio': ratio[0, j].max(),
                                'worst_neighbor_family': int(ratio[0, j].argmax())})
            for k in range(j + 1, 6):
                pair_rows.append({'display_name': name, 'family_a': j, 'family_b': k,
                                  'centroid_distance': distances[0, j, k], 'ratio': ratio[0, j, k]})
        verification[name] = {'silhouette_reproduced': float(per_sil.mean()),
                              'silhouette_saved': float(old_sil.value),
                              'db_sklearn': point, 'db_gram': float(point_gram[0]),
                              'bootstrap_checks': checks, 'dimension': x.shape[1]}
        print(name, point, lo, hi, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'db_summary.csv', index=False)
    pd.DataFrame(boot_all).to_csv(OUT / 'db_paired_bootstrap.csv', index_label='replicate')
    pd.DataFrame(family_rows).to_csv(OUT / 'db_per_family.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / 'db_family_pairs.csv', index=False)
    diffs = []
    for ref in ['Raw spectrum', 'DreaMS']:
        delta = boot_all['UltraMS'] - boot_all[ref]
        point = rows[2]['value'] - next(r['value'] for r in rows if r['display_name'] == ref)
        lo, hi = np.quantile(delta, [0.025, 0.975])
        diffs.append({'comparison': f'UltraMS - {ref}', 'difference': point,
                      'diagnostic_bootstrap_q025': lo, 'diagnostic_bootstrap_q975': hi,
                      'direction': 'negative_favors_UltraMS'})
    pd.DataFrame(diffs).to_csv(OUT / 'db_paired_differences.csv', index=False)
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    protocol = {'status': 'complete', 'n_molecules': 661, 'family_sizes': [len(g) for g in groups],
                'definition': 'Standard Euclidean Davies-Bouldin on frozen Butina structure families',
                'normalization': 'Original float32 row L2 normalization; distance arithmetic float64',
                'bootstrap': {'n': N_BOOT, 'seed': SEED, 'unit': 'molecule within structure family',
                              'paired': True, 'diagnostic_bounds': '2.5/97.5 percentiles',
                              'use': 'diagnostic only; systematic bootstrap shift; not validated confidence intervals',
                              'centroids_and_dispersion': 'recomputed for every resample'},
                'software': {'python': platform.python_version(), 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
                'input_sha256': EXPECTED, 'frozen_labels_sha256': sha(selected_path),
                'script_sha256': sha(Path(__file__))}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
