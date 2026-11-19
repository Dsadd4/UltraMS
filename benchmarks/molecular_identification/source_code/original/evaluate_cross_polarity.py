"""Exact molecule-level Recall@1/5/10 and MRR on frozen full cross-polarity tests.

No mass filtering, retraining, new encoding, library pruning or query selection.
One molecule's score is its highest original reference-spectrum cosine; ties
use the earliest original reference row attaining that score.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROTOCOL = dict(version=1, metrics=["molecular_Recall@1", "molecular_Recall@5", "molecular_Recall@10", "molecular_MRR"],
    query="all original 6202 positive and 3193 negative test queries in exact original order",
    reference="all original opposite-mode 137000 negative or 295434 positive spectra; no deletions",
    training=False, encoding=False, mass_filter=False, checkpoint_selection=False,
    identity="unchanged exact frozen reference SMILES identity strings; no new standardization",
    molecule_score="maximum original float32 normalized cosine over all spectra of this molecule",
    molecular_tie="earliest original library-cache row attaining that molecule maximum",
    matrix_multiplication="original float32 full-reference q @ library.T, query batch 128",
    rank="1 + number of molecule scores strictly greater than GT maximum + number equal with earlier first maximum row",
    supported_denominators=dict(pos_to_neg=2421, neg_to_pos=2476),
    no_reference="GT rank=-1; recall hits false and reciprocal rank zero; retained in all-query denominator",
    scope="frozen cross-polarity adapted model ranking comparison; not FDR controlled identification or a pretraining-only ablation")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def bind(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen contract changed: {path}"
    else:
        save(path, value)


def load_metadata(source, base, out):
    roles = json.loads((base / "ultra/molecule_roles.json").read_text())
    assert roles == json.loads((base / "dreams/molecule_roles.json").read_text())
    data, population = {}, {}
    with h5py.File(source / "msnlib_library_spectra.h5", "r") as h:
        for mode in ("pos", "neg"):
            labels = np.asarray(json.loads((source / f"smiles_{mode}.json").read_text()), dtype=object)
            names = labels[np.asarray(h[mode]["smiles_index"])]
            p = np.load(source / f"protocol_{mode}.npz")
            qr, lr = p["test_query_rows"], p["test_library_rows"]
            source_rows = np.asarray(h[mode]["source_row"])
            qnames, lnames = names[qr], names[lr]
            assert len(qr) == (6202 if mode == "pos" else 3193)
            assert len(lr) == (295434 if mode == "pos" else 137000)
            assert len(set(qnames)) == len(qr)
            assert not set(qnames) & (set(roles["train"]) | set(roles["val"]))
            ids, molecules = pd.factorize(lnames, sort=False)
            assert (ids >= 0).all()
            molecules = np.asarray(molecules, dtype=object)
            ordered_rows = np.argsort(ids, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(ids[ordered_rows])) + 1]
            assert np.array_equal(ids[ordered_rows[starts]], np.arange(len(molecules)))
            lookup = {str(s): i for i, s in enumerate(molecules)}
            mapping = pd.DataFrame(dict(molecule_id=np.arange(len(molecules)), smiles=molecules,
                n_reference_spectra=np.bincount(ids), first_library_cache_row=ordered_rows[starts]))
            mapping.to_csv(out / f"reference_molecules_{mode}.csv.gz", index=False)
            data[mode] = dict(qr=qr, lr=lr, qnames=qnames, lnames=lnames, source=source_rows,
                ids=ids, molecules=molecules, lookup=lookup, ordered_rows=ordered_rows, starts=starts)
            population[mode] = dict(n_query=len(qr), n_reference_spectra=len(lr), n_reference_molecules=len(molecules),
                overlap_adaptation_train=0, overlap_selection_val=0)
    save(out / "POPULATION.json", population)
    return data


def evaluate(data, base, full, out, model, qm, lm):
    direction = f"{qm}_to_{lm}"
    dest = out / model
    dest.mkdir(exist_ok=True)
    qd, ld = data[qm], data[lm]
    q = np.load(full / model / f"queries_{qm}.npy", mmap_mode="r")
    library = np.load(base / model / f"full_library_{lm}.npy", mmap_mode="r")
    assert q.shape == (len(qd["qr"]), 256) and library.shape == (len(ld["lr"]), 256)
    assert q.dtype == library.dtype == np.float32
    original = pd.read_csv(full / model / f"{direction}_per_query.csv", keep_default_na=False)
    assert np.array_equal(original.query_h5_row, qd["qr"])
    assert np.array_equal(original.query_source_row, qd["source"][qd["qr"]])
    assert np.array_equal(original.query_smiles, qd["qnames"])
    gt = np.asarray([ld["lookup"].get(str(s), -1) for s in qd["qnames"]], dtype=np.int64)
    support = gt >= 0
    assert np.array_equal(support, original.has_library_positive)
    assert int(support.sum()) == PROTOCOL["supported_denominators"][direction]
    n, ng = len(q), len(ld["molecules"])
    assert ng >= 10
    rank = np.full(n, -1, dtype=np.int64)
    top_ids = np.full((n, 10), -1, dtype=np.int64)
    top_rows = np.full((n, 10), -1, dtype=np.int64)
    top_scores = np.full((n, 10), np.nan, dtype=np.float32)
    progress = dest / f"{direction}_progress.npz"
    start = 0
    if progress.exists():
        r = np.load(progress)
        start = int(r["next_query"])
        rank[:], top_ids[:], top_rows[:], top_scores[:] = r["gt_rank"], r["top_ids"], r["top_rows"], r["top_scores"]
    for begin in range(start, n, 128):
        end = min(begin + 128, n)
        matrix = np.asarray(q[begin:end]) @ library.T
        assert matrix.dtype == np.float32 and np.isfinite(matrix).all()
        raw_best = matrix.argmax(axis=1)
        raw_score = matrix[np.arange(end-begin), raw_best]
        for j, i in enumerate(range(begin, end)):
            s = matrix[j]
            ms = np.maximum.reduceat(s[ld["ordered_rows"]], ld["starts"])
            winners = np.flatnonzero(s == ms[ld["ids"]])
            first = np.full(ng, len(s), dtype=np.int64)
            np.minimum.at(first, ld["ids"][winners], winners)
            assert (first < len(s)).all()
            # Include every score tie at the Top10 boundary before tie sorting.
            cut = np.partition(ms, ng - 10)[ng - 10]
            choices = np.flatnonzero(ms >= cut)
            order = np.lexsort((first[choices], -ms[choices]))
            chosen = choices[order[:10]]
            top_ids[i] = chosen
            top_rows[i] = first[chosen]
            # Preserve actual representative-row score bits, including +/-0.
            top_scores[i] = s[first[chosen]]
            assert first[chosen[0]] == raw_best[j]
            if support[i]:
                g = gt[i]
                rank[i] = 1 + int((ms > ms[g]).sum()) + int(((ms == ms[g]) & (first < first[g])).sum())
                assert (rank[i] == 1) == bool(original.top1_correct.iloc[i])
                assert (rank[i] <= 10) == (g in chosen)
                if rank[i] <= 10:
                    assert chosen[rank[i]-1] == g
        temp = progress.with_suffix(".part.npz")
        np.savez(temp, next_query=end, gt_rank=rank, top_ids=top_ids, top_rows=top_rows, top_scores=top_scores)
        os.replace(temp, progress)
        if begin % 1024 == 0 or end == n:
            print(json.dumps(dict(model=model, direction=direction, done=end, total=n)), flush=True)
    assert np.array_equal(rank == 1, original.top1_correct)
    rr = np.zeros(n, dtype=np.float64)
    rr[support] = 1.0 / rank[support]
    frame = pd.DataFrame(dict(original_query_index=np.arange(n), query_h5_row=qd["qr"],
        query_source_row=qd["source"][qd["qr"]], query_smiles=qd["qnames"],
        has_library_positive=support, gt_molecule_id=gt, gt_molecule_rank=rank, reciprocal_rank=rr))
    result = dict(model=model, direction=direction, n_query=n, n_reference_spectra=len(library),
        n_reference_molecules=ng, n_supported=int(support.sum()),
        mrr_supported=float(rr[support].mean()), mrr_all=float(rr.mean()))
    for k in (1, 5, 10):
        hits = support & (rank <= k)
        frame[f"hit_at_{k}"] = hits
        result[f"n_hit_at_{k}"] = int(hits.sum())
        result[f"recall_at_{k}_supported"] = float(hits.sum()/support.sum())
        result[f"recall_at_{k}_all"] = float(hits.mean())
    frame.to_csv(dest / f"{direction}_per_query.csv.gz", index=False)
    flat_rows = top_rows.ravel()
    top = pd.DataFrame(dict(original_query_index=np.repeat(np.arange(n), 10),
        molecular_rank=np.tile(np.arange(1, 11), n), molecule_id=top_ids.ravel(),
        smiles=ld["molecules"][top_ids.ravel()], score=top_scores.ravel(),
        representative_library_cache_row=flat_rows, representative_h5_row=ld["lr"][flat_rows],
        representative_source_row=ld["source"][ld["lr"][flat_rows]]))
    top.to_csv(dest / f"{direction}_top10_molecules.csv.gz", index=False)
    save(dest / f"{direction}_SUMMARY.json", result)
    print(json.dumps(result), flush=True)
    return result


def paired_summary(out):
    rows = []
    for direction in ("pos_to_neg", "neg_to_pos"):
        u, d = [pd.read_csv(out / model / f"{direction}_per_query.csv.gz", keep_default_na=False)
                for model in ("ultra", "dreams")]
        columns = ["query_h5_row", "query_source_row", "query_smiles", "has_library_positive"]
        pd.testing.assert_frame_equal(u[columns], d[columns])
        supported = int(u.has_library_positive.sum())
        for k in (1, 5, 10):
            uh = u.gt_molecule_rank.between(1, k)
            dh = d.gt_molecule_rank.between(1, k)
            gained, lost = int((uh & ~dh).sum()), int((dh & ~uh).sum())
            rows.append(dict(direction=direction, k=k, n_query=len(u), n_supported=supported,
                ultra_correct=int(uh.sum()), dreams_correct=int(dh.sum()), both_correct=int((uh & dh).sum()),
                dreams_wrong_ultra_correct=gained, dreams_correct_ultra_wrong=lost,
                net_new_correct=gained-lost, delta_supported_percentage_points=100*(gained-lost)/supported))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bind(out / "PROTOCOL.json", PROTOCOL)
    base = root / "results/cross_polarity/models"
    full = base
    source = root / "train/output/comparison/molecular_benchmarks/same_adduct_manifest"
    data = load_metadata(source, base, out)
    results = []
    for model in ("ultra", "dreams"):
        for qm, lm in (("pos", "neg"), ("neg", "pos")):
            save(out / "STATE.json", dict(stage="running", model=model, direction=f"{qm}_to_{lm}"))
            results.append(evaluate(data, base, full, out, model, qm, lm))
    pd.DataFrame(results).to_csv(out / "retrieval_summary.csv", index=False)
    paired_summary(out).to_csv(out / "paired_improvement.csv", index=False)
    save(out / "COMPLETE.json", dict(status="complete", training=False, gpu_used=False, mass_filter=False, results=results))
    save(out / "STATE.json", dict(stage="complete"))


if __name__ == "__main__":
    main()
