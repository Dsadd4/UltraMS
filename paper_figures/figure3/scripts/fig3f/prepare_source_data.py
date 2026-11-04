"""Freeze complete plotting inputs and check them against all original queries."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "molecular_topk_results_20261026"
DEST = HERE / "source_data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    assert json.loads((ORIGINAL / "COMPLETE.json").read_text())["status"] == "complete"
    for n in ("INDEPENDENT_RESULT_VERIFICATION.json", "POSTRUN_INPUT_VERIFICATION.json"):
        assert json.loads((ORIGINAL / n).read_text())["status"] == "pass"
    summary = pd.read_csv(ORIGINAL / "retrieval_summary.csv", float_precision="round_trip")
    paired = pd.read_csv(ORIGINAL / "paired_improvement.csv", float_precision="round_trip")
    inputs = ["retrieval_summary.csv", "paired_improvement.csv", "PROTOCOL.json", "INPUTS.json",
              "COMPLETE.json", "INDEPENDENT_RESULT_VERIFICATION.json", "POSTRUN_INPUT_VERIFICATION.json"]
    inputs += [f"{model}/{direction}_per_query.csv.gz" for model in ("ultra", "dreams")
               for direction in ("pos_to_neg", "neg_to_pos")]
    bindings = {}
    frames, recall_rows = {}, []
    for row in summary.itertuples(index=False):
        frame = pd.read_csv(ORIGINAL / row.model / f"{row.direction}_per_query.csv.gz",
                            keep_default_na=False, float_precision="round_trip")
        frames[row.model, row.direction] = frame
        assert len(frame) == row.n_query
        assert int(frame.has_library_positive.sum()) == row.n_supported
        assert len(set(frame.query_smiles)) == row.n_query
        rr = frame.reciprocal_rank.to_numpy()
        assert np.isclose(rr[frame.has_library_positive].mean(), row.mrr_supported, atol=1e-14, rtol=0)
        for k in (1, 5, 10):
            n = int(frame[f"hit_at_{k}"].sum())
            assert n == getattr(row, f"n_hit_at_{k}")
            value = 100 * n / row.n_supported
            assert np.isclose(value / 100, getattr(row, f"recall_at_{k}_supported"), atol=1e-14, rtol=0)
            recall_rows.append(dict(model=row.model, direction=row.direction, k=k,
                n_query=row.n_query, n_supported=row.n_supported, n_hit=n, recall_percent=value,
                n_reference_spectra=row.n_reference_spectra, n_reference_molecules=row.n_reference_molecules))
    for row in paired.itertuples(index=False):
        u, d = frames["ultra", row.direction], frames["dreams", row.direction]
        for key in ("query_h5_row", "query_source_row", "query_smiles", "has_library_positive"):
            assert np.array_equal(u[key], d[key])
        a, b = u[f"hit_at_{row.k}"], d[f"hit_at_{row.k}"]
        assert int((a & ~b).sum()) == row.dreams_wrong_ultra_correct
        assert int((~a & b).sum()) == row.dreams_correct_ultra_wrong
        assert int(a.sum() - b.sum()) == row.net_new_correct
    for name in inputs:
        target = DEST / "original" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL / name, target)
        assert sha(target) == sha(ORIGINAL / name)
        bindings[name] = sha(target)
    pd.DataFrame(recall_rows).to_csv(DEST / "recall_points.csv", index=False)
    paired.to_csv(DEST / "paired_points.csv", index=False)
    summary[["model", "direction", "n_query", "n_supported", "mrr_supported", "mrr_all"]].to_csv(DEST / "mrr.csv", index=False)
    (DEST / "SOURCE_BINDING.json").write_text(json.dumps(dict(status="pass", original=str(ORIGINAL),
        full_queries_recounted=True, original_files=bindings,
        fixed_k=[1, 5, 10], mass_filter=False, uncertainty="none plotted; fixed-model full-cohort point estimates"), indent=2) + "\n")


if __name__ == "__main__":
    main()
