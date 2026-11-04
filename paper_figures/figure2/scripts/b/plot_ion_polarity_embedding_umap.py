"""
Ion-polarity embedding projection for Figure 2 Task 4.

Samples m/z-matched positive/negative MSnLib spectra, extracts spectrum-level
embeddings with UltraMS rt_only_d and DreaMS, and renders a compact UMAP panel.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
FIG2_DIR = os.path.join(PROJECT_ROOT, "main_figure", "fig2")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
COMPARISON_DIR = os.path.join(TRAIN_DIR, "comparison")
DATA_DIR = os.path.join(FIG2_DIR, "data", "ion_polarity_embedding")
OUT_DIR = os.path.join(FIG2_DIR, "output", "task4_ion_polarity_embedding")
MSNLIB_CSV = os.path.join(PROJECT_ROOT, "datasets", "MSnLib", "MSnLib.csv")
COMPARISON_SCRIPT = os.path.join(COMPARISON_DIR, "10_cross_modal_deep_proj_plot.py")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

C_ULTRA = "#E64B35"
C_DREAM = "#4DBBD5"
C_POS = "#D55E00"
C_NEG = "#0072B2"
C_GRAY = "#666666"

NATURE = {
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(NATURE)


def _load_comparison_module():
    sys.path.insert(0, TRAIN_DIR)
    sys.path.insert(0, COMPARISON_DIR)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "supplement", "pretrain_lighting"))
    spec = importlib.util.spec_from_file_location("ultrams_comparison_10", COMPARISON_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ultrams_comparison_10", mod)
    spec.loader.exec_module(mod)
    return mod


def _load_dreams_loader():
    src = os.path.join(COMPARISON_DIR, "dreams_loader.py")
    spec = importlib.util.spec_from_file_location("fig2_dreams_loader", src)
    mod = importlib.util.module_from_spec(spec)
    # The migrated source is missing a top-level import os; inject it before exec.
    mod.os = os
    sys.modules.setdefault("fig2_dreams_loader", mod)
    spec.loader.exec_module(mod)
    return mod


def select_mz_matched_indices(csv_path, n_pairs=500, seed=42, split="test"):
    usecols = ["precursor_mz", "adduct", "fold"]
    meta = pd.read_csv(csv_path, usecols=usecols)
    meta["row_idx"] = np.arange(len(meta), dtype=np.int64)
    if split and "fold" in meta.columns:
        split_meta = meta[meta["fold"].astype(str) == split].copy()
        if (split_meta["adduct"].eq("[M+H]+").sum() >= n_pairs and
                split_meta["adduct"].eq("[M-H]-").sum() >= n_pairs):
            meta = split_meta

    pos = meta[meta["adduct"].eq("[M+H]+")].copy()
    neg = meta[meta["adduct"].eq("[M-H]-")].copy()
    if len(pos) < n_pairs or len(neg) < n_pairs:
        raise ValueError(f"Need at least {n_pairs} positive and negative spectra; got pos={len(pos)}, neg={len(neg)}")

    low = max(pos["precursor_mz"].quantile(0.01), neg["precursor_mz"].quantile(0.01))
    high = min(pos["precursor_mz"].quantile(0.99), neg["precursor_mz"].quantile(0.99))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = max(pos["precursor_mz"].min(), neg["precursor_mz"].min())
        high = min(pos["precursor_mz"].max(), neg["precursor_mz"].max())

    rng = np.random.RandomState(seed)
    targets = np.linspace(low, high, n_pairs)
    targets += rng.normal(0, (high - low) / max(n_pairs * 6, 1), size=n_pairs)
    targets = np.clip(targets, low, high)

    def choose_nearest(df, target_values):
        values = df["precursor_mz"].to_numpy(dtype=float)
        rows = df["row_idx"].to_numpy(dtype=np.int64)
        order = np.argsort(values)
        values = values[order]
        rows = rows[order]
        used = np.zeros(len(values), dtype=bool)
        chosen = []
        for target in target_values:
            ins = int(np.searchsorted(values, target))
            best_j, best_d = None, np.inf
            left = ins - 1
            right = ins
            while left >= 0 or right < len(values):
                progressed = False
                if left >= 0:
                    d = abs(values[left] - target)
                    if d < best_d and not used[left]:
                        best_j, best_d = left, d
                    progressed = True
                    left -= 1
                if right < len(values):
                    d = abs(values[right] - target)
                    if d < best_d and not used[right]:
                        best_j, best_d = right, d
                    progressed = True
                    right += 1
                if best_j is not None:
                    next_left = abs(values[left] - target) if left >= 0 else np.inf
                    next_right = abs(values[right] - target) if right < len(values) else np.inf
                    if min(next_left, next_right) > best_d:
                        break
                if not progressed:
                    break
            if best_j is None:
                raise RuntimeError("Could not choose a unique m/z-matched row")
            used[best_j] = True
            chosen.append((int(rows[best_j]), float(values[best_j])))
        return chosen

    pos_chosen = choose_nearest(pos, targets)
    neg_chosen = choose_nearest(neg, targets)
    rows = []
    for pair_id, ((pos_idx, pos_mz), (neg_idx, neg_mz)) in enumerate(zip(pos_chosen, neg_chosen)):
        delta = abs(pos_mz - neg_mz)
        rows.append({"sample_idx": pos_idx, "pair_id": pair_id, "polarity": "positive",
                     "adduct": "[M+H]+", "precursor_mz": pos_mz, "paired_mz_delta": delta})
        rows.append({"sample_idx": neg_idx, "pair_id": pair_id, "polarity": "negative",
                     "adduct": "[M-H]-", "precursor_mz": neg_mz, "paired_mz_delta": delta})
    selected = pd.DataFrame(rows)
    selected["plot_order"] = np.arange(len(selected), dtype=np.int64)
    return selected


def load_selected_samples(csv_path, selected):
    wanted = set(selected["sample_idx"].astype(int).tolist())
    sample_rows = {}
    usecols = ["mzs", "intensities", "smiles", "precursor_mz", "adduct", "collision_energy", "fold"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=20000):
        hit = chunk[chunk.index.isin(wanted)]
        for row_idx, row in hit.iterrows():
            mzs = np.array([float(x) for x in str(row["mzs"]).split(",")], dtype=np.float32)
            ints = np.array([float(x) for x in str(row["intensities"]).split(",")], dtype=np.float32)
            sample_rows[int(row_idx)] = {
                "sample_idx": int(row_idx),
                "spectrum": np.stack([mzs, ints], axis=-1),
                "precursor_mz": float(row["precursor_mz"]),
                "smiles": "" if str(row.get("smiles", "")) == "nan" else str(row.get("smiles", "")),
                "adduct": row.get("adduct"),
                "fold": row.get("fold"),
                "collision_energy": row.get("collision_energy"),
            }
        if len(sample_rows) == len(wanted):
            break

    missing = sorted(wanted - set(sample_rows))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected rows from CSV; first missing={missing[:5]}")
    return [sample_rows[int(i)] for i in selected.sort_values("plot_order")["sample_idx"]]


def embedding_cache_path(model_key):
    return os.path.join(DATA_DIR, f"ion_polarity_mzmatched_{model_key}_embeddings.npy")


def extract_embeddings(samples, model_key, device, batch_size, overwrite=False):
    path = embedding_cache_path(model_key)
    if os.path.exists(path) and not overwrite:
        return np.load(path)

    if model_key == "dreams":
        dreams_mod = _load_dreams_loader()
        model = dreams_mod.load_dreams_encoder(device)
        emb = dreams_mod.extract_embeddings(model, samples, device, batch_size=batch_size)
    else:
        mod = _load_comparison_module()
        model, _ = mod.load_model(model_key, device)
        emb = mod.EXTRACTORS[model_key](model, samples, device, bs=batch_size)
    np.save(path, emb.astype(np.float32))

    try:
        import torch
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return emb


def reduce_embedding(emb, method="umap", seed=42):
    from sklearn.preprocessing import StandardScaler, normalize
    x = StandardScaler().fit_transform(emb)
    x = normalize(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=35, learning_rate="auto",
                    init="pca", random_state=seed, metric="cosine").fit_transform(x)
    import umap
    return umap.UMAP(n_components=2, n_neighbors=35, min_dist=0.22,
                     metric="cosine", random_state=seed).fit_transform(x)


def separation_metrics(coords, polarity):
    from sklearn.metrics import silhouette_score
    labels = np.array([1 if p == "positive" else 0 for p in polarity])
    try:
        sil = float(silhouette_score(coords, labels))
    except Exception:
        sil = float("nan")
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    center_dist = float(np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0)))
    spread = float(np.mean([np.trace(np.cov(pos.T)), np.trace(np.cov(neg.T))]))
    return {
        "silhouette": sil,
        "center_distance": center_dist,
        "spread": spread,
        "separation_ratio": center_dist / np.sqrt(spread) if spread > 0 else float("nan"),
    }


def confidence_ellipse(ax, xy, color, n_std=1.8, lw=1.0):
    if len(xy) < 3:
        return
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=xy.mean(axis=0), width=width, height=height, angle=angle,
                  facecolor=color, edgecolor=color, lw=lw, alpha=0.12)
    ax.add_patch(ell)
    ax.add_patch(Ellipse(xy=xy.mean(axis=0), width=width, height=height, angle=angle,
                         facecolor="none", edgecolor=color, lw=lw, alpha=0.75))


def plot_projection(selected, coords_by_model, metrics_by_model, method):
    fig = plt.figure(figsize=(6.9, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.25], wspace=0.34)
    ax_mz = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    pos = selected[selected["polarity"] == "positive"]["precursor_mz"].to_numpy()
    neg = selected[selected["polarity"] == "negative"]["precursor_mz"].to_numpy()
    bins = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 26)
    ax_mz.hist(neg, bins=bins, density=True, histtype="stepfilled", color=C_NEG, alpha=0.18, lw=0)
    ax_mz.hist(pos, bins=bins, density=True, histtype="stepfilled", color=C_POS, alpha=0.18, lw=0)
    ax_mz.hist(neg, bins=bins, density=True, histtype="step", color=C_NEG, lw=1.1)
    ax_mz.hist(pos, bins=bins, density=True, histtype="step", color=C_POS, lw=1.1)
    ax_mz.set_xlabel("Precursor m/z")
    ax_mz.set_ylabel("Density")
    ax_mz.set_title("m/z-matched sample", pad=4)
    ax_mz.text(0.03, 0.96,
               f"500 + / 500 -\nmedian |Delta m/z|={selected['paired_mz_delta'].median():.2f}",
               transform=ax_mz.transAxes, ha="left", va="top", fontsize=6.5, color=C_GRAY)
    ax_mz.grid(axis="y", lw=0.3, alpha=0.35)

    model_specs = [
        ("rt_only_d", "UltraMs", C_ULTRA),
        ("dreams", "DreaMS", C_DREAM),
    ]
    polarity = selected.sort_values("plot_order")["polarity"].to_numpy()
    mask_pos = polarity == "positive"
    mask_neg = polarity == "negative"
    for ax, (model_key, label, title_color) in zip(axes, model_specs):
        coords = coords_by_model[model_key]
        confidence_ellipse(ax, coords[mask_neg], C_NEG)
        confidence_ellipse(ax, coords[mask_pos], C_POS)
        ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1], s=8, c=C_NEG, alpha=0.58,
                   linewidths=0, rasterized=True)
        ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1], s=8, c=C_POS, alpha=0.58,
                   linewidths=0, rasterized=True)
        ax.set_title(label, color=title_color, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        m = metrics_by_model[model_key]
        ax.text(0.02, 0.98,
                f"silhouette={m['silhouette']:.2f}\nsep={m['separation_ratio']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=C_GRAY)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#CCCCCC")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_NEG,
               markeredgecolor="none", markersize=5, label="Negative ion mode"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_POS,
               markeredgecolor="none", markersize=5, label="Positive ion mode"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 1.03),
               frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.2)
    return fig


def write_plot_data(selected, coords_by_model, metrics_by_model, method, args):
    rows = selected.sort_values("plot_order").copy()
    for model_key, coords in coords_by_model.items():
        rows[f"{model_key}_{method}1"] = coords[:, 0]
        rows[f"{model_key}_{method}2"] = coords[:, 1]
    rows.to_csv(os.path.join(DATA_DIR, f"ion_polarity_mzmatched_{method}_coords.csv"), index=False)
    summary = {
        "method": method,
        "n_pairs": int(args.n_pairs),
        "seed": int(args.seed),
        "split": args.split,
        "models": {"rt_only_d": "UltraMs Stage-D epoch 1", "dreams": "DreaMS"},
        "metrics": metrics_by_model,
        "sample": {
            "n_positive": int((selected["polarity"] == "positive").sum()),
            "n_negative": int((selected["polarity"] == "negative").sum()),
            "median_paired_mz_delta": float(selected["paired_mz_delta"].median()),
            "mean_paired_mz_delta": float(selected["paired_mz_delta"].mean()),
        },
    }
    with open(os.path.join(DATA_DIR, f"ion_polarity_mzmatched_{method}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ultra-batch-size", type=int, default=32)
    parser.add_argument("--dreams-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    coords_cache = os.path.join(DATA_DIR, f"ion_polarity_mzmatched_{args.method}_coords.csv")
    summary_cache = os.path.join(DATA_DIR, f"ion_polarity_mzmatched_{args.method}_summary.json")
    if os.path.exists(coords_cache) and os.path.exists(summary_cache) and not args.overwrite:
        selected = pd.read_csv(coords_cache)
        with open(summary_cache) as f:
            summary = json.load(f)
        coords_by_model = {
            "rt_only_d": selected[[f"rt_only_d_{args.method}1", f"rt_only_d_{args.method}2"]].to_numpy(dtype=float),
            "dreams": selected[[f"dreams_{args.method}1", f"dreams_{args.method}2"]].to_numpy(dtype=float),
        }
        metrics_by_model = summary["metrics"]
        fig = plot_projection(selected, coords_by_model, metrics_by_model, args.method)
        base = os.path.join(OUT_DIR, f"fig4e_ion_polarity_embedding_{args.method}")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{base}.{ext}")
        plt.close(fig)
        print(f"saved from cached coordinates: {base}.png/.pdf/.svg")
        return

    selected_path = os.path.join(DATA_DIR, "ion_polarity_mzmatched_samples.csv")
    if os.path.exists(selected_path) and not args.overwrite:
        selected = pd.read_csv(selected_path)
    else:
        selected = select_mz_matched_indices(MSNLIB_CSV, n_pairs=args.n_pairs,
                                             seed=args.seed, split=args.split)
        selected.to_csv(selected_path, index=False)

    samples = load_selected_samples(MSNLIB_CSV, selected)
    coords_by_model = {}
    metrics_by_model = {}
    for model_key, bs in [("rt_only_d", args.ultra_batch_size), ("dreams", args.dreams_batch_size)]:
        emb = extract_embeddings(samples, model_key, args.device, bs, overwrite=args.overwrite)
        coords = reduce_embedding(emb, method=args.method, seed=args.seed)
        coords_by_model[model_key] = coords
        metrics_by_model[model_key] = separation_metrics(
            coords, selected.sort_values("plot_order")["polarity"].to_numpy())

    write_plot_data(selected, coords_by_model, metrics_by_model, args.method, args)
    fig = plot_projection(selected, coords_by_model, metrics_by_model, args.method)
    base = os.path.join(OUT_DIR, f"fig4e_ion_polarity_embedding_{args.method}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)
    print(f"saved: {base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
