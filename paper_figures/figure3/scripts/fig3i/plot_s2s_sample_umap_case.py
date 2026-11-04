#!/usr/bin/env python3
"""Sample-level S2S case study for Figure 3.

This panel uses real Figure 4 application embeddings, not the reference
library index. It asks whether UltraMS retrieves repeated spectra of the same
anchor molecule from the sample-level spectrum embedding space.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIG_DIR = PROJECT_ROOT / "main_figure" / "fig3"
SOURCE_DIR = PROJECT_ROOT / "output" / "figure4_s2s_application"
EMBED_DIR = SOURCE_DIR / "stage3_s2s_mgf_embeddings"
MGF_DIR = SOURCE_DIR / "raw_mgf"
OUT_DIR = FIG_DIR / "output" / "task8_s2s_sample_umap"
DATA_DIR = OUT_DIR / "data"

TARGET_ANCHOR = "Proline betaine"
RANDOM_SEED = 42
MAX_BACKGROUND_SPECTRA = 3200
PRECURSOR_EXCLUSION_DA = 1.0


COLORS = {
    "UltraMS red": "#E64B35",
    "DreaMS cyan": "#4DBBD5",
    "blue": "#3C5488",
    "teal": "#00A087",
    "purple": "#7E57C2",
    "orange": "#F39B2F",
    "gray": "#B8C0CC",
    "dark": "#252A32",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def safe_umap(x: np.ndarray, *, n_neighbors: int, min_dist: float, seed: int) -> tuple[np.ndarray, str]:
    """Run UMAP with a deterministic fallback for environments without umap-learn."""
    x = np.asarray(x, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )
        return reducer.fit_transform(x), "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=seed).fit_transform(x)
        return coords, "PCA fallback"


def load_anchor_embeddings(candidates: pd.DataFrame, sample_table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    by_file = sample_table.set_index("mgf_file").to_dict("index")
    cache: dict[str, np.ndarray] = {}
    rows = []
    emb = []
    for idx, row in candidates.iterrows():
        info = by_file.get(row["query_mgf_file"])
        if info is None:
            continue
        emb_path = str(PROJECT_ROOT / info["embedding_path"])
        if emb_path not in cache:
            cache[emb_path] = np.load(emb_path, mmap_mode="r")
        local_idx = int(row["query_local_spectrum_index"])
        shard = cache[emb_path]
        if local_idx >= len(shard):
            continue
        rows.append(idx)
        emb.append(np.asarray(shard[local_idx], dtype=np.float32))
    out = candidates.loc[rows].reset_index(drop=True)
    x = np.vstack(emb).astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return out, x


def choose_query(candidates: pd.DataFrame, x: np.ndarray, anchor: str) -> int:
    anchor_mask = candidates["anchor_name"].eq(anchor).to_numpy()
    bio_mask = ~candidates["comparison_label"].isin(["QC", "Blank"]).to_numpy()
    eligible = np.where(anchor_mask & bio_mask)[0]
    if len(eligible) == 0:
        eligible = np.where(anchor_mask)[0]
    centroid = x[anchor_mask].mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    return int(eligible[np.argmax(x[eligible] @ centroid)])


def load_background_embeddings(sample_row: pd.Series, max_n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(PROJECT_ROOT / sample_row["metadata_path"])
    x = np.load(PROJECT_ROOT / sample_row["embedding_path"], mmap_mode="r")
    rng = np.random.default_rng(seed)
    n = len(meta)
    if n > max_n:
        chosen = np.sort(rng.choice(np.arange(n), size=max_n, replace=False))
    else:
        chosen = np.arange(n)
    bg = meta.iloc[chosen].copy().reset_index(drop=True)
    emb = np.asarray(x[chosen], dtype=np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return bg, emb


def read_mgf_spectrum(path: Path, local_index: int) -> np.ndarray:
    current = -1
    mzs: list[float] = []
    ints: list[float] = []
    inside = False
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                current += 1
                inside = current == local_index
                mzs = []
                ints = []
                continue
            if not inside:
                continue
            if line == "END IONS":
                arr = np.column_stack([mzs, ints]).astype(np.float32)
                if len(arr) == 0:
                    return arr
                arr[:, 1] = arr[:, 1] / max(float(arr[:, 1].max()), 1e-12)
                return arr
            if not line or "=" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mzs.append(float(parts[0]))
                    ints.append(float(parts[1]))
                except ValueError:
                    pass
    raise IndexError(f"local spectrum index {local_index} not found in {path}")


def simplify_label(label: str) -> str:
    replacements = {
        "Adenosine monophosphate": "AMP",
        "Proline betaine": "Proline betaine",
        "Dihydrothymine": "Dihydrothymine",
        "L-Valine": "L-Valine",
        "Proline": "Proline",
    }
    return replacements.get(label, label)


def draw_stem(ax: plt.Axes, spectrum: np.ndarray, *, color: str, yscale: float = 1.0, label: str | None = None) -> None:
    if len(spectrum) == 0:
        return
    y = spectrum[:, 1] * yscale
    ax.vlines(spectrum[:, 0], 0, y, color=color, lw=0.75, alpha=0.9, label=label)


def peak_level_cosine(a: np.ndarray, b: np.ndarray, tol_da: float = 0.05) -> tuple[float, int]:
    """Simple peak-level cosine after greedy m/z matching."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0
    a = a[np.argsort(a[:, 0])]
    b = b[np.argsort(b[:, 0])]
    ai = a[:, 1] / max(float(np.linalg.norm(a[:, 1])), 1e-12)
    bi = b[:, 1] / max(float(np.linalg.norm(b[:, 1])), 1e-12)
    i = 0
    j = 0
    score = 0.0
    n_matches = 0
    while i < len(a) and j < len(b):
        delta = float(a[i, 0] - b[j, 0])
        if abs(delta) <= tol_da:
            best = j
            best_delta = abs(delta)
            k = j + 1
            while k < len(b) and abs(float(a[i, 0] - b[k, 0])) <= tol_da:
                candidate_delta = abs(float(a[i, 0] - b[k, 0]))
                if candidate_delta < best_delta:
                    best = k
                    best_delta = candidate_delta
                k += 1
            score += float(ai[i] * bi[best])
            n_matches += 1
            i += 1
            j = best + 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return float(score), int(n_matches)


def fragment_only_spectrum(spectrum: np.ndarray, precursor_mz: float, exclusion_da: float = PRECURSOR_EXCLUSION_DA) -> np.ndarray:
    """Remove precursor/near-precursor peaks and renormalize remaining fragments."""
    if len(spectrum) == 0 or not np.isfinite(float(precursor_mz)):
        return spectrum.copy()
    keep = np.abs(spectrum[:, 0] - float(precursor_mz)) > exclusion_da
    fragments = spectrum[keep].copy()
    if len(fragments) == 0:
        return spectrum.copy()
    fragments[:, 1] = fragments[:, 1] / max(float(fragments[:, 1].max()), 1e-12)
    return fragments


def choose_semantic_same_match(
    rank_table: pd.DataFrame,
    query_fragment_spectrum: np.ndarray,
    *,
    target_anchor: str,
    max_rank: int,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Choose a non-rank-1 same-molecule spectrum with less direct peak overlap."""
    target = rank_table.loc[
        rank_table["is_target_anchor"] & rank_table["rank"].gt(1) & rank_table["rank"].le(max_rank)
    ].copy()
    if target.empty:
        target = rank_table.loc[rank_table["anchor_name"].eq(target_anchor) & rank_table["rank"].gt(1)].head(10).copy()

    scored = []
    for _, row in target.iterrows():
        spec = read_mgf_spectrum(MGF_DIR / row["query_mgf_file"], int(row["query_local_spectrum_index"]))
        spec_fragments = fragment_only_spectrum(spec, float(row["query_precursor_mz"]))
        peak_cosine, n_matches = peak_level_cosine(query_fragment_spectrum, spec_fragments)
        row = row.copy()
        row["fragment_peak_cosine_to_seed"] = peak_cosine
        row["n_fragment_peak_matches_to_seed"] = n_matches
        row["n_fragment_peaks"] = int(len(spec_fragments))
        scored.append((peak_cosine, -float(row["s2s_similarity"]), int(row["rank"]), row, spec, spec_fragments))
    if not scored:
        raise RuntimeError(f"No non-rank-1 same-molecule spectrum found for {target_anchor}")
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, best_row, best_spectrum, best_fragments = scored[0]
    return best_row, best_spectrum, best_fragments


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_table = pd.read_csv(EMBED_DIR / "sample_centroid_table.csv")
    sample_embeddings = np.load(EMBED_DIR / "sample_centroid_embeddings.npy")
    sample_embeddings = np.asarray(sample_embeddings, dtype=np.float32)
    sample_coords, sample_method = safe_umap(sample_embeddings, n_neighbors=15, min_dist=0.25, seed=RANDOM_SEED)
    sample_points = sample_table.copy()
    sample_points["umap1"] = sample_coords[:, 0]
    sample_points["umap2"] = sample_coords[:, 1]

    candidates_raw = pd.read_csv(EMBED_DIR / "anchor_spectrum_candidates.csv")
    candidates, candidate_embeddings = load_anchor_embeddings(candidates_raw, sample_table)
    positive_mask = candidates["anchor_polarity"].eq("positive").to_numpy()
    candidates_pos = candidates.loc[positive_mask].reset_index(drop=True)
    x_pos = candidate_embeddings[positive_mask]

    query_idx = choose_query(candidates_pos, x_pos, TARGET_ANCHOR)
    query = candidates_pos.iloc[query_idx].copy()
    sims = x_pos @ x_pos[query_idx]
    sims[query_idx] = -np.inf
    order = np.argsort(sims)[::-1]
    rank_table = candidates_pos.iloc[order].copy().reset_index(drop=True)
    rank_table["rank"] = np.arange(1, len(rank_table) + 1)
    rank_table["s2s_similarity"] = sims[order]
    rank_table["is_target_anchor"] = rank_table["anchor_name"].eq(TARGET_ANCHOR)
    first_distractor = rank_table.loc[~rank_table["is_target_anchor"]].iloc[0]
    target_label = simplify_label(TARGET_ANCHOR)
    off_target_label = simplify_label(first_distractor["anchor_name"])

    sample_row = sample_table.loc[sample_table["mgf_file"].eq(query["query_mgf_file"])].iloc[0]
    bg_meta, bg_embeddings = load_background_embeddings(sample_row, MAX_BACKGROUND_SPECTRA, RANDOM_SEED)
    combined_embeddings = np.vstack([bg_embeddings, x_pos])
    spectrum_coords, spectrum_method = safe_umap(
        combined_embeddings,
        n_neighbors=30,
        min_dist=0.12,
        seed=RANDOM_SEED,
    )
    bg_points = bg_meta.copy()
    bg_points["umap1"] = spectrum_coords[: len(bg_points), 0]
    bg_points["umap2"] = spectrum_coords[: len(bg_points), 1]
    candidate_points = candidates_pos.copy()
    candidate_points["umap1"] = spectrum_coords[len(bg_points) :, 0]
    candidate_points["umap2"] = spectrum_coords[len(bg_points) :, 1]
    candidate_points["is_target_anchor"] = candidate_points["anchor_name"].eq(TARGET_ANCHOR)
    candidate_points["is_query"] = False
    candidate_points.loc[query_idx, "is_query"] = True
    candidate_points["display_anchor"] = candidate_points["anchor_name"].map(simplify_label)

    sample_points["has_target_anchor"] = sample_points["mgf_file"].isin(
        candidates_pos.loc[candidates_pos["anchor_name"].eq(TARGET_ANCHOR), "query_mgf_file"]
    )
    sample_points["is_query_sample"] = sample_points["mgf_file"].eq(query["query_mgf_file"])

    query_spectrum = read_mgf_spectrum(MGF_DIR / query["query_mgf_file"], int(query["query_local_spectrum_index"]))
    query_fragment_spectrum = fragment_only_spectrum(query_spectrum, float(query["query_precursor_mz"]))
    semantic_same, same_spectrum, same_fragment_spectrum = choose_semantic_same_match(
        rank_table,
        query_fragment_spectrum,
        target_anchor=TARGET_ANCHOR,
        max_rank=max(2, int(first_distractor["rank"]) - 1),
    )
    distractor_spectrum = read_mgf_spectrum(
        MGF_DIR / first_distractor["query_mgf_file"], int(first_distractor["query_local_spectrum_index"])
    )
    distractor_fragment_spectrum = fragment_only_spectrum(
        distractor_spectrum,
        float(first_distractor["query_precursor_mz"]),
    )

    # Persist enough data for replotting without recomputing embeddings or rerunning UMAP.
    sample_points.to_csv(DATA_DIR / "sample_umap_points.csv", index=False)
    bg_points.to_csv(DATA_DIR / "sample_background_spectrum_umap_points.csv", index=False)
    candidate_points.to_csv(DATA_DIR / "anchor_candidate_spectrum_umap_points.csv", index=False)
    rank_table.to_csv(DATA_DIR / "query_s2s_rank_table.csv", index=False)
    spectra_long = []
    for role, row, view, spec in [
        ("seed_msms_spectrum", query, "full", query_spectrum),
        ("seed_msms_spectrum", query, "fragment_only", query_fragment_spectrum),
        ("semantic_same_anchor", semantic_same, "full", same_spectrum),
        ("semantic_same_anchor", semantic_same, "fragment_only", same_fragment_spectrum),
        ("nearest_distractor_anchor", first_distractor, "full", distractor_spectrum),
        ("nearest_distractor_anchor", first_distractor, "fragment_only", distractor_fragment_spectrum),
    ]:
        for mz, intensity in spec:
            spectra_long.append(
                {
                    "role": role,
                    "spectrum_view": view,
                    "anchor_name": row["anchor_name"],
                    "sample_name": row["sample_name"],
                    "mgf_file": row["query_mgf_file"],
                    "local_spectrum_index": int(row["query_local_spectrum_index"]),
                    "mz": float(mz),
                    "relative_intensity": float(intensity),
                }
            )
    pd.DataFrame(spectra_long).to_csv(DATA_DIR / "spectra_overlay_long.csv", index=False)

    summary = {
        "target_anchor": TARGET_ANCHOR,
        "sample_umap_method": sample_method,
        "spectrum_umap_method": spectrum_method,
        "selected_source_run": {
            "sample_name": query["sample_name"],
            "comparison_label": query["comparison_label"],
            "mgf_file": query["query_mgf_file"],
        },
        "seed_msms_spectrum": query.to_dict(),
        "semantic_same_anchor": semantic_same.to_dict(),
        "nearest_distractor_anchor": first_distractor.to_dict(),
        "top_k_target_counts": {
            str(k): int(rank_table.head(k)["is_target_anchor"].sum()) for k in [10, 20, 50, 100]
        },
        "source_files": {
            "sample_centroid_table": str(EMBED_DIR / "sample_centroid_table.csv"),
            "sample_centroid_embeddings": str(EMBED_DIR / "sample_centroid_embeddings.npy"),
            "anchor_spectrum_candidates": str(EMBED_DIR / "anchor_spectrum_candidates.csv"),
            "raw_mgf_dir": str(MGF_DIR),
        },
        "random_seed": RANDOM_SEED,
        "max_background_spectra": MAX_BACKGROUND_SPECTRA,
        "precursor_exclusion_da": PRECURSOR_EXCLUSION_DA,
    }
    with (DATA_DIR / "plot_data.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    fig = plt.figure(figsize=(15.2, 3.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.35, 1.0, 1.22], wspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    group_colors = {
        "Symptomatic": COLORS["UltraMS red"],
        "Asymptomatic": COLORS["blue"],
        "QC": "#9AA5B1",
        "Blank": "#D3D7DE",
    }
    for group, sub in sample_points.groupby("comparison_label"):
        ax0.scatter(
            sub["umap1"],
            sub["umap2"],
            s=18 if group in {"Symptomatic", "Asymptomatic"} else 13,
            c=group_colors.get(group, COLORS["gray"]),
            edgecolor="white",
            linewidth=0.25,
            alpha=0.88 if group in {"Symptomatic", "Asymptomatic"} else 0.55,
            label=group,
        )
    target_samples = sample_points.loc[sample_points["has_target_anchor"]]
    ax0.scatter(
        target_samples["umap1"],
        target_samples["umap2"],
        s=42,
        facecolors="none",
        edgecolors=COLORS["teal"],
        linewidth=0.9,
        label=f"{target_label} candidate run",
    )
    query_sample = sample_points.loc[sample_points["is_query_sample"]].iloc[0]
    ax0.scatter(
        [query_sample["umap1"]],
        [query_sample["umap2"]],
        marker="*",
        s=130,
        c=COLORS["orange"],
        edgecolor="black",
        linewidth=0.45,
        zorder=6,
        label=f"{target_label} source run",
    )
    ax0.set_title("Cohort sample map")
    ax0.set_xlabel(f"{sample_method} 1")
    ax0.set_ylabel(f"{sample_method} 2")
    ax0.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)

    ax1.scatter(
        bg_points["umap1"],
        bg_points["umap2"],
        s=4,
        c="#D7DCE3",
        alpha=0.38,
        linewidths=0,
        label=f"{query['sample_name']} run spectra",
    )
    other = candidate_points.loc[~candidate_points["is_target_anchor"]]
    top_other_names = list(other["display_anchor"].value_counts().head(4).index)
    other_palette = [COLORS["purple"], COLORS["orange"], "#8491B4", "#91D1C2"]
    for color, name in zip(other_palette, top_other_names):
        sub = other.loc[other["display_anchor"].eq(name)]
        ax1.scatter(sub["umap1"], sub["umap2"], s=18, c=color, alpha=0.72, linewidths=0, label=name)
    remaining = other.loc[~other["display_anchor"].isin(top_other_names)]
    if len(remaining):
        ax1.scatter(remaining["umap1"], remaining["umap2"], s=13, c="#C8CDD5", alpha=0.48, linewidths=0, label="other anchors")
    target = candidate_points.loc[candidate_points["is_target_anchor"]]
    ax1.scatter(
        target["umap1"],
        target["umap2"],
        s=24,
        c=COLORS["teal"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.25,
        label=target_label,
    )
    qpt = candidate_points.iloc[query_idx]
    ax1.scatter(
        [qpt["umap1"]],
        [qpt["umap2"]],
        s=115,
        marker="*",
        c=COLORS["UltraMS red"],
        edgecolor="black",
        linewidth=0.5,
        zorder=8,
        label=f"{target_label} MS/MS spectrum",
    )
    ax1.set_title(f"{target_label} spectrum neighborhood")
    ax1.set_xlabel(f"{spectrum_method} 1")
    ax1.set_ylabel(f"{spectrum_method} 2")
    ax1.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1,
        markerscale=1.1,
        handletextpad=0.25,
        borderpad=0.2,
    )

    top_n = 90
    top = rank_table.head(top_n).copy()
    bar_colors = np.where(top["is_target_anchor"], COLORS["teal"], "#BFC6D0")
    ax2.bar(top["rank"], top["s2s_similarity"], color=bar_colors, width=0.82, linewidth=0)
    first_decoy_rank = int(first_distractor["rank"])
    ax2.axvline(first_decoy_rank, color=COLORS["UltraMS red"], lw=0.9, ls="--")
    ax2.text(
        first_decoy_rank + 1.2,
        min(0.985, float(top["s2s_similarity"].max()) - 0.005),
        f"first off-target\n{off_target_label}, rank {first_decoy_rank}",
        color=COLORS["UltraMS red"],
        ha="left",
        va="top",
        fontsize=6.7,
    )
    for k, y in [(10, 0.75), (20, 0.69), (50, 0.63)]:
        ax2.text(
            4,
            y,
            f"top {k}: {int(rank_table.head(k)['is_target_anchor'].sum())}/{k} {target_label}",
            color=COLORS["dark"],
            fontsize=7,
        )
    ax2.set_ylim(0, 1.03)
    ax2.set_xlim(0, top_n + 1)
    ax2.set_title(f"{target_label} spectrum S2S retrieval")
    ax2.set_xlabel("Nearest-neighbor rank")
    ax2.set_ylabel("Cosine similarity")

    same_color = COLORS["teal"]
    distractor_color = COLORS["purple"]
    query_color = COLORS["dark"]
    draw_stem(ax3, query_fragment_spectrum, color=query_color, yscale=1.0, label=f"{target_label} fragments")
    draw_stem(ax3, same_fragment_spectrum, color=same_color, yscale=-0.92, label=f"matched {target_label}")
    ax3.axhline(0, color="#3A3A3A", lw=0.55)
    ax3.text(
        0.34,
        0.025,
        f"off-target: {off_target_label}, rank {first_decoy_rank}",
        transform=ax3.transAxes,
        color=distractor_color,
        ha="center",
        va="bottom",
        fontsize=7,
    )
    mz_min = min(
        np.min(query_fragment_spectrum[:, 0]),
        np.min(same_fragment_spectrum[:, 0]),
        np.min(distractor_fragment_spectrum[:, 0]),
    )
    mz_max = max(
        np.max(query_fragment_spectrum[:, 0]),
        np.max(same_fragment_spectrum[:, 0]),
        np.max(distractor_fragment_spectrum[:, 0]),
    )
    ax3.set_xlim(max(0, mz_min - 8), mz_max + 8)
    ax3.set_ylim(-1.05, 1.05)
    ax3.set_title(f"{target_label} fragment evidence")
    ax3.set_xlabel("m/z")
    ax3.set_ylabel("Relative intensity")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["matched", "0", target_label])

    ax3c = ax3.inset_axes([0.72, 0.70, 0.26, 0.22])
    metric_values = [
        float(semantic_same["s2s_similarity"]),
        float(semantic_same["fragment_peak_cosine_to_seed"]),
    ]
    metric_labels = ["embedding", "spectrum"]
    metric_colors = [same_color, "#7A8594"]
    ypos = np.array([0.58, 0.42])
    bars = ax3c.barh(ypos, metric_values, color=metric_colors, height=0.12, linewidth=0)
    ax3c.set_xlim(0, 1.05)
    ax3c.set_ylim(0.30, 0.70)
    ax3c.set_xticks([0, 0.5, 1.0])
    ax3c.set_xticklabels(["0", "0.5", "1"], fontsize=5.2)
    ax3c.set_yticks(ypos)
    ax3c.set_yticklabels(metric_labels, fontsize=5.6)
    ax3c.set_title(f"rank {int(semantic_same['rank'])}", fontsize=6.2, pad=1.0)
    ax3c.tick_params(length=1.8, width=0.45, pad=1.2)
    for bar, value in zip(bars, metric_values):
        ax3c.text(
            max(value - 0.04, 0.08) if value > 0.45 else min(value + 0.04, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="right" if value > 0.45 else "left",
            va="center",
            fontsize=5.8,
            color="white" if value > 0.45 else COLORS["dark"],
        )
    ax3c.spines["top"].set_visible(False)
    ax3c.spines["right"].set_visible(False)
    for spine in ax3c.spines.values():
        spine.set_linewidth(0.45)

    ax3b = ax3.inset_axes([0.16, 0.08, 0.36, 0.36])
    draw_stem(ax3b, query_fragment_spectrum, color=query_color, yscale=1.0)
    draw_stem(ax3b, distractor_fragment_spectrum, color=distractor_color, yscale=-0.92)
    ax3b.axhline(0, color="#4A4A4A", lw=0.45)
    ax3b.set_xlim(ax3.get_xlim())
    ax3b.set_ylim(-1.05, 1.05)
    ax3b.set_xticks([])
    ax3b.set_yticks([])
    ax3b.set_title(f"{target_label} vs {off_target_label}", fontsize=6.4, pad=1)
    for spine in ax3b.spines.values():
        spine.set_linewidth(0.45)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.8, width=0.6)

    fig.suptitle(
        "Sample-level spectrum-to-spectrum retrieval case study",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.18, wspace=0.38)

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"row3_s2s_sample_umap_case.{suffix}", bbox_inches="tight")
    plt.close(fig)

    print(f"Selected anchor: {TARGET_ANCHOR}")
    print(f"Selected source run: {query['sample_name']} ({query['comparison_label']}), {query['query_mgf_file']}")
    print(
        "Top-k target counts:",
        {k: int(rank_table.head(k)["is_target_anchor"].sum()) for k in [10, 20, 50, 100]},
    )
    print(
        f"First off-target anchor: rank {first_decoy_rank}, {first_distractor['anchor_name']}, "
        f"sim={first_distractor['s2s_similarity']:.3f}"
    )
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
