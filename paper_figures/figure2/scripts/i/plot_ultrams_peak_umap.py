#!/usr/bin/env python
"""Single-panel UltraMS peak-token UMAP from MAGMa-annotated MSnLib peaks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = {
    "Carbohydrate": "#2fb7a3",
    "Carbohydrate with N": "#2f6fc2",
    "Carbohydrate with S/P": "#9467bd",
    "Carbohydrate with N and S": "#7b52ab",
    "Carbohydrate with N and P": "#7b52ab",
    "Halogenated": "#ee6677",
    "N-containing": "#3366cc",
    "S/P-containing": "#8d70c9",
    "O-rich": "#22aa99",
    "Hydrocarbon": "#f2a541",
    "Lipid-like": "#d9902f",
    "Peptide-like": "#4c78a8",
    "Aromatic O-rich": "#72b7b2",
    "N-heteroaromatic": "#1f77b4",
    "Aromatic carbocycle": "#b279a2",
    "Single aromatic ring": "#9c755f",
    "Polycyclic aromatic": "#755f9c",
    "Heteroaromatic ring": "#2f6fc2",
    "Saturated ring": "#59a14f",
    "Acyclic": "#8cd17d",
    "Mixed ring system": "#bcbd22",
    "CHO-rich fragment": "#2fb7a3",
    "CHON fragment": "#2f6fc2",
    "CHOS/P fragment": "#9467bd",
    "Halogenated fragment": "#ee6677",
    "Aromatic fragment": "#b279a2",
    "Small fragment (<100 Da)": "#6b6ecf",
    "Mid fragment (100-200 Da)": "#17becf",
    "Large fragment (200-350 Da)": "#ff9da7",
    "Very large fragment (>350 Da)": "#e15759",
    "Aromatic fragment": "#b279a2",
    "Heteroaromatic fragment": "#2f6fc2",
    "Aliphatic fragment": "#59a14f",
    "Unknown fragment": "#b9bec6",
    "Small loss (<50 Da)": "#4e79a7",
    "Mid loss (50-150 Da)": "#59a14f",
    "Large loss (150-300 Da)": "#f28e2b",
    "Very large loss (>300 Da)": "#e15759",
    "Other": "#8b919a",
}

COLOR_SPECS = {
    "frag_label": ("categorical", "MAGMa fragment class", None),
    "molecule_label": ("categorical", "Parent molecule class", None),
    "parent_biofamily": ("categorical", "Parent biofamily", None),
    "parent_scaffold": ("categorical", "Parent scaffold class", None),
    "fragment_chemotype": ("categorical", "Fragment chemotype", None),
    "fragment_aromaticity": ("categorical", "Fragment aromaticity", None),
    "peak_mz_bin": ("categorical", "Fragment m/z bin", None),
    "parent_mw": ("continuous", "Parent molecular weight", "viridis"),
    "parent_logp": ("continuous", "Parent cLogP", "coolwarm"),
    "parent_tpsa": ("continuous", "Parent TPSA", "YlGnBu"),
    "parent_hetero": ("continuous", "Parent hetero atoms", "magma"),
    "parent_rings": ("continuous", "Parent ring count", "cividis"),
    "parent_aromatic_fraction": ("continuous", "Parent aromatic atom fraction", "cividis"),
    "parent_o_c_ratio": ("continuous", "Parent O/C ratio", "YlGnBu"),
    "peak_mz": ("continuous", "Fragment peak m/z", "magma"),
    "neutral_loss_da": ("continuous", "Precursor - fragment m/z", "inferno_r"),
    "neutral_loss_bin": ("categorical", "Neutral loss bin", None),
    "fragment_heavy_atoms": ("continuous", "Fragment heavy atoms", "viridis"),
    "fragment_hetero_atoms": ("continuous", "Fragment hetero atoms", "magma"),
    "fragment_o_c_ratio": ("continuous", "Fragment O/C ratio", "YlGnBu"),
    "intensity": ("continuous", "Peak intensity", "plasma"),
    "match_rate": ("continuous", "MAGMa spectrum match rate", "viridis"),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def balanced_sample(meta: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    if len(meta) <= max_points:
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    labels = meta["frag_label"].to_numpy()
    groups = [np.where(labels == lab)[0] for lab in sorted(set(labels))]
    per_group = max(1, max_points // len(groups))
    idx: list[int] = []
    for group in groups:
        idx.extend(rng.choice(group, size=min(len(group), per_group), replace=False).tolist())
    if len(idx) < max_points:
        rest = np.setdiff1d(np.arange(len(meta)), np.asarray(idx), assume_unique=False)
        idx.extend(rng.choice(rest, size=min(len(rest), max_points - len(idx)), replace=False).tolist())
    return np.asarray(idx, dtype=int)


def fit_umap(
    x: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    n_epochs: int,
) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        init="random",
        n_epochs=n_epochs,
        random_state=seed,
        low_memory=True,
    )
    return reducer.fit_transform(l2_normalize(x))


def formula_counts(formula: str) -> dict[str, int]:
    import re
    counts: dict[str, int] = {}
    if not isinstance(formula, str):
        return counts
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def parent_properties(meta: pd.DataFrame) -> pd.DataFrame:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except Exception:
        return meta

    props: dict[str, dict[str, float]] = {}
    for smi in sorted(set(meta["smiles"].fillna("").astype(str))):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            props[smi] = {
                "parent_mw": np.nan,
                "parent_logp": np.nan,
                "parent_tpsa": np.nan,
                "parent_hetero": np.nan,
                "parent_rings": np.nan,
                "parent_aromatic_fraction": np.nan,
                "parent_o_c_ratio": np.nan,
                "parent_biofamily": "Other",
                "parent_scaffold": "Other",
            }
            continue
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        counts = {elem: atoms.count(elem) for elem in set(atoms)}
        c = counts.get("C", 0)
        n = counts.get("N", 0)
        o = counts.get("O", 0)
        s = counts.get("S", 0)
        p = counts.get("P", 0)
        hal = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        rings = int(rdMolDescriptors.CalcNumRings(mol))
        aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
        heavy_atoms = max(1, mol.GetNumHeavyAtoms())
        aromatic_n = any(atom.GetIsAromatic() and atom.GetSymbol() == "N" for atom in mol.GetAtoms())
        amide_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3][CX3](=O)")))

        carbohydrate_like = c >= 3 and o >= 3 and (o / max(c, 1)) >= 0.42 and logp < 2.5
        if hal:
            biofamily = "Halogenated"
        elif carbohydrate_like and n and s:
            biofamily = "Carbohydrate with N and S"
        elif carbohydrate_like and n and p:
            biofamily = "Carbohydrate with N and P"
        elif carbohydrate_like and n:
            biofamily = "Carbohydrate with N"
        elif carbohydrate_like and (s or p):
            biofamily = "Carbohydrate with S/P"
        elif carbohydrate_like:
            biofamily = "Carbohydrate"
        elif c >= 12 and (logp >= 2.5 or (o <= 4 and n <= 1)) and rings <= 2:
            biofamily = "Lipid-like"
        elif n >= 2 and o >= 2 and amide_count:
            biofamily = "Peptide-like"
        elif aromatic_n:
            biofamily = "N-heteroaromatic"
        elif aromatic_atoms and o >= 2:
            biofamily = "Aromatic O-rich"
        elif s or p:
            biofamily = "S/P-containing"
        elif n:
            biofamily = "N-containing"
        elif o >= 3:
            biofamily = "O-rich"
        else:
            biofamily = "Other"

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if not atom_rings:
            scaffold = "Acyclic"
        else:
            has_aromatic_ring = False
            has_hetero_ring = False
            has_saturated_ring = False
            for ring in atom_rings:
                ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                aromatic_ring = all(atom.GetIsAromatic() for atom in ring_atoms)
                hetero_ring = any(atom.GetAtomicNum() not in (6, 1) for atom in ring_atoms)
                has_aromatic_ring = has_aromatic_ring or aromatic_ring
                has_hetero_ring = has_hetero_ring or hetero_ring
                has_saturated_ring = has_saturated_ring or not aromatic_ring
            if has_aromatic_ring and has_hetero_ring:
                scaffold = "Heteroaromatic ring"
            elif has_aromatic_ring and len(atom_rings) > 1:
                scaffold = "Polycyclic aromatic"
            elif has_aromatic_ring:
                scaffold = "Single aromatic ring"
            elif has_saturated_ring and has_hetero_ring:
                scaffold = "Mixed ring system"
            elif has_saturated_ring:
                scaffold = "Saturated ring"
            else:
                scaffold = "Other"

        props[smi] = {
            "parent_mw": float(Descriptors.ExactMolWt(mol)),
            "parent_logp": logp,
            "parent_tpsa": tpsa,
            "parent_hetero": float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
            "parent_rings": float(rings),
            "parent_aromatic_fraction": float(len(aromatic_atoms) / heavy_atoms),
            "parent_o_c_ratio": float(o / max(c, 1)),
            "parent_biofamily": biofamily,
            "parent_scaffold": scaffold,
        }
    prop_df = pd.DataFrame.from_dict(props, orient="index")
    for col in prop_df.columns:
        meta[col] = meta["smiles"].astype(str).map(prop_df[col])
    return meta


def add_precursor_context(meta: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "msnlib_magma_subset.csv"
    if not path.exists() or "spec_id" not in meta.columns:
        meta["precursor_mz"] = np.nan
        meta["neutral_loss_da"] = np.nan
        return meta
    subset = pd.read_csv(path, usecols=lambda c: c in {"spec_id", "precursor_mz"})
    prec = subset.drop_duplicates("spec_id").set_index("spec_id")["precursor_mz"]
    meta["precursor_mz"] = meta["spec_id"].astype(str).map(prec).astype(float)
    meta["neutral_loss_da"] = meta["precursor_mz"] - meta["mz"].astype(float)
    return meta


def add_discrete_classes(meta: pd.DataFrame) -> pd.DataFrame:
    try:
        from rdkit import Chem
    except Exception:
        Chem = None

    fragment_labels = []
    fragment_aromaticity = []
    fragment_heavy_atoms = []
    fragment_hetero_atoms = []
    fragment_o_c_ratio = []
    for _, row in meta.iterrows():
        counts = formula_counts(row.get("frag_formula", ""))
        c = counts.get("C", 0)
        n = counts.get("N", 0)
        o = counts.get("O", 0)
        s = counts.get("S", 0)
        p = counts.get("P", 0)
        hal = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
        raw_smi = row.get("frag_smiles", "")
        smi = "" if pd.isna(raw_smi) else str(raw_smi)
        if smi.lower() == "nan":
            smi = ""
        aromatic = False
        heteroaromatic = False
        heavy_from_mol = np.nan
        hetero_from_mol = np.nan
        if Chem is not None and smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
                heteroaromatic = any(atom.GetIsAromatic() and atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms())
                heavy_from_mol = float(mol.GetNumHeavyAtoms())
                hetero_from_mol = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
        if hal:
            label = "Halogenated fragment"
        elif c >= 2 and o >= 2 and n and (s or p):
            label = "CHOS/P fragment"
        elif c >= 2 and o >= 2 and n:
            label = "CHON fragment"
        elif c >= 2 and o >= 2:
            label = "CHO-rich fragment"
        elif aromatic:
            label = "Aromatic fragment"
        elif set(counts).issubset({"C", "H"}):
            label = "Hydrocarbon"
        elif s or p:
            label = "S/P-containing"
        elif n:
            label = "N-containing"
        else:
            label = "Other"
        fragment_labels.append(label)
        if heteroaromatic:
            fragment_aromaticity.append("Heteroaromatic fragment")
        elif aromatic:
            fragment_aromaticity.append("Aromatic fragment")
        elif smi:
            fragment_aromaticity.append("Aliphatic fragment")
        else:
            fragment_aromaticity.append("Unknown fragment")
        fragment_heavy_atoms.append(float(np.nan_to_num(heavy_from_mol, nan=sum(v for k, v in counts.items() if k != "H"))))
        fragment_hetero_atoms.append(float(np.nan_to_num(hetero_from_mol, nan=sum(v for k, v in counts.items() if k not in {"C", "H"}))))
        fragment_o_c_ratio.append(float(o / max(c, 1)))
    meta["fragment_chemotype"] = fragment_labels
    meta["fragment_aromaticity"] = fragment_aromaticity
    meta["fragment_heavy_atoms"] = fragment_heavy_atoms
    meta["fragment_hetero_atoms"] = fragment_hetero_atoms
    meta["fragment_o_c_ratio"] = fragment_o_c_ratio

    meta["peak_mz_bin"] = pd.cut(
        meta["peak_mz"].astype(float),
        bins=[-np.inf, 100, 200, 350, np.inf],
        labels=["Small fragment (<100 Da)", "Mid fragment (100-200 Da)", "Large fragment (200-350 Da)", "Very large fragment (>350 Da)"],
    ).astype(str)
    meta["neutral_loss_bin"] = pd.cut(
        meta["neutral_loss_da"].astype(float),
        bins=[-np.inf, 50, 150, 300, np.inf],
        labels=["Small loss (<50 Da)", "Mid loss (50-150 Da)", "Large loss (150-300 Da)", "Very large loss (>300 Da)"],
    ).astype(str)
    return meta


def draw_fragment_insets(ax, coords: np.ndarray, meta: pd.DataFrame, max_mols: int = 6) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    except Exception:
        return

    valid = meta[meta["frag_smiles"].fillna("").astype(str) != ""].copy()
    candidates: list[str] = []
    for _, group in valid.groupby("frag_label"):
        ranked = (
            group.groupby("frag_smiles")
            .agg(n=("frag_smiles", "size"), match_rate=("match_rate", "mean"))
            .sort_values(["n", "match_rate"], ascending=False)
            .head(3)
            .index
            .tolist()
        )
        candidates.extend(ranked)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xrange = max(xmax - xmin, 1e-6)
    yrange = max(ymax - ymin, 1e-6)
    used_xy: list[np.ndarray] = []
    used_boxes: list[np.ndarray] = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or mol.GetNumAtoms() < 3 or mol.GetNumAtoms() > 24:
            continue
        idx = np.where(meta["frag_smiles"].astype(str).to_numpy() == str(smi))[0]
        if len(idx) < 6:
            continue
        xy = np.median(coords[idx], axis=0)
        if any(np.linalg.norm(xy - prev) < 2.2 for prev in used_xy):
            continue
        box_xy = xy.copy()
        box_xy[0] = np.clip(box_xy[0], xmin + 0.12 * xrange, xmax - 0.12 * xrange)
        box_xy[1] = np.clip(box_xy[1], ymin + 0.16 * yrange, ymax - 0.23 * yrange)
        if xy[1] > ymin + 0.74 * yrange:
            box_xy[1] -= 0.14 * yrange
        if any(np.linalg.norm(box_xy - prev) < 2.6 for prev in used_boxes):
            continue
        used_xy.append(xy)
        used_boxes.append(box_xy)
        img = Draw.MolToImage(mol, size=(150, 98), kekulize=True)
        arrowprops = None
        if np.linalg.norm(box_xy - xy) > 0.35:
            arrowprops = dict(arrowstyle="-", lw=0.35, color="#5a5a5a", alpha=0.55)
        ax.add_artist(AnnotationBbox(
            OffsetImage(img, zoom=0.16),
            xy,
            xybox=box_xy,
            xycoords="data",
            boxcoords="data",
            frameon=False,
            arrowprops=arrowprops,
            zorder=6,
        ))
        if len(used_xy) >= max_mols:
            break


def main() -> None:
    root = project_root()
    default_data = root / "main_figure/fig2/data/task5_peak_embedding_umap/large"
    default_out = root / "main_figure/fig2/output/task5_peak_embedding_umap"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--output-dir", type=Path, default=default_out)
    ap.add_argument("--max-points", type=int, default=24000)
    ap.add_argument("--n-neighbors", type=int, default=28)
    ap.add_argument("--min-dist", type=float, default=0.38)
    ap.add_argument("--spread", type=float, default=1.45)
    ap.add_argument("--n-epochs", type=int, default=420)
    ap.add_argument("--color-by", choices=sorted(COLOR_SPECS), default="frag_label")
    ap.add_argument("--output-stem", default=None)
    ap.add_argument("--show-insets", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    meta_all = pd.read_csv(args.data_dir / "peak_embedding_metadata.csv")
    emb_all = np.load(args.data_dir / "peak_embeddings_ultra.npy")
    keep_labels = [lab for lab, n in meta_all["frag_label"].value_counts().items() if n >= 50]
    keep = meta_all["frag_label"].isin(keep_labels).to_numpy()
    meta_all = meta_all[keep].reset_index(drop=True)
    emb_all = emb_all[keep]

    idx = balanced_sample(meta_all, args.max_points, args.seed)
    meta = meta_all.iloc[idx].reset_index(drop=True)
    emb = emb_all[idx]
    meta = parent_properties(meta)
    meta["peak_mz"] = meta["mz"].astype(float)
    meta = add_precursor_context(meta, args.data_dir)
    meta = add_discrete_classes(meta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    param_tag = (
        f"n{args.n_neighbors}_d{str(args.min_dist).replace('.', 'p')}"
        f"_s{str(args.spread).replace('.', 'p')}_e{args.n_epochs}_seed{args.seed}"
    )
    coord_path = args.output_dir / f"ultrams_peak_umap_coords_{param_tag}.csv"
    if coord_path.exists():
        coords_df = pd.read_csv(coord_path)
        if len(coords_df) == len(meta) and "umap1" in coords_df.columns:
            xy = coords_df[["umap1", "umap2"]].to_numpy()
        else:
            xy = fit_umap(emb, args.seed, args.n_neighbors, args.min_dist, args.spread, args.n_epochs)
    else:
        xy = fit_umap(emb, args.seed, args.n_neighbors, args.min_dist, args.spread, args.n_epochs)
    coords = meta.copy()
    coords["umap1"] = xy[:, 0]
    coords["umap2"] = xy[:, 1]
    coords.to_csv(coord_path, index=False)
    coords.to_csv(args.output_dir / "ultrams_peak_umap_coords.csv", index=False)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = "fig2_ultrams_peak_embedding_umap" if args.color_by == "frag_label" else f"fig2_ultrams_peak_embedding_umap_{args.color_by}"
    summary = {
        "n_peaks_plotted": int(len(meta)),
        "n_spectra": int(meta["spec_id"].nunique()),
        "color_by": args.color_by,
        "output_stem": output_stem,
        "fragment_label_counts": meta["frag_label"].value_counts().to_dict(),
        "umap": {
            "n_neighbors": int(args.n_neighbors),
            "min_dist": float(args.min_dist),
            "spread": float(args.spread),
            "n_epochs": int(args.n_epochs),
            "metric": "cosine",
            "coord_cache": str(coord_path),
        },
    }
    with open(args.output_dir / "ultrams_peak_umap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_dir / f"{output_stem}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(4.35, 3.72), dpi=350)
    color_kind, color_label, cmap = COLOR_SPECS[args.color_by]
    point_size = float(np.clip(1.32 * np.sqrt(24000 / max(len(meta), 1)), 0.38, 1.32))
    point_alpha = float(np.clip(0.48 * np.sqrt(24000 / max(len(meta), 1)), 0.24, 0.52))
    if color_kind == "categorical":
        labels = [lab for lab in PALETTE if lab in set(meta[args.color_by])]
        extra_labels = sorted(set(meta[args.color_by]) - set(labels))
        labels.extend(extra_labels)
        fallback = plt.get_cmap("tab10")
        for k, lab in enumerate(labels):
            ii = np.where(meta[args.color_by].to_numpy() == lab)[0]
            color = PALETTE.get(lab, fallback(k % 10))
            ax.scatter(
                xy[ii, 0], xy[ii, 1],
                s=point_size, c=[color], alpha=point_alpha, linewidths=0,
                rasterized=True, label=lab,
            )
    else:
        values = meta[args.color_by].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            raise RuntimeError(f"No finite values available for --color-by {args.color_by}")
        vmin, vmax = np.nanpercentile(values[valid], [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(values[valid])), float(np.nanmax(values[valid]))
        sc = ax.scatter(
            xy[valid, 0], xy[valid, 1],
            s=point_size, c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=max(point_alpha, 0.30), linewidths=0, rasterized=True,
        )
        if np.any(~valid):
            ax.scatter(
                xy[~valid, 0], xy[~valid, 1],
                s=1.1, c="#c9cdd2", alpha=0.28, linewidths=0, rasterized=True,
            )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.012)
        cbar.set_label(color_label, fontsize=6.2)
        cbar.ax.tick_params(labelsize=5.8, length=2)

    xpad = (xy[:, 0].max() - xy[:, 0].min()) * 0.08
    ypad = (xy[:, 1].max() - xy[:, 1].min()) * 0.10
    ax.set_xlim(xy[:, 0].min() - xpad, xy[:, 0].max() + xpad)
    ax.set_ylim(xy[:, 1].min() - ypad, xy[:, 1].max() + ypad)
    if args.show_insets:
        draw_fragment_insets(ax, xy, meta, max_mols=6)

    ax.set_title(f"UltraMS peak-token embedding: {color_label}", fontsize=8.2, fontweight="bold", pad=3)
    ax.text(0.02, 0.98, f"{len(meta):,} MAGMa-annotated peaks",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(0.18, 0.08), xytext=(0.04, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color="black"))
    ax.annotate("", xy=(0.04, 0.22), xytext=(0.04, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color="black"))
    ax.text(0.19, 0.055, "UMAP 1", transform=ax.transAxes, fontsize=6.6, va="top")
    ax.text(0.018, 0.23, "UMAP 2", transform=ax.transAxes, fontsize=6.6, rotation=90, va="bottom")
    if color_kind == "categorical":
        ncol = 4 if len(labels) > 6 else 3
        y_anchor = -0.24 if len(labels) > 6 else -0.17
        legend_font = 5.4 if len(labels) > 6 else 6.1
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, y_anchor), ncol=ncol, frameon=False,
            fontsize=legend_font, handletextpad=0.22, columnspacing=0.70, markerscale=3.0,
        )
    fig.savefig(args.output_dir / f"{output_stem}.png", bbox_inches="tight", dpi=350)
    fig.savefig(args.output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir / f'{output_stem}.png'}")


if __name__ == "__main__":
    main()
