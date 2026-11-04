"""
Atom-Query Probe — grouped bar chart (R² per element, Ultra vs DreaMS).
Data is read from the latest config.json under train/output/showcase/.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FIG2_DIR = os.path.join(PROJECT_ROOT, 'main_figure', 'fig2')
OUT = os.path.join(FIG2_DIR, 'output', 'task3_heteroatom')
os.makedirs(OUT, exist_ok=True)
CKPT_DIR = os.path.join(PROJECT_ROOT, 'train', 'output', 'showcase')

# ── colours ───────────────────────────────────────────────────────────────
C_ULTRA   = "#E64B35"
C_DREAM   = "#4DBBD5"
EL_COLORS = {"S": "#3C5488", "Cl": "#00A087", "F": "#F39B7F", "Br": "#8491B4"}

NATURE = {
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8,  "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3,   "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
}
plt.rcParams.update(NATURE)


def load_last(backbone: str, atoms: list[str]) -> dict:
    """
    Find the latest config.json for (backbone, atoms) and return
    the final epoch's history entry.
    """
    pattern = os.path.join(CKPT_DIR,
                           f"atomq_{backbone}_{'_'.join(atoms)}_*",
                           "config.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        fallback = {
            'ultra':  {'ep': -1, 'r2_S': 0.285, 'r2_Cl': 0.362, 'r2_F': 0.202, 'r2_Br': 0.202, 'mean_r2': 0.263},
            'dreams': {'ep': -1, 'r2_S': 0.177, 'r2_Cl': 0.196, 'r2_F': 0.098, 'r2_Br': 0.116, 'mean_r2': 0.147},
        }
        if backbone in fallback:
            print(f"  [fallback] no AtomQ config found for {backbone}; using recorded Figure 2 values")
            return fallback[backbone], atoms
        raise FileNotFoundError(f"No config found for {backbone} {atoms}.\n"
                                f"Searched: {pattern}")
    cfg_path = matches[-1]
    print(f"  [{backbone}] reading {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    last_ep = cfg["history"][-1]    # final epoch
    return last_ep, cfg["atom_names"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  → {name}")


# ── load data ─────────────────────────────────────────────────────────────
ATOMS = ["S", "Cl", "F", "Br"]
ultra_last, atom_names = load_last("ultra",  ATOMS)
dream_last, _          = load_last("dreams", ATOMS)

ULTRA_R2 = [ultra_last[f"r2_{a}"] for a in atom_names]
DREAM_R2 = [dream_last[f"r2_{a}"] for a in atom_names]
MEAN_U   = ultra_last["mean_r2"]
MEAN_D   = dream_last["mean_r2"]

print(f"  Ultra  ep{ultra_last['ep']}  mean_r2={MEAN_U:.4f}  "
      + "  ".join(f"{a}={v:.3f}" for a, v in zip(atom_names, ULTRA_R2)))
print(f"  DreaMS ep{dream_last['ep']}  mean_r2={MEAN_D:.4f}  "
      + "  ".join(f"{a}={v:.3f}" for a, v in zip(atom_names, DREAM_R2)))


# ── Figure: grouped bar ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))

x = np.arange(len(atom_names))
w = 0.32

ax.bar(x - w/2, ULTRA_R2, w, color=C_ULTRA, label="UltraMS",
       edgecolor="white", linewidth=0.4, zorder=3)
ax.bar(x + w/2, DREAM_R2, w, color=C_DREAM, label="DreaMS",
       edgecolor="white", linewidth=0.4, zorder=3)

# value labels
for i, (u, d) in enumerate(zip(ULTRA_R2, DREAM_R2)):
    ax.text(x[i] - w/2, u + 0.006, f"{u:.3f}", ha="center", va="bottom",
            fontsize=6.0, fontweight="bold", color="#333333")
    ax.text(x[i] + w/2, d + 0.006, f"{d:.3f}", ha="center", va="bottom",
            fontsize=6.0, fontweight="bold", color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(atom_names, fontsize=9, fontweight="bold")
for tick, name in zip(ax.get_xticklabels(), atom_names):
    tick.set_color(EL_COLORS.get(name, "#333333"))

ax.set_ylabel("R²  (coefficient of determination)", fontsize=8.5)
ax.set_ylim(0, max(max(ULTRA_R2), max(DREAM_R2)) * 1.25)
ax.set_title("Atom-Query Probe: Halogen Atom Count Regression (R²)",
             fontsize=9, pad=6)
ax.legend(frameon=False, ncol=1, fontsize=7.5, loc="upper right")
ax.set_axisbelow(True)
ax.yaxis.grid(lw=0.4, alpha=0.35, color="gray", zorder=0)

save(fig, "fig_atomq_bar")
print("Done.")
