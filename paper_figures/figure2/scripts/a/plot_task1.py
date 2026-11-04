"""
Task 1 — m/z Peak Reconstruction Accuracy
4 figures: grouped bar, log-scale mae, lollipop fold-change, dot-range scatter
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FIG2_DIR = os.path.join(PROJECT_ROOT, 'main_figure', 'fig2')
OUT = os.path.join(FIG2_DIR, 'output', 'task1_mz_reconstruction')
os.makedirs(OUT, exist_ok=True)

C_ULTRA = "#E64B35"
C_DREAM = "#4DBBD5"
C_GREEN = "#00A087"
C_GRAY  = "#AAAAAA"

NATURE = {
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
}
plt.rcParams.update(NATURE)

# ── Data ──────────────────────────────────────────────────────────────────────
DATASETS   = ["GeMS", "MassSpecGym"]
ACC_ULTRA  = [0.552, 0.377]
ACC_DREAM  = [0.238, 0.035]
MAE_ULTRA  = [2.18, 8.7]
MAE_DREAM  = [65.4, 100.4]
FOLD_ACC   = [u / d for u, d in zip(ACC_ULTRA, ACC_DREAM)]
FOLD_MAE   = [d / u for d, u in zip(MAE_DREAM, MAE_ULTRA)]   # lower is better

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  → {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar: mz_acc_01da on both datasets
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 3.2))
x = np.arange(len(DATASETS))
w = 0.32
ax.bar(x - w / 2, ACC_ULTRA, w, color=C_ULTRA, label="UltraMSMS",
       edgecolor="white", linewidth=0.4, zorder=3)
ax.bar(x + w / 2, ACC_DREAM, w, color=C_DREAM, label="DreaMS",
       edgecolor="white", linewidth=0.4, zorder=3)

for i, (u, d) in enumerate(zip(ACC_ULTRA, ACC_DREAM)):
    ax.text(x[i] - w / 2, u + 0.006, f"{u:.1%}", ha="center", va="bottom",
            fontsize=6.5, color="#444444")
    ax.text(x[i] + w / 2, d + 0.006, f"{d:.1%}", ha="center", va="bottom",
            fontsize=6.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(DATASETS)
ax.set_ylim(0, 0.60)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("m/z reconstruction accuracy (absolute error ≤0.05 Da)", fontsize=9, pad=6)
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)
save(fig, "fig1a_accuracy_bar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — MAE comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 3.2))
x = np.arange(len(DATASETS))
w = 0.32
ax.bar(x - w / 2, MAE_ULTRA, w, color=C_ULTRA, label="UltraMSMS",
       edgecolor="white", linewidth=0.4, zorder=3)
ax.bar(x + w / 2, MAE_DREAM, w, color=C_DREAM, label="DreaMS",
       edgecolor="white", linewidth=0.4, zorder=3)

for i, (u, d) in enumerate(zip(MAE_ULTRA, MAE_DREAM)):
    ax.text(x[i] - w / 2, u + 1.2, f"{u:.1f}", ha="center", va="bottom",
            fontsize=6.5, color="#444444")
    ax.text(x[i] + w / 2, d + 1.2, f"{d:.1f}", ha="center", va="bottom",
            fontsize=6.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(DATASETS)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Mean Absolute Error of Reconstructed m/z", fontsize=9, pad=6)
ax.legend(frameon=False, ncol=2)
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)
save(fig, "fig1b_mae_bar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Log-scale MAE with log y-axis (dramatic difference)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 3.2))
x  = np.arange(len(DATASETS)) * 0.65
w  = 0.22
ax.bar(x - w/2, MAE_ULTRA, w, color=C_ULTRA, label="UltraMSMS",
       edgecolor="white", linewidth=0.4, zorder=3)
ax.bar(x + w/2, MAE_DREAM, w, color=C_DREAM, label="DreaMS",
       edgecolor="white", linewidth=0.4, zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.5, 105)

for i, (u, d) in enumerate(zip(MAE_ULTRA, MAE_DREAM)):
    ax.text(x[i] - w/2, u * 1.4, f"{u:.1f} Da", ha="center", fontsize=6.5,
            color="#444444")
    ax.text(x[i] + w/2, d * 1.4, f"{d:.1f} Da", ha="center", fontsize=6.5,
            color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(DATASETS)
ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
ax.set_title("m/z Mean Absolute Error (Log Scale)", fontsize=9, pad=6)
ax.legend(frameon=False, ncol=2)
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", which="major", zorder=0)
save(fig, "fig1c_mae_logscale")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Lollipop / dot-range: Fold improvement both metrics both datasets
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.0, 3.0))

labels  = ["GeMS\n(Accuracy)", "MassSpecGym\n(Accuracy)", "GeMS\n(MAE)", "MassSpecGym\n(MAE)"]
folds   = [FOLD_ACC[0], FOLD_ACC[1], FOLD_MAE[0], FOLD_MAE[1]]
colors_ = [C_ULTRA, C_ULTRA, C_GREEN, C_GREEN]
marker  = ["o", "o", "s", "s"]

y = np.arange(len(labels))
ax.barh(y, folds, color=colors_, alpha=0.20, height=0.5)
for yi, (f, c, m) in enumerate(zip(folds, colors_, marker)):
    ax.plot(f, yi, m, color=c, markersize=10, zorder=4)
    ax.text(f + 0.2, yi, f"{f:.1f}", va="center", fontsize=8,
            color=c, fontweight="bold")

ax.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.7)
ax.text(1.05, -0.6, "DreaMS\nbaseline", fontsize=6.5, color="gray", va="top")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("Fold Improvement over DreaMS", fontsize=8.5)
ax.set_xlim(0, max(folds) * 1.2)
ax.set_title("UltraMSMS Performance Improvement over DreaMS", fontsize=9, pad=6)

patches = [mpatches.Patch(color=C_ULTRA, alpha=0.7, label="Reconstruction Accuracy"),
           mpatches.Patch(color=C_GREEN, alpha=0.7, label="MAE Improvement")]
ax.legend(handles=patches, frameon=False, fontsize=7, loc="lower right")
ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
save(fig, "fig1d_fold_lollipop")

print("Task 1 done.")
