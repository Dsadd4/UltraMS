"""
Task 2 — Neutral Loss Detection
5 figures: grouped bar, scatter diagonal, horizontal fold bar, radar, heatmap-style strip
Data: auto-loaded from latest nl_ultra_* / nl_dreams_* run directories
"""
import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FIG2_DIR = os.path.join(PROJECT_ROOT, 'main_figure', 'fig2')
OUT = os.path.join(FIG2_DIR, 'output', 'task2_neutral_loss')
os.makedirs(OUT, exist_ok=True)
RUNS_BASE = os.path.join(PROJECT_ROOT, 'train', 'output', 'showcase')

C_ULTRA = "#E64B35"
C_DREAM = "#4DBBD5"
C_GOLD  = "#F39B7F"
C_GREEN = "#00A087"

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
sns.set_theme(style="ticks", rc=NATURE)
PALETTE = {"UltraMSMS": C_ULTRA, "DreaMS": C_DREAM}

NL_KEYS  = ["H2O",  "NH3",  "CO2",  "HF",  "HCl",  "CO",  "CH3"]
NL_SHORT = ["H₂O",  "NH₃",  "CO₂",  "HF",  "HCl",  "CO",  "CH₃"]
NL_FULL  = ["H₂O\n(−18 Da)", "NH₃\n(−17 Da)", "CO₂\n(−44 Da)",
             "HF\n(−20 Da)",  "HCl\n(−36 Da)", "CO\n(−28 Da)",  "CH₃\n(−15 Da)"]
NL_MASS  = [18, 17, 44, 20, 36, 28, 15]


def latest_run(pattern):
    dirs = sorted(glob.glob(os.path.join(RUNS_BASE, pattern)))
    if not dirs:
        raise FileNotFoundError(f"No runs matching {pattern} in {RUNS_BASE}")
    return dirs[-1]


def load_metrics(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    m = cfg['test_metrics']
    ap  = [m[f'ap_{k}']  for k in NL_KEYS]
    auc = [m[f'auc_{k}'] for k in NL_KEYS]
    return ap, auc, m['mean_ap'], m['mean_auc']


FALLBACK_METRICS = {
    'ultra': {
        'ap':  [0.658, 0.322, 0.141, 0.125, 0.134, 0.264, 0.314],
        'auc': [0.963, 0.958, 0.880, 0.900, 0.900, 0.933, 0.980],
        'mean_ap': 0.280,
        'mean_auc': 0.913,
    },
    'dreams': {
        'ap':  [0.340, 0.166, 0.079, 0.042, 0.086, 0.221, 0.161],
        'auc': [0.831, 0.852, 0.807, 0.800, 0.800, 0.757, 0.879],
        'mean_ap': 0.156,
        'mean_auc': 0.834,
    },
}


def load_or_fallback(pattern, key):
    try:
        run_dir = latest_run(pattern)
        print(f"{key} run: {run_dir}")
        return load_metrics(run_dir)
    except FileNotFoundError as exc:
        m = FALLBACK_METRICS[key]
        print(f"[fallback] {exc}; using recorded Figure 2 metrics for {key}")
        return m['ap'], m['auc'], m['mean_ap'], m['mean_auc']


ULTRA_AP, ULTRA_AUC_LIST, MEAN_ULTRA, ULTRA_AUC = load_or_fallback('nl_ultra_*', 'ultra')
DREAM_AP, DREAM_AUC_LIST, MEAN_DREAM, DREAM_AUC = load_or_fallback('nl_dreams_*', 'dreams')
RATIO_AP = [u / max(d, 1e-9) for u, d in zip(ULTRA_AP, DREAM_AP)]

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  → {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar: per-type AP
# ══════════════════════════════════════════════════════════════════════════════
df_ap = pd.DataFrame({
    "Type":  NL_SHORT * 2,
    "Model": ["UltraMSMS"] * len(NL_SHORT) + ["DreaMS"] * len(NL_SHORT),
    "AP":    ULTRA_AP + DREAM_AP,
})
fig, ax = plt.subplots(figsize=(6.5, 3.4))
sns.barplot(data=df_ap, x="Type", y="AP", hue="Model",
            palette=PALETTE, width=0.65, gap=0.08,
            alpha=0.88, edgecolor="white", linewidth=0.3,
            ax=ax, zorder=3)

ax.set_ylim(0, 0.80)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Neutral Loss Detection — Per-Type Average Precision", fontsize=9, pad=5)
ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)
sns.despine(ax=ax)
save(fig, "fig2a_ap_grouped_bar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Scatter: DreaMS AP vs UltraMs AP with diagonal reference (wider)
# ══════════════════════════════════════════════════════════════════════════════
MARKERS = ["o", "s", "^", "P", "X"]
SKIP_NL = {"HF", "HCl"}   # excluded from scatter (too rare / low AP)

_scatter_data = [
    (short, u, d, r, mk)
    for short, u, d, r, mk in zip(NL_SHORT, ULTRA_AP, DREAM_AP, RATIO_AP,
                                   ["o", "s", "^", ">", "v", "P", "X"])
    if short not in {"HF", "HCl"}
]
_u_plot = [x[1] for x in _scatter_data]
_d_plot = [x[2] for x in _scatter_data]
_r_plot = [x[3] for x in _scatter_data]

_ratio_max = max(_r_plot)
_x_max = max(_d_plot) * 1.18
_y_max = max(_u_plot) * 1.15
_lim   = max(_x_max, _y_max)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
cmap  = plt.cm.RdYlGn
norm  = Normalize(vmin=1.0, vmax=max(5.5, _ratio_max * 0.9))
handles = []
for short, u, d, r, mk in _scatter_data:
    c = cmap(norm(r))
    sc = ax.scatter(d, u, s=160, color=c, marker=mk, edgecolors="white",
                    linewidths=0.6, zorder=4, label=f"{short}  ({r:.1f}×)")
    handles.append(sc)
    ax.annotate(short, (d, u), xytext=(4, 4), textcoords='offset points',
                fontsize=6.5, color='#333333', zorder=5)

# diagonal y = x
ax.plot([0, _lim], [0, _lim], "--", color="gray", lw=0.8, alpha=0.6, zorder=1)
ax.fill_between([0, _lim], [0, _lim], _lim, alpha=0.04, color=C_ULTRA)
ax.text(_lim * 0.65, _lim * 0.72, "UltraMSMS\nbetter", fontsize=6.5,
        color=C_ULTRA, alpha=0.6, ha='center')

# colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02)
cb.set_label("Fold Improvement (Ultra / DreaMS)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

ax.set_xlabel("DreaMS Average Precision", fontsize=8.5)
ax.set_ylabel("UltraMSMS Average Precision", fontsize=8.5)
ax.set_xlim(-0.01, max(_d_plot) * 1.18)
ax.set_ylim(-0.01, max(_u_plot) * 1.15)
ax.set_title("Neutral Loss AP: UltraMSMS vs DreaMS", fontsize=9, pad=5)
ax.grid(lw=0.4, alpha=0.3, ls=":", zorder=0)
save(fig, "fig2b_ap_scatter_diagonal")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Horizontal lollipop: fold improvement sorted descending
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.0, 3.6))
order = np.argsort(RATIO_AP)[::-1]
labels_sorted = [NL_SHORT[i] for i in order]
ratios_sorted = [RATIO_AP[i]  for i in order]
y = np.arange(len(labels_sorted))

cmap2 = plt.cm.YlOrRd
_r_span = max(ratios_sorted) - 1.0 if max(ratios_sorted) > 1.0 else 1.0
colors_sorted = [cmap2(0.3 + 0.6 * (r - 1.0) / _r_span) for r in ratios_sorted]

ax.barh(y, ratios_sorted, color=colors_sorted, alpha=0.35, height=0.5, zorder=2)
for yi, (r, c) in enumerate(zip(ratios_sorted, colors_sorted)):
    ax.plot(r, yi, "o", color=c, markersize=11, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.text(r + 0.04, yi, f"{r:.2f}×", va="center", fontsize=8,
            color=c, fontweight="bold")

ax.axvline(1.0, color="gray", lw=0.9, ls="--", alpha=0.7)
ax.text(1.03, -0.7, "DreaMS\nbaseline", fontsize=6.5, color="gray")
ax.set_yticks(y)
ax.set_yticklabels(labels_sorted, fontsize=8.5)
ax.set_xlabel("Fold improvement (UltraMs / DreaMS)", fontsize=8.5)
ax.set_xlim(0.8, max(ratios_sorted) * 1.18)
ax.set_title("Neutral Loss: Fold Improvement per Type\n(sorted by advantage)", fontsize=9, pad=5)
ax.grid(axis="x", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)
save(fig, "fig2c_fold_lollipop_sorted")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Radar: per-type AP polygon (UltraMs vs DreaMS)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(4.2, 3.8))
ax  = fig.add_subplot(111, projection="polar")
N   = len(NL_SHORT)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
u_vals = ULTRA_AP + ULTRA_AP[:1]
d_vals = DREAM_AP + DREAM_AP[:1]

ax.plot(angles, u_vals, "o-", color=C_ULTRA, lw=1.4, markersize=4, label="UltraMs")
ax.fill(angles, u_vals, color=C_ULTRA, alpha=0.20)
ax.plot(angles, d_vals, "s-", color=C_DREAM, lw=1.4, markersize=4, label="DreaMS")
ax.fill(angles, d_vals, color=C_DREAM, alpha=0.20)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(NL_SHORT, fontsize=8)
_radar_max = max(max(ULTRA_AP), max(DREAM_AP)) * 1.2
ax.set_ylim(0, _radar_max)
ax.set_yticks([round(_radar_max * t, 1) for t in [0.33, 0.67, 1.0]])
ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=6, color="gray")
ax.spines["polar"].set_linewidth(0.5)
ax.set_title("Neutral Loss Detection — AP Radar", fontsize=9, pad=14)
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=7.5)
save(fig, "fig2d_ap_radar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Summary strip: AP + AUC side-by-side with mean labels
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(4.5, 3.0), sharey=False)

models = ["UltraMSMS", "DreaMS"]
x = np.arange(2)
bw = 0.45
bar_colors = [C_ULTRA, C_DREAM]

# left: mean AP
ax = axes[0]
ax.bar(x, [MEAN_ULTRA, MEAN_DREAM], width=bw, color=bar_colors,
       edgecolor="white", linewidth=0.4, zorder=3)
for xi, v in zip(x, (MEAN_ULTRA, MEAN_DREAM)):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
            fontsize=7.5, color="#444444")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.36)
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Mean Average Precision", fontsize=8.5)
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)

# right: mean AUC
ax = axes[1]
ax.bar(x, [ULTRA_AUC, DREAM_AUC], width=bw, color=bar_colors,
       edgecolor="white", linewidth=0.4, zorder=3)
for xi, v in zip(x, (ULTRA_AUC, DREAM_AUC)):
    ax.text(xi, v + 0.001, f"{v:.3f}", ha="center", va="bottom",
            fontsize=7.5, color="#444444")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.78, 0.96)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Mean AUC-ROC", fontsize=8.5)
ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
save(fig, "fig2e_mean_ap_auc")

print("Task 2 done.")
