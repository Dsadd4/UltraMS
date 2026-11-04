"""
Task 4 — Ion Polarity Classification
4 figures: metric bar, horizontal advantage, gauge-style arc, confidence strip
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FIG2_DIR = os.path.join(PROJECT_ROOT, 'main_figure', 'fig2')
DATA_PATH = os.path.join(FIG2_DIR, 'data', 'ion_polarity',
                         'polarity_linear_probe_combined.json')
OUT = os.path.join(FIG2_DIR, 'output', 'task4_ion_polarity')
os.makedirs(OUT, exist_ok=True)

C_ULTRA = "#E64B35"
C_DREAM = "#4DBBD5"
C_GRAY  = "#AAAAAA"
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
plt.rcParams.update(NATURE)

# ── Data ──────────────────────────────────────────────────────────────────────
POLARITY_ICONS = {
    "negative": "\u2296",
    "positive": "\u2295",
}
ULTRA_MODEL_KEY = "rt_only_d"
DREAM_MODEL_KEY = "dreams"


def load_ion_polarity_data(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    ultra = results[ULTRA_MODEL_KEY]
    dreams = results[DREAM_MODEL_KEY]

    metric_specs = [
        ("Val Acc", "mlp_val", "accuracy"),
        ("Test Acc", "mlp_test", "accuracy"),
        ("Test F1", "mlp_test", "f1_macro"),
    ]
    metric_labels = [label for label, _, _ in metric_specs]
    ultra_vals = [float(ultra[split][metric]) for _, split, metric in metric_specs]
    dream_vals = [float(dreams[split][metric]) for _, split, metric in metric_specs]

    gauge_specs = [
        ("Negative Mode", "f1_neg", POLARITY_ICONS["negative"]),
        ("Positive Mode", "f1_pos", POLARITY_ICONS["positive"]),
    ]
    gauge_panels = [
        (
            label,
            float(ultra["mlp_test"][metric]),
            float(dreams["mlp_test"][metric]),
            icon,
        )
        for label, metric, icon in gauge_specs
    ]

    return metric_labels, ultra_vals, dream_vals, gauge_panels


METRICS, ULTRA_VALS, DREAM_VALS, GAUGE_PANELS = load_ion_polarity_data()
DELTA = [u - d for u, d in zip(ULTRA_VALS, DREAM_VALS)]

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  → {name}")


def fig_swatch_legend_pair(fig, cx, y, hs, ws, gap, mid, w_u, w_d, fs, lw, pair, left_align=False):
    """Two [■ name] groups in figure coords. If left_align, cx is x0 of first swatch; else centered on cx."""
    seg_u = ws + gap + w_u
    seg_d = ws + gap + w_d
    if left_align:
        x_u = cx
        x_d = x_u + seg_u + mid
    else:
        total = seg_u + mid + seg_d
        x_u = cx - total / 2
        x_d = x_u + seg_u + mid
    for x0, (c, name) in zip((x_u, x_d), pair):
        fig.add_artist(
            Rectangle(
                (x0, y - hs / 2),
                ws,
                hs,
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="white",
                linewidth=lw,
                clip_on=False,
                zorder=10,
            )
        )
        fig.text(
            x0 + ws + gap,
            y,
            name,
            ha="left",
            va="center",
            fontsize=fs,
            color=c,
            fontweight="bold",
            transform=fig.transFigure,
            zorder=10,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar: all 3 metrics
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.8, 3.4))
x  = np.arange(len(METRICS))
w  = 0.32
bu = ax.bar(x - w/2, ULTRA_VALS, w, color=C_ULTRA, label="UltraMs",
            edgecolor="white", linewidth=0.4, zorder=3)
bd = ax.bar(x + w/2, DREAM_VALS, w, color=C_DREAM, label="DreaMS",
            edgecolor="white", linewidth=0.4, zorder=3)

for i, (u, d, delta) in enumerate(zip(ULTRA_VALS, DREAM_VALS, DELTA)):
    ax.text(x[i], max(u, d) + 0.003, f"+{delta*100:.1f}pp",
            ha="center", fontsize=7.5, fontweight="bold", color=C_GREEN)
    ax.text(x[i] - w/2, u - 0.005, f"{u:.1%}", ha="center", va="top",
            fontsize=6.5, color="white", fontweight="bold")
    ax.text(x[i] + w/2, d - 0.005, f"{d:.1%}", ha="center", va="top",
            fontsize=6.5, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=8.5)
ax.set_ylabel("Score", fontsize=8.5)
ax.set_ylim(0.84, 0.975)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Ion Polarity Classification — Val / Test Metrics", fontsize=9, pad=5)
ax.legend(frameon=False, ncol=2, loc="lower right")
ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
save(fig, "fig4a_metrics_bar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Horizontal bar: absolute improvement (pp)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 2.6))
y = np.arange(len(METRICS))
bars = ax.barh(y, [d * 100 for d in DELTA], color=C_GREEN, height=0.45,
               edgecolor="white", linewidth=0.4, zorder=3)
for yi, d in enumerate(DELTA):
    ax.text(d * 100 + 0.08, yi, f"+{d*100:.1f} pp", va="center",
            fontsize=8.5, fontweight="bold", color=C_GREEN)

ax.set_yticks(y)
ax.set_yticklabels(METRICS, fontsize=8.5)
ax.set_xlabel("Absolute improvement over DreaMS (percentage points)", fontsize=8)
ax.set_xlim(0, 9)
ax.set_title("Ion Polarity: UltraMs Improvement over DreaMS", fontsize=9, pad=5)
ax.axvline(0, color="gray", lw=0.7)
ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
save(fig, "fig4b_improvement_hbar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Gauge / Arc chart: 2 panels (Negative Mode, Positive Mode)
# ══════════════════════════════════════════════════════════════════════════════
# Per-class F1 loaded from polarity_linear_probe_combined.json (mlp_test).

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0),
                         subplot_kw={"projection": "polar"})
fig.subplots_adjust(wspace=0.05, top=0.78, bottom=0.14)

for ax, (title, ultra_val, dream_val, icon) in zip(axes, GAUGE_PANELS):
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    delta = ultra_val - dream_val

    # background full arc
    th_bg = np.linspace(0, np.pi, 300)
    ax.plot(th_bg, np.ones(300), lw=24, color="#EEEEEE", solid_capstyle="round")

    # DreaMS arc (wider, behind)
    th_d = np.linspace(0, dream_val * np.pi, 300)
    ax.plot(th_d, np.ones_like(th_d), lw=16, color=C_DREAM,
            solid_capstyle="round", alpha=0.90)

    # UltraMs arc (thinner, on top)
    th_u = np.linspace(0, ultra_val * np.pi, 300)
    ax.plot(th_u, np.ones_like(th_u), lw=8, color=C_ULTRA,
            solid_capstyle="round")

    ax.set_ylim(0, 1.65)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # ── Icon + +pp in center ──────────────────────────────────────────────
    ax.text(np.pi / 2, 0.68, icon, ha="center", va="center",
            fontsize=22, color="#444444")
    ax.text(np.pi / 2, 0.30, f"+{delta * 100:.1f}pp", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_GREEN)

    # ── Per-model values outside the arc ─────────────────────────────────
    theta_u = ultra_val * np.pi
    ax.annotate(f"{ultra_val:.1%}",
                xy=(theta_u, 1.0), xytext=(theta_u, 1.44),
                xycoords="data", textcoords="data",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_ULTRA,
                arrowprops=dict(arrowstyle="-", color=C_ULTRA,
                                lw=0.8, shrinkA=2, shrinkB=2))

    theta_d = dream_val * np.pi
    ax.annotate(f"{dream_val:.1%}",
                xy=(theta_d, 1.0), xytext=(theta_d, 1.50),
                xycoords="data", textcoords="data",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_DREAM,
                arrowprops=dict(arrowstyle="-", color=C_DREAM,
                                lw=0.8, shrinkA=2, shrinkB=2))

# Legend: top-left of figure, near upper-left of drawing area (left + slightly lower)
pos0, pos1 = axes[0].get_position(), axes[1].get_position()
_x_left = pos0.x0 + 0.01
# well below subplot top so the shift is obvious (figure coords: smaller y = lower)
_y = max(pos0.y1, pos1.y1) - 0.20
fig_swatch_legend_pair(
    fig,
    _x_left,
    _y,
    hs=0.022,
    ws=0.014,
    gap=0.007,
    mid=0.055,
    w_u=0.048,
    w_d=0.046,
    fs=9,
    lw=0.45,
    pair=((C_ULTRA, "UltraMs"), (C_DREAM, "DreaMS")),
    left_align=True,
)
save(fig, "fig4c_accuracy_gauge")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Error-bar style dot plot: all metrics both models
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 3.2))
fig.subplots_adjust(bottom=0.18)
y = np.arange(len(METRICS))
offset = 0.14

# UltraMs dots
ax.scatter(ULTRA_VALS, y + offset, s=110, color=C_ULTRA, zorder=4,
           edgecolors="white", linewidths=0.6)
# DreaMS dots
ax.scatter(DREAM_VALS, y - offset, s=110, color=C_DREAM, zorder=4,
           edgecolors="white", linewidths=0.6)

for i, (u, d) in enumerate(zip(ULTRA_VALS, DREAM_VALS)):
    ax.plot([d, u], [i - offset, i + offset], lw=1.0, color="gray",
            alpha=0.6, zorder=2)
    ax.text(u + 0.003, i + offset, f"{u:.1%}", va="center",
            fontsize=7, color=C_ULTRA)
    ax.text(d - 0.003, i - offset, f"{d:.1%}", va="center",
            ha="right", fontsize=7, color=C_DREAM)

ax.set_yticks(y)
ax.set_yticklabels(METRICS, fontsize=8.5)
ax.set_xlabel("Score", fontsize=8.5)
ax.set_xlim(0.86, 0.970)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Ion Polarity Classification — Dot Plot", fontsize=9, pad=5)
pos = ax.get_position()
_cx4 = (pos.x0 + pos.x1) / 2
_y4 = pos.y0 - 0.028
fig_swatch_legend_pair(
    fig,
    _cx4,
    _y4,
    hs=0.018,
    ws=0.012,
    gap=0.006,
    mid=0.048,
    w_u=0.042,
    w_d=0.040,
    fs=8.5,
    lw=0.4,
    pair=((C_ULTRA, "UltraMs"), (C_DREAM, "DreaMS")),
)
ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
save(fig, "fig4d_dot_plot")

print("Task 4 done.")
