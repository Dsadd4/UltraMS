"""
Discriminability figures that clearly show the performance gap.

Panel A: Score margin distribution (GT_sim - Top1_sim)
  > 0  →  GT is the top-1 (correct retrieval)
  < 0  →  an imposter beats GT (wrong retrieval)
  UltraMS should have a taller spike at 0 and shorter negative tail.

Panel B: CDF of GT rank (log x-axis)
  P(rank ≤ k) vs k — exactly the Top-K curve, continuous version.
  Vertical lines mark k = 1, 5, 10, 20.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
import csv

SCRIPT_PATH = Path(__file__).resolve()
FIG3_ROOT = SCRIPT_PATH.parents[2]
ROOT = SCRIPT_PATH.parents[4]
BASE = ROOT / 'train/output/comparison'
PANEL_NAME = SCRIPT_PATH.parents[1].name
SOURCE_BASE = FIG3_ROOT / "source_results/train_output_comparison"
PANEL_DATA = SCRIPT_PATH.parents[1] / "source_data"
OUT = SCRIPT_PATH.parents[1] / "output"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = 'test'

C_ULTRA  = '#3965BB'
C_DREAMS = '#E9A0AA'
LS_ULTRA  = '-'
LS_DREAMS = '--'

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          6.5,
    'axes.labelsize':     6.5,
    'axes.titlesize':     6.5,
    'xtick.labelsize':    6.5,
    'ytick.labelsize':    6.5,
    'legend.fontsize':    6.5,
    'legend.frameon':     False,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'pdf.fonttype':       42,
    'svg.fonttype':       'none',
    'figure.dpi':         300,
})

def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"Missing raw score npz. Tried:\n{tried}")


# ── Load ────────────────────────────────────────────────────
ULTRA_NPZ = first_existing(
    BASE / f'10_raw_scores_msnlib_mass_rt_only_d11_{SPLIT}.npz',
    SOURCE_BASE / f'10_raw_scores_msnlib_mass_rt_only_d11_{SPLIT}.npz',
    PANEL_DATA / f'10_raw_scores_msnlib_mass_rt_only_d11_{SPLIT}.npz',
)
DREAMS_NPZ = first_existing(
    BASE / f'10_raw_scores_msnlib_mass_dreams_{SPLIT}.npz',
    SOURCE_BASE / f'10_raw_scores_msnlib_mass_dreams_{SPLIT}.npz',
    PANEL_DATA / f'10_raw_scores_msnlib_mass_dreams_{SPLIT}.npz',
)
ultra  = np.load(ULTRA_NPZ)
dreams = np.load(DREAMS_NPZ)

def get_arrays(d):
    gt    = d['gt_sims'].astype(np.float64)
    top1  = d['top1_sims'].astype(np.float64)
    ranks = d['ranks'].astype(np.int64)
    margin = gt - top1          # 0 if correct, <0 if imposter wins
    return gt, top1, ranks, margin

gt_u, top1_u, ranks_u, margin_u = get_arrays(ultra)
gt_d, top1_d, ranks_d, margin_d = get_arrays(dreams)

n = len(ranks_u)
print(f'UltraMS  Top-1 acc: {(ranks_u==1).mean()*100:.2f}%  (margin==0: {(margin_u==0).mean()*100:.2f}%)')
print(f'DreAMS   Top-1 acc: {(ranks_d==1).mean()*100:.2f}%  (margin==0: {(margin_d==0).mean()*100:.2f}%)')

def plot_margin_panel(ax):
    """Panel A: Score margin distribution."""
    for margin, color, ls, label in [
        (margin_u, C_ULTRA,  LS_ULTRA,  'UltraMS'),
        (margin_d, C_DREAMS, LS_DREAMS, 'DreaMS'),
    ]:
        frac_correct = (margin == 0).mean()
        wrong = margin[margin < 0]
        if len(wrong) > 10:
            kde = gaussian_kde(wrong, bw_method=0.08)
            x_neg = np.linspace(wrong.min() - 0.01, -1e-4, 300)
            y_neg = kde(x_neg) * (1 - frac_correct)
            ax.fill_between(x_neg, y_neg, alpha=0.18, color=color)
            ax.plot(x_neg, y_neg, color=color, lw=1.2, linestyle=ls, label=label)

    ax.axvline(0, color='#888888', lw=0.7, linestyle=':', zorder=3)
    ax.set_xlabel('True − top-ranked score')
    ax.set_ylabel('Scaled density')
    ax.set_xlim(-0.55, 0.0)
    ax.set_ylim(0, 4.35)
    ax.set_xticks([-0.5, -0.3, -0.1, 0.0])
    ax.legend(loc='upper left', handlelength=1.2)
    ax.yaxis.grid(True, color='#D8D8D8', linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def plot_rank_cdf_panel(ax):
    """Panel B: CDF of GT rank (log scale)."""
    k_vals = np.arange(1, 1001)

    for ranks, color, ls, label in [
        (ranks_u, C_ULTRA,  LS_ULTRA,  'UltraMS'),
        (ranks_d, C_DREAMS, LS_DREAMS, 'DreaMS'),
    ]:
        cdf = np.array([(ranks <= k).mean() * 100 for k in k_vals])
        ax.plot(k_vals, cdf, color=color, lw=1.3, linestyle=ls, label=label)

    for k in [1, 5, 20]:
        ax.axvline(k, color='#cccccc', lw=0.6, linestyle=':', zorder=0)

    ax.set_xscale('log')
    ax.set_xlabel('Candidate rank, $k$')
    ax.set_ylabel('Ground truth in top $k$ (%)')
    ax.set_xlim(1, 1000)
    ax.set_ylim(15, 85)
    ax.legend(loc='upper left', handlelength=1.2)
    ax.yaxis.grid(True, color='#D8D8D8', linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


# ── Save final panel ────────────────────────────────────────
draw_func = plot_margin_panel if PANEL_NAME == "fig3f" else plot_rank_cdf_panel
mm = 1 / 25.4
fig_single, ax_single = plt.subplots(figsize=(42 * mm, 42 * mm))
draw_func(ax_single)
fig_single.subplots_adjust(left=0.25, right=0.98, bottom=0.24, top=0.98)
for ext in ("pdf", "png", "svg"):
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
    if ext == "png":
        kwargs["dpi"] = 450
    fig_single.savefig(OUT / f"{PANEL_NAME}.{ext}", **kwargs)
plt.close(fig_single)
print(f"Saved {PANEL_NAME}.pdf/.png/.svg")

summary_csv = OUT / "fig_sim_discriminability_plot_data.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "source_npz", "n_queries", "top1_accuracy", "median_rank", "mean_margin"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "model": "UltraMS",
            "source_npz": str(ULTRA_NPZ),
            "n_queries": int(len(ranks_u)),
            "top1_accuracy": float((ranks_u == 1).mean()),
            "median_rank": float(np.median(ranks_u)),
            "mean_margin": float(np.mean(margin_u)),
        }
    )
    writer.writerow(
        {
            "model": "DreaMS",
            "source_npz": str(DREAMS_NPZ),
            "n_queries": int(len(ranks_d)),
            "top1_accuracy": float((ranks_d == 1).mean()),
            "median_rank": float(np.median(ranks_d)),
            "mean_margin": float(np.mean(margin_d)),
        }
    )
print(f"Saved {summary_csv}")

plot_data_csv = OUT / f"{PANEL_NAME}_plot_data.csv"
with plot_data_csv.open("w", newline="", encoding="utf-8") as f:
    if PANEL_NAME == "fig3f":
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "score_margin", "scaled_density", "n_queries", "n_wrong", "wrong_fraction", "kde_bandwidth"],
        )
        writer.writeheader()
        for margin, model in [(margin_u, "UltraMS"), (margin_d, "DreaMS")]:
            wrong = margin[margin < 0]
            wrong_fraction = len(wrong) / len(margin)
            kde = gaussian_kde(wrong, bw_method=0.08)
            x_values = np.linspace(wrong.min() - 0.01, -1e-4, 300)
            for x_value, density in zip(x_values, kde(x_values) * wrong_fraction):
                writer.writerow({
                    "model": model,
                    "score_margin": float(x_value),
                    "scaled_density": float(density),
                    "n_queries": int(len(margin)),
                    "n_wrong": int(len(wrong)),
                    "wrong_fraction": float(wrong_fraction),
                    "kde_bandwidth": 0.08,
                })
    else:
        writer = csv.DictWriter(f, fieldnames=["candidate_rank_k", "UltraMS", "DreaMS"])
        writer.writeheader()
        for k in range(1, 1001):
            writer.writerow({
                "candidate_rank_k": k,
                "UltraMS": float((ranks_u <= k).mean() * 100),
                "DreaMS": float((ranks_d <= k).mean() * 100),
            })
print(f"Saved {plot_data_csv}")
