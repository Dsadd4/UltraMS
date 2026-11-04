#!/usr/bin/env python3
"""Fig. 3e, 102.5 x 68 mm: same data as the paper panel, axes rebuilt so curves and bars are in proportion."""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source_data'
W, H = 102.5, 68.0
FS = 6.5
INK = '#1A1A1A'
COL = {'DeepSets': '#C48A4A', 'Codebook': '#5AA69A', 'Linear': '#A7A7A7', 'Fourier': '#8767AD', 'DreaMS': '#E9A0AA', 'UltraMS': '#3965BB'}
LW = {'UltraMS': 1.2, 'DreaMS': 1.0}
GROUPS = {'pos': ('[M+H-H2O]+', '[M+NH4]+', '[M+Na]+'), 'neg': ('[M+Cl]-', '[M+FA]-')}
PAPER_BARS = {'pos': dict(Codebook=1.90, DeepSets=6.03, DreaMS=10.83, Linear=12.26, Fourier=14.25, UltraMS=17.58),
              'neg': dict(DeepSets=3.98, Codebook=4.87, Linear=5.82, Fourier=6.01, DreaMS=32.38, UltraMS=35.72)}   # read from the vector bars of Ultramsp/sec/figs/fig3.pdf
TITLE = {'pos': '[M+H]$^{+}$', 'neg': '[M−H]$^{−}$'}
LABEL = {'pos': 'Additional positive', 'neg': 'Additional negative'}


def box(x0, y0, x1, y1):
    return [x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H]


def spines(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.55)
    ax.tick_params(length=2, width=.55, pad=1.5, labelsize=FS)


def main():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom',
                         'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.default': 'regular'})
    cur = pd.read_csv(SRC / 'fdr_envelopes_primary_seed0.csv.gz')
    work = pd.read_csv(SRC / 'working_points_primary_seed0.csv')
    low = pd.read_csv(SRC / 'other_strict_adduct_low_fdr_summary.csv')
    assert set(cur.method) == set(COL) == set(work.method) == set(low.method)

    bars, table = {}, []
    for g, adducts in GROUPS.items():
        s = low[low.adduct.isin(adducts)]
        assert set(s.adduct) == set(adducts)
        bars[g] = {m: 100 * float(np.average(x.mean_identification_recall_fdr_0_5pct, weights=x.n_query)) for m, x in s.groupby('method')}
        for m, v in bars[g].items():
            assert abs(v - PAPER_BARS[g][m]) < .02, (g, m, v)
            table.append(dict(group=LABEL[g], method=m, n_query=int(s[s.method == m].n_query.sum()), mean_micro_recall_fdr_0_5pct=v))
    pd.DataFrame(table).to_csv(SRC / 'fig3e_additional_adduct_bars.csv', index=False)

    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
    cols = {'pos': (10.2, 51.2), 'neg': (59.3, 100.3)}
    ctop, cbot, btop, bbot = 8.2, 38.6, 48.4, 62.8
    for g, (x0, x1) in cols.items():
        ax = fig.add_axes(box(x0, ctop, x1, cbot))
        c = cur[cur['mode'] == g]
        ax.axhline(100, color='none')
        for y in (25, 50, 75, 100):
            ax.axhline(y, color='#EDEFF2', lw=.5, zorder=0)
        ax.axvline(5, color=INK, lw=.5, ls=(0, (1, 2)), zorder=1)
        for m in COL:
            f = c[c.method == m].sort_values('fdr_budget')
            ax.plot(100 * f.fdr_budget, 100 * f.identification_recall_envelope, color=COL[m], lw=LW.get(m, .75), solid_capstyle='round', solid_joinstyle='round', zorder=3, clip_on=True)
        for m in COL:
            p = work[(work['mode'] == g) & (work.method == m)].iloc[0]
            ax.scatter([5], [100 * p.identification_recall], s=11, color=COL[m], edgecolor='white', linewidth=.35, zorder=5, clip_on=False)
        ax.set_xlim(0, 7.5); ax.set_ylim(0, 102)
        ax.set_xticks([0, 2.5, 5, 7.5], ['0', '2.5', '5', '7.5']); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel('False discovery rate (%)', fontsize=FS, labelpad=1.5)
        ax.set_title(TITLE[g], fontsize=FS, pad=2.5)
        if g == 'pos':
            ax.set_ylabel('Recall (%)', fontsize=FS, labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
        spines(ax)

        bx = fig.add_axes(box(x0, btop, x1, bbot))
        ranked = sorted(bars[g], key=bars[g].get)
        for i, m in enumerate(ranked):
            v = bars[g][m]
            bx.bar(i, v, width=.84, color=COL[m], linewidth=0)
            bx.text(i, v + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=6.0)
        bx.set_xlim(-.62, 5.62); bx.set_ylim(0, 42); bx.set_yticks([0, 20, 40]); bx.set_xticks([])
        bx.set_xlabel(LABEL[g], fontsize=FS, labelpad=2.6)
        if g == 'pos':
            bx.set_ylabel('Mean micro recall,\nFDR 0–5% (%)', fontsize=FS, labelpad=1.5, linespacing=1.02)
        else:
            bx.tick_params(labelleft=False)
        spines(bx)

    sc = fig.add_axes([0, 0, 1, 1], zorder=10); sc.set_xlim(0, W); sc.set_ylim(H, 0); sc.axis('off'); sc.patch.set_alpha(0)
    names = list(COL)
    widths = [3.1 + .8 + plt.matplotlib.textpath.TextPath((0, 0), n, size=FS, prop='Arial').get_extents().width * 25.4 / 72 for n in names]
    gap = 2.6
    x = (W - sum(widths) - gap * (len(names) - 1)) / 2
    for n, w in zip(names, widths):
        sc.plot([x, x + 3.1], [1.9, 1.9], color=COL[n], lw=LW.get(n, 1.0) + .25, solid_capstyle='round')
        sc.text(x + 3.9, 1.95, n, va='center', fontsize=FS)
        x += w + gap

    (ROOT / 'FIGURE_CHECKS.json').write_text(json.dumps(dict(size_mm=[W, H], curve_axes_mm=[41, cbot - ctop], bar_axes_mm=[41, bbot - btop],
                                                              bars=bars, paper_bars=PAPER_BARS), indent=1))
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / f'figure3e.{ext}', dpi=500)


if __name__ == '__main__':
    main()
