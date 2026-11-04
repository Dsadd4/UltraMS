"""Keep the original width; stack the existing metrics above horizontal DB bars."""
from pathlib import Path
import copy
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/compact_v2'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
COLORS = {'Raw spectrum': '#D9D9D9', 'DreaMS': '#E9A0AA', 'UltraMS': '#3965BB'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 6.5,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.linewidth': .55, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black',
                         'ytick.color': 'black'})
    original_path = ROOT / 'source_data/original_figure2c.svg'
    root = ET.parse(original_path).getroot()
    width, height = map(float, root.get('viewBox').split()[2:])
    parent = {c: p for p in root.iter() for c in p}
    axes = next(e for e in root.iter() if e.get('id') == 'axes_3')
    legend = copy.deepcopy(next(e for e in axes.iter() if e.get('id') == 'legend_1'))
    parent[axes].remove(axes)
    # Original UMAP and molecular groups retain their original SVG coordinates.
    offset = 215.7
    panel_width = width - offset
    left = (227.089553 - offset) / panel_width
    plot_width = (309.26592 - 227.089553) / panel_width
    fig = plt.figure(figsize=(panel_width/72, height/72))
    ax = fig.add_axes([left, .55, plot_width, .32])
    metrics = pd.read_csv(ROOT / 'source_data/original_cluster_metrics.csv')
    indexed = metrics.set_index(['display_name', 'metric'])
    order = ['nmi', 'ari', 'silhouette']
    for j, (name, color) in enumerate(COLORS.items()):
        values = np.array([indexed.loc[(name, m), 'value'] for m in order])
        low = np.array([indexed.loc[(name, m), 'ci_low'] for m in order])
        high = np.array([indexed.loc[(name, m), 'ci_high'] for m in order])
        x = np.arange(3) + (j-1)*.22
        ax.bar(x, values, width=.22, color=color, linewidth=0, zorder=2)
        ax.errorbar(x, values, yerr=[values-low, high-values], fmt='none',
                    ecolor='#303438', elinewidth=.55, capsize=1.3, capthick=.55, zorder=3)
    ax.set_xticks(range(3), ['NMI', 'ARI', 'Silhouette'])
    ax.set_ylim(0, .50)
    ax.set_yticks([0, .2, .4])
    ax.set_xlim(-.47, 2.47)
    db = pd.read_csv(ROOT / 'source_data/db_summary.csv').set_index('display_name')
    lower = fig.add_axes([left, .15, plot_width, .23])
    for j, (name, color) in enumerate(COLORS.items()):
        y = 2-j
        value = float(db.loc[name, 'value'])
        lower.barh(y, value, height=.6, color=color, linewidth=0, zorder=2)
        lower.text(value+.16, y, f'{value:.2f}', va='center', ha='left', fontsize=6)
    lower.set_xlim(0, 10)
    lower.set_ylim(-.6, 2.6)
    lower.set_xticks([0, 4, 8])
    lower.set_yticks([])
    lower.set_xlabel('DB index ↓', labelpad=1)
    for a in [ax, lower]:
        a.spines[['top', 'right']].set_visible(False)
        a.tick_params(width=.5, length=2.2, pad=1.5)
        a.set_axisbelow(True)
    ax.grid(axis='y', color='#E7E9EA', linewidth=.45)
    # No additional legend: reuse the exact accepted color legend at the top.
    tmp_svg = OUT / 'stacked_metrics.svg'
    fig.savefig(tmp_svg)
    plt.close(fig)
    text = tmp_svg.read_text()
    for ident in sorted(re.findall(r'\bid="([^"]+)"', text), key=len, reverse=True):
        text = text.replace(f'id="{ident}"', f'id="stack_{ident}"')
        text = text.replace(f'#{ident}"', f'#stack_{ident}"')
        text = text.replace(f'#{ident})', f'#stack_{ident})')
    overlay = ET.fromstring(text)
    overlay.set('x', str(offset)); overlay.set('y', '0')
    overlay.set('width', str(panel_width)); overlay.set('height', str(height))
    root.append(overlay)
    root.append(legend)
    ids = [e.get('id') for e in root.iter() if e.get('id')]
    assert len(ids) == len(set(ids))
    assert not list(root.iter(f'{{{NS}}}image'))
    target = OUT / 'figure2c_with_db_compact.svg'
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(target.with_suffix('.pdf')), str(target)], check=True)
    subprocess.run(['rsvg-convert', '-d', '600', '-p', '600', '-o', str(target.with_suffix('.png')), str(target)], check=True)
    manifest = {'width_pt': width, 'height_pt': height, 'width_mm': width*25.4/72,
                'height_mm': height*25.4/72, 'upper_ylim': [0, .5],
                'upper_axes_height_pt': height*.32, 'db_orientation': 'horizontal',
                'model_order_top_to_bottom': list(COLORS),
                'original_umap_and_molecule_groups': 'unchanged coordinates and physical size',
                'original_legend': 'unchanged', 'db_values': db.value.to_dict(),
                'original_svg_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()}
    (OUT/'render_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
