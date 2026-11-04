"""Append a separately scaled DB axis, keeping the accepted Fig. 2c SVG intact."""
from pathlib import Path
import copy
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'fig2c_supervised_structure_families'
OUT = ROOT / 'output'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def main():
    OUT.mkdir(exist_ok=True)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 6.5,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.linewidth': .55, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black',
                         'ytick.color': 'black'})
    rows = pd.read_csv(ROOT / 'source_data/db_summary.csv').set_index('display_name')
    old_path = BASE / 'figure2c_supervised_negative_structure_families.svg'
    old = ET.parse(old_path).getroot()
    width, height = map(float, old.get('viewBox').split()[2:])
    added = 85.04
    fig = plt.figure(figsize=(added / 72, height / 72))
    ax = fig.add_axes([.24, .247, .72, .623])
    for i, (name, color) in enumerate([('Raw spectrum', '#D9D9D9'),
                                      ('DreaMS', '#E9A0AA'), ('UltraMS', '#3965BB')]):
        value = rows.loc[name, 'value']
        ax.bar(i, value, width=.66, color=color, linewidth=0)
        ax.text(i, value + .16, f'{value:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xlim(-.6, 2.6)
    ax.set_ylim(0, 9.3)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_xticks([1], ['DB index ↓'])
    ax.tick_params(width=.5, length=2.2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E7E9EA', linewidth=.45)
    ax.set_axisbelow(True)
    fig.savefig(OUT / 'db_axis.svg')
    fig.savefig(OUT / 'db_axis.pdf')
    fig.savefig(OUT / 'db_axis.png', dpi=600)
    plt.close(fig)

    # Prefix every new SVG ID so clip paths and glyphs cannot collide with the original.
    new_text = (OUT / 'db_axis.svg').read_text()
    ids = re.findall(r'\bid="([^"]+)"', new_text)
    for ident in sorted(ids, key=len, reverse=True):
        new_text = new_text.replace(f'id="{ident}"', f'id="db_{ident}"')
        new_text = new_text.replace(f'#{ident}"', f'#db_{ident}"')
        new_text = new_text.replace(f'#{ident})', f'#db_{ident})')
    new = ET.fromstring(new_text)
    combined = ET.Element(f'{{{NS}}}svg', {'width': f'{width+added}pt', 'height': f'{height}pt',
                                         'viewBox': f'0 0 {width+added} {height}', 'version': '1.1'})
    old = copy.deepcopy(old)
    old.set('x', '0'); old.set('y', '0')
    old.set('width', str(width)); old.set('height', str(height))
    new.set('x', str(width)); new.set('y', '0')
    new.set('width', str(added)); new.set('height', str(height))
    combined.append(old)
    combined.append(new)
    assert not list(combined.iter(f'{{{NS}}}image')), 'Raster images are prohibited'
    target = OUT / 'figure2c_with_db.svg'
    ET.ElementTree(combined).write(target, encoding='utf-8', xml_declaration=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(OUT / 'figure2c_with_db.pdf'), str(target)], check=True)
    subprocess.run(['rsvg-convert', '-d', '600', '-p', '600', '-o', str(OUT / 'figure2c_with_db.png'), str(target)], check=True)
    # Reusable vector copies of the six unchanged molecular illustrations.
    mol_dir = ROOT / 'source_data/molecule_svgs'
    mol_dir.mkdir(exist_ok=True)
    molecules = [e for e in old.iter(f'{{{NS}}}svg') if e is not old and e.get('viewBox') == '0 0 300 150']
    assert len(molecules) == 6
    for i, element in enumerate(molecules, 1):
        molecule = copy.deepcopy(element)
        molecule.attrib.pop('x', None); molecule.attrib.pop('y', None)
        ET.ElementTree(molecule).write(mol_dir / f'family_{i}.svg', encoding='utf-8', xml_declaration=True)
    metadata = {'original_svg_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
                'original_width_pt': width, 'original_height_pt': height,
                'new_width_pt': width+added, 'original_content': 'unscaled nested SVG',
                'db_axis': 'original full-cohort point estimates; no bootstrap confidence intervals',
                'n_vector_molecules': len(molecules), 'n_raster_images': 0}
    (OUT / 'render_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
