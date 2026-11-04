#!/usr/bin/env python3
"""Fig. 3c without the property strip, taken from the paper's own fig3.pdf: objects outside the kept region are deleted, not covered."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[4] / 'Ultramsp/sec/figs/fig3.pdf'
MM = 72 / 25.4
pymupdf.TOOLS.mupdf_display_errors(False)
src = pymupdf.open(PAPER); page = src[0]
y0, y1 = 60.0 * MM, 115.9 * MM     # first heading of panel c to just above the Top-20 strip, read from the paper
keep = pymupdf.Rect(3 * MM, y0, 207 * MM, y1)
W, H = page.rect.width, page.rect.height
doc = pymupdf.open(PAPER); p = doc[0]
for r in (pymupdf.Rect(0, 0, W, y0), pymupdf.Rect(0, y1, W, H), pymupdf.Rect(0, y0, keep.x0, y1), pymupdf.Rect(keep.x1, y0, W, y1)):
    p.add_redact_annot(r)
p.apply_redactions(images=2, graphics=2, text=0)
p.clean_contents()
# "isomer #1" sat on "target #1" in the ranking plot: its eight glyphs move to the right of the diamond, nothing else changes.
DX = 45.0
xref = p.get_contents()[0]
assert len(p.get_contents()) == 1
stream = doc.xref_stream(xref)
DY = -2.0
for x, y, end in (('490.951', '587.687', ' .8240051'), ('490.951', '588.87', ' .8240051')):
    token = f'{x} {y}{end}'.encode()
    assert stream.count(token) == 1, token
    stream = stream.replace(token, f'{float(x) + DX:.3f} {float(y) + DY:.3f}{end}'.encode())
for x, y in (('492.3286', '585.4631'), ('496.6087', '586.1306'), ('499.5593', '587.6869'), ('506.804', '586.3978'), ('509.0806', '584.5753'), ('514.9455', '587.1534'), ('519.2292', '584.5753')):
    token = f'1 0 0 1 {x} {y} cm'.encode()
    assert stream.count(token) == 1, token
    stream = stream.replace(token, f'1 0 0 1 {float(x) + DX:.4f} {float(y) + DY:.4f} cm'.encode())
doc.update_stream(xref, stream)
p.set_cropbox(keep)
out = ROOT / 'figure3c.pdf'
doc.save(out, garbage=4, deflate=True, clean=True)
new = pymupdf.open(out)[0]
Z = pymupdf.Matrix(500 / 72, 500 / 72)
new.get_pixmap(matrix=Z).save(ROOT / 'figure3c.png')
(ROOT / 'figure3c.svg').write_text(new.get_svg_image(text_as_path=False))

m = pymupdf.Matrix(300 / 72, 300 / 72)
arr = lambda q: np.frombuffer(q.samples, dtype=np.uint8).reshape(q.height, q.width, q.n)[:, :, :3].astype(int)
full = pymupdf.open(out)[0]; full.set_cropbox(full.mediabox)
a, b = arr(full.get_pixmap(matrix=m, clip=keep)), arr(page.get_pixmap(matrix=m, clip=keep))
h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
d = np.abs(a[:h, :w] - b[:h, :w]).sum(2)
text = new.get_text()
report = dict(source=str(PAPER.relative_to(ROOT.parents[4])), source_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(), kept_mm=[round(v / MM, 2) for v in keep],
              size_mm=[round(new.rect.width / MM, 2), round(new.rect.height / MM, 2)], changed_pixels_300dpi=int((d > 30).sum()), strong_changes_outside_label=int((d > 150)[:, :int((491 - keep.x0) * 300 / 72) - 4].sum() + (d > 150)[:, int((575 - keep.x0) * 300 / 72):].sum() + (d > 150)[int((262 - keep.y0) * 300 / 72):, :].sum() + (d > 150)[:int((251.5 - keep.y0) * 300 / 72), :].sum()), total_pixels=int(d.size),
              strip_text_left=[k for k in ('Top-20', 'TPSA', 'cLogP', 'darker') if k in text], other_panel_text_left=[k for k in ('Fourier', 'Hit@1', 'Single') if k in text],
              moved_label='isomer #1', moved_pt=[DX, DY], fonts=sorted({f[3] for f in new.get_fonts()}), bitmaps=len(new.get_images()), file_bytes=out.stat().st_size)
(ROOT / 'EXTRACTION_CHECK.json').write_text(json.dumps(report, indent=1))
print(report)
