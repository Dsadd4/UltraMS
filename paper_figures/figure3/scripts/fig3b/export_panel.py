from pathlib import Path
import hashlib,json,shutil
import pymupdf
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
src=ROOT/'Ultramsp/sec/figs/fig3.pdf'
snapshot=HERE/'author_approved_figure3.pdf'
if not snapshot.exists():shutil.copy2(src,snapshot)
doc=pymupdf.open(snapshot);out=pymupdf.open();clip=pymupdf.Rect(312,0,595.276,148)
p=out.new_page(width=clip.width,height=clip.height);p.show_pdf_page(p.rect,doc,0,clip=clip)
out.save(HERE/'figure3b.pdf',garbage=4,deflate=True);p.get_pixmap(dpi=500).save(HERE/'figure3b.png');(HERE/'figure3b.svg').write_text(p.get_svg_image(text_as_path=False))
proof={'source':'Ultramsp/sec/figs/fig3.pdf','source_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'clip_pt':list(clip),'dpi':500,'outputs':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['figure3b.png','figure3b.pdf','figure3b.svg']},'data_changed':False}
(HERE/'qa/EXPORT_VERIFICATION.json').write_text(json.dumps(proof,indent=2))
