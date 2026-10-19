"""Reopen PDF, verify figure/citation coverage, and record page metadata."""
from pathlib import Path
import json
import re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
reader=PdfReader(ROOT/'alpha_vs_higanplus_vi.pdf')
texts=[page.extract_text() or '' for page in reader.pages]
all_text='\n'.join(texts)
landscape_pages=[index+1 for index,page in enumerate(reader.pages)
                 if float(page.mediabox.width)>float(page.mediabox.height)]
figures={str(number):[index+1 for index,text in enumerate(texts)
                     if f'Hình {number}.' in text] for number in range(1,8)}
for number,pages in figures.items():
    assert len(pages)==1,(number,pages)
for number in range(1,8):
    assert f'[P{number}]' in all_text
assert len(landscape_pages)==7,landscape_pages
assert '2.446330' in all_text and '0.063558' in all_text
assert 'chưa có FID/KID/HWD' in all_text
short_pages=[(i+1,len(text.strip()),text.strip()) for i,text in enumerate(texts) if len(text.strip())<=60]
assert not short_pages,('unexpected empty page',short_pages)
html=(ROOT/'alpha_vs_higanplus_vi.html').read_text(encoding='utf-8')
assert html.count('<svg ')==7
assert '{{figure:' not in html
assert '<script' not in html
for path in (ROOT/'figures').glob('*.pdf'):
    assert len(PdfReader(path).pages)==1
metadata={'pages':len(reader.pages),'landscape_pages':landscape_pages,
          'figure_pages':figures,'extracted_text_characters':len(all_text),
          'html_embedded_svg_count':7,'pdf_bytes':(ROOT/'alpha_vs_higanplus_vi.pdf').stat().st_size,
          'html_bytes':(ROOT/'alpha_vs_higanplus_vi.html').stat().st_size,
          'claims':'Reopened generated artifacts; no model training or metric evaluation.'}
(ROOT/'artifact_verification.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=False,indent=2))
