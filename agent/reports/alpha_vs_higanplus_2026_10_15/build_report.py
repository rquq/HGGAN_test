"""Build the self-contained HTML and the portable PDF rendering input.

Run in the existing HiGANplus Python environment. No model is imported.
"""
from pathlib import Path
import re
import json
import markdown
from figures import make_figures

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'alpha_vs_higanplus_vi.md'
FIGURES = make_figures()


def markup(text):
    rendered = markdown.markdown(text, extensions=['tables'])
    return re.sub(r'^<p>|</p>$', '', rendered)


def blocks(text):
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        figure = re.fullmatch(r'\{\{figure:([^|]+)\|(.+)\}\}', line)
        if figure:
            name, caption = figure.groups()
            result.append({'type':'figure', 'name':name, 'caption':caption})
            i += 1
        elif line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            result.append({'type':'code','text':'\n'.join(code)})
            i += 1
        elif re.match(r'^#{1,6} ', line):
            level = len(line.split(' ',1)[0])
            heading=re.sub(r'^(\d+)\. ',r'\1\\. ',line[level+1:])
            result.append({'type':'heading','level':level,'html':markup(heading)})
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',cell) for cell in row):
                    rows.append([markup(cell) for cell in row])
                i += 1
            result.append({'type':'table','rows':rows})
        elif line.startswith('- ') or re.match(r'^\d+\. ',line):
            text = re.sub(r'^(?:- |\d+\. )','',line)
            result.append({'type':'bullet','html':markup(text)})
            i += 1
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','{{figure:','|','- ')):
                paragraph.append(lines[i].strip())
                i += 1
            result.append({'type':'paragraph','html':markup(' '.join(paragraph))})
    return result


def main():
    text = SOURCE.read_text(encoding='utf-8')
    html_source = re.sub(r'\{\{figure:([^|]+)\|(.+?)\}\}',
                         lambda m: f'<figure id="{m[1]}">{FIGURES[m[1]].svg()}<figcaption>{m[2]}</figcaption></figure>',
                         text)
    converter=markdown.Markdown(extensions=['tables','fenced_code','toc','attr_list'])
    body=converter.convert(html_source)
    toc='<details class="contents"><summary>Mục lục</summary>'+converter.toc+'</details>'
    css = '''
    :root { color-scheme:light; --ink:#233344; --accent:#255e78; }
    body { margin:0; color:var(--ink); background:#eef1f4; font:17px/1.75 Arial,"DejaVu Sans",sans-serif; }
    article { max-width:1120px; margin:32px auto; padding:56px 64px; background:white; box-shadow:0 3px 24px #18202a12; }
    h1 { font:700 37px/1.25 Georgia,serif; margin-bottom:26px; }
    h2 { font-size:26px; margin-top:46px; border-bottom:1px solid #cfd9e0; padding-bottom:10px; }
    h3 { font-size:20px; margin-top:30px; color:#255e78; }
    p { margin:12px 0 18px; } a { color:#175e88; overflow-wrap:anywhere; }
    table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.5; margin:24px 0; }
    th { background:#eaf0f5; text-align:left; } td,th { border-bottom:1px solid #d6dfe6; padding:10px 11px; vertical-align:top; }
    tr:nth-child(even) td { background:#fafbfc; }
    code { font:0.85em "DejaVu Sans Mono",Consolas,monospace; background:#f0f4f7; padding:2px 4px; border-radius:3px; overflow-wrap:anywhere; }
    pre { background:#f4f7f9; padding:18px; border-left:3px solid #668fa7; white-space:pre-wrap; line-height:1.6; }
    pre code { background:none; padding:0; } figure { margin:34px -28px; break-inside:avoid; }
    svg { width:100%; height:auto; } figcaption { font-size:14px; line-height:1.6; padding:12px 28px; color:#45586b; }
    .notice { background:#fff5df; padding:15px 20px; border-left:4px solid #b38325; }
    .contents { padding:18px 22px; border:1px solid #d6dfe6; margin-bottom:36px; background:#f8fafc; }
    .contents summary { cursor:pointer; font-weight:700; } .contents ul { font-size:14px;line-height:1.65; }
    @media print { body {background:white;font-size:10pt;} article {padding:0;margin:0;max-width:none;box-shadow:none;}
       h2 {break-after:avoid;} figure {margin:15px 0;} table {font-size:8pt;} a {color:inherit;} }
    '''
    destination = ROOT / 'alpha_vs_higanplus_vi.html'
    destination.write_text('<!doctype html><html lang="vi"><head><meta charset="utf-8">'
                           '<meta name="viewport" content="width=device-width,initial-scale=1">'
                           '<title>ALPHA so với HiGAN+ — báo cáo contribution</title>'
                           f'<style>{css}</style></head><body><article>{toc}{body}</article></body></html>',encoding='utf-8')
    output_blocks = blocks(text)
    (ROOT/'report_blocks.json').write_text(json.dumps({'blocks':output_blocks,
                          'figures':{key:{'width':value.width,'height':value.height,'commands':value.commands}
                                     for key,value in FIGURES.items()}},ensure_ascii=False),encoding='utf-8')
    folder = ROOT/'figures'
    folder.mkdir(exist_ok=True)
    for name, diagram in FIGURES.items():
        (folder/(name+'.svg')).write_text(diagram.svg(),encoding='utf-8')
    print(f'HTML: {destination}; {len(output_blocks)} content blocks; {len(FIGURES)} embedded vector diagrams.')


if __name__ == '__main__':
    main()
