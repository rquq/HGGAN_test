"""Render the report with bundled Python/ReportLab, without installing packages.

Run from Windows bundled Python against this script's WSL UNC path. All drawing
commands are shared with the HTML SVG figures, not raster screenshots.
"""
from pathlib import Path
import json
import re
import math
import textwrap
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                               Spacer, Table, TableStyle, Flowable, PageBreak,
                               NextPageTemplate, XPreformatted)

ROOT = Path(__file__).resolve().parent
FONT_ROOT = Path(r'\\wsl.localhost\Ubuntu\usr\share\fonts\truetype\dejavu')
for name,filename in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),
                      ('Serif','DejaVuSerif.ttf'),('SerifBold','DejaVuSerif-Bold.ttf'),
                      ('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT_ROOT/filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='Serif',boldItalic='SerifBold')


def paragraph_markup(text):
    text = re.sub(r'<code>(.*?)</code>',r'<font name="Mono" size="8">\1</font>',text)
    text = text.replace('<strong>','<b>').replace('</strong>','</b>')
    text = text.replace('<em>','<i>').replace('</em>','</i>')
    text = re.sub(r'<a href="([^"]+)"[^>]*>',r'<a href="\1" color="#175e88">',text)
    return text


def draw_diagram(c, diagram):
    height=diagram['height']
    for command in diagram['commands']:
        kind,*args=command
        if kind=='box':
            x,y,w,h,label,fill,size=args
            lines=label.split('\n')
            max_width=max(pdfmetrics.stringWidth(line,'Body',size) for line in lines)
            size=min(size,size*(w-16)/max_width) if max_width else size
            c.setFillColor(colors.HexColor(fill))
            c.setStrokeColor(colors.HexColor('#263746'))
            c.setLineWidth(1.4)
            c.roundRect(x,height-y-h,w,h,5,fill=1,stroke=1)
            c.setFillColor(colors.HexColor('#263746'))
            c.setFont('Body',size)
            for i,line in enumerate(lines):
                baseline=height-(y+h/2+(i-(len(lines)-1)/2)*size*1.32)-size*.32
                c.drawCentredString(x+w/2,baseline,line)
        elif kind=='text':
            x,y,label,size,color,anchor=args
            c.setFillColor(colors.HexColor(color))
            c.setFont('Body',size)
            (c.drawCentredString if anchor=='middle' else c.drawString)(x,height-y,label)
        else:
            points,color,dashed=args
            c.setStrokeColor(colors.HexColor(color))
            c.setFillColor(colors.HexColor(color))
            c.setLineWidth(1.6)
            c.setDash([6,4] if dashed else [])
            path=c.beginPath()
            path.moveTo(points[0][0],height-points[0][1])
            for x,y in points[1:]:
                path.lineTo(x,height-y)
            c.drawPath(path)
            c.setDash([])
            x,y=points[-1]
            px,py=points[-2]
            angle=math.atan2(y-py,x-px)
            path=c.beginPath()
            path.moveTo(x,height-y)
            for sign in (1,-1):
                ax=x-8*math.cos(angle)+sign*4*math.sin(angle)
                ay=y-8*math.sin(angle)-sign*4*math.cos(angle)
                path.lineTo(ax,height-ay)
            path.close()
            c.drawPath(path,stroke=0,fill=1)


class DiagramFlowable(Flowable):
    def __init__(self, diagram, width):
        super().__init__()
        self.diagram=diagram
        self.width=width
        self.height=width*diagram['height']/diagram['width']
    def draw(self):
        self.canv.saveState()
        self.canv.scale(self.width/self.diagram['width'],self.width/self.diagram['width'])
        draw_diagram(self.canv,self.diagram)
        self.canv.restoreState()


def main():
    payload=json.loads((ROOT/'report_blocks.json').read_text(encoding='utf-8'))
    body=ParagraphStyle('body',fontName='Body',fontSize=10,leading=15.5,spaceAfter=9,textColor=colors.HexColor('#263746'))
    heading1=ParagraphStyle('h1',parent=body,fontName='SerifBold',fontSize=23,leading=30,spaceAfter=18,keepWithNext=True)
    heading2=ParagraphStyle('h2',parent=body,fontName='BodyBold',fontSize=15,leading=21,spaceBefore=18,spaceAfter=10,keepWithNext=True)
    heading3=ParagraphStyle('h3',parent=body,fontName='BodyBold',fontSize=11.5,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#255e78'))
    table_style=ParagraphStyle('table',parent=body,fontSize=7.8,leading=11,spaceAfter=0)
    caption=ParagraphStyle('caption',parent=body,fontSize=9,leading=14,spaceBefore=12,spaceAfter=8)
    code=ParagraphStyle('code',fontName='Mono',fontSize=7.6,leading=11,backColor=colors.HexColor('#f1f5f8'),borderPadding=9,spaceBefore=8,spaceAfter=13)
    bullet=ParagraphStyle('bullet',parent=body,leftIndent=12,firstLineIndent=-8)
    normal=A4
    wide=landscape(A4)
    margin=46
    width=normal[0]-2*margin
    wide_width=wide[0]-2*margin
    def header(c,doc):
        size=wide if doc.pageTemplate.id=='Landscape' else normal
        c.setPageSize(size)
        c.saveState()
        c.setStrokeColor(colors.HexColor('#d0dbe3'))
        c.line(margin,size[1]-30,size[0]-margin,size[1]-30)
        c.setFont('Body',8)
        c.setFillColor(colors.HexColor('#53697c'))
        c.drawString(margin,19,'ALPHA / HiGAN+ — Báo cáo kỹ thuật · 15/10/2026')
        c.drawRightString(size[0]-margin,19,str(doc.page))
        c.restoreState()
    destination=ROOT/'alpha_vs_higanplus_vi.pdf'
    doc=BaseDocTemplate(str(destination),pagesize=normal,title='ALPHA so với HiGAN+: contribution và bằng chứng',author='Báo cáo kỹ thuật từ code và checkpoint',leftMargin=margin,rightMargin=margin,topMargin=44,bottomMargin=40)
    doc.addPageTemplates([
        PageTemplate('Portrait',[Frame(margin,40,width,normal[1]-84,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=header,pagesize=normal),
        PageTemplate('Landscape',[Frame(margin,40,wide_width,wide[1]-84,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=header,pagesize=wide),
    ])
    outline_count=0
    def add_outline(flowable):
        nonlocal outline_count
        if isinstance(flowable,Paragraph) and flowable.style.name in ('h1','h2','h3'):
            key=f'heading-{outline_count}'
            outline_count+=1
            doc.canv.bookmarkPage(key)
            doc.canv.addOutlineEntry(flowable.getPlainText(),key,
                 level={'h1':0,'h2':1,'h3':2}[flowable.style.name],closed=False)
    doc.afterFlowable=add_outline
    story=[]
    for block_index, block in enumerate(payload['blocks']):
        kind=block['type']
        if kind=='heading':
            style=heading1 if block['level']==1 else heading2 if block['level']==2 else heading3
            story.append(Paragraph(paragraph_markup(block['html']),style))
        elif kind in ('paragraph','bullet'):
            text=paragraph_markup(block['html'])
            story.append(Paragraph(('• ' if kind=='bullet' else '')+text,bullet if kind=='bullet' else body))
        elif kind=='code':
            from xml.sax.saxutils import escape
            wrapped='\n'.join(textwrap.fill(line,width=88,subsequent_indent='    ',
                              replace_whitespace=False,break_long_words=False)
                              if line else '' for line in block['text'].splitlines())
            story.append(XPreformatted(escape(wrapped),code))
        elif kind=='table':
            rows=[[Paragraph(paragraph_markup(cell),table_style) for cell in row] for row in block['rows']]
            columns=len(rows[0])
            fractions={3:[.21,.32,.47],4:[.16,.27,.30,.27],5:[.25,.20,.18,.18,.19],6:[.20,.20,.13,.13,.13,.21]}.get(columns,[1/columns]*columns)
            table=Table(rows,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f6')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafb')]),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#7894a8')),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
                ('BOTTOMPADDING',(0,0),(-1,-1),6),
            ]))
            story.extend([Spacer(1,5),table,Spacer(1,12)])
        else:
            diagram=payload['figures'][block['name']]
            story.extend([NextPageTemplate('Landscape'),PageBreak(),
                          DiagramFlowable(diagram,wide_width),Paragraph(block['caption'],caption)])
            next_is_figure=(block_index+1<len(payload['blocks'])
                            and payload['blocks'][block_index+1]['type']=='figure')
            if not next_is_figure:
                story.extend([NextPageTemplate('Portrait'),PageBreak()])
    doc.build(story)
    for name,diagram in payload['figures'].items():
        output=ROOT/'figures'/(name+'.pdf')
        c=canvas.Canvas(str(output),pagesize=(diagram['width'],diagram['height']))
        draw_diagram(c,diagram)
        c.save()
    print(f'PDF written: {destination}')


if __name__=='__main__':
    main()
