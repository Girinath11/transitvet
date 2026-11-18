"""Build the IEEE conference version of the TransitVet paper from ieee.md.

Outputs (in this folder):
  transitvet_ieee.tex / .pdf   IEEEtran conference layout (pdflatex)
  transitvet_ieee.docx         Word file on the IEEE conference Word template (template.docx)
Run:  python3 build_ieee.py && pdflatex transitvet_ieee && pdflatex transitvet_ieee
"""
import re, os, copy, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'ieee.md'), encoding='utf-8').read()


# ---------------- parse ----------------
def parse(src):
    meta = dict(authors=[], refs=[])
    blocks = []                      # (kind, data)
    lines = src.split('\n')
    i = 0
    para = []

    def flush():
        if para:
            blocks.append(('p', ' '.join(para)))
            para.clear()
    while i < len(lines):
        l = lines[i].rstrip()
        if l.startswith('TITLE|'):
            meta['title'] = l[6:]
        elif l.startswith('AUTHOR|'):
            meta['authors'].append(l.split('|')[1:])
        elif l.startswith('ABSTRACT|'):
            meta['abstract'] = l[9:]
        elif l.startswith('KEYWORDS|'):
            meta['keywords'] = l[9:]
        elif l.startswith('ACK|'):
            meta['ack'] = l[4:]
        elif l.startswith('REF|'):
            meta['refs'].append(l[4:])
        elif l.startswith('# '):
            flush(); blocks.append(('h1', l[2:]))
        elif l.startswith('## '):
            flush(); blocks.append(('h2', l[3:]))
        elif l.startswith('- '):
            flush()
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(lines[i][2:].strip()); i += 1
            blocks.append(('ul', items)); continue
        elif l.startswith('FIG|'):
            flush(); _, f, w, cap = l.split('|', 3); blocks.append(('fig', (f, w, cap)))
        elif l.startswith('TABLE|'):
            flush()
            cap = l[6:]; i += 1
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            blocks.append(('table', (cap, rows[0], rows[2:]))); continue
        elif not l.strip():
            flush()
        else:
            para.append(l.strip())
        i += 1
    flush()
    return meta, blocks


META, BLOCKS = parse(SRC)


# ---------------- LaTeX ----------------
def tex_inline(t):
    t = t.replace('\\', '\\textbackslash{}')
    for a, b in (('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'), ('$', '\\$')):
        t = t.replace(a, b)
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'\*(.+?)\*', r'\\textit{\1}', t)
    t = re.sub(r'"(.+?)"', r"``\1''", t)
    t = re.sub(r'\[(\d+)\]', r'\\cite{r\1}', t)
    t = t.replace('\\cite{r', '\\cite{r')
    t = re.sub(r'(\d)×10⁻⁸', r'\1$\\times10^{-8}$', t)
    rep = {'±': '$\\pm$', 'α': '$\\alpha$', '×': '$\\times$', '−': '$-$', '–': '--', 'µ': '$\\mu$', 'ō': '\\={o}',
           'á': "\\'a", '≥': '$\\geq$'}
    for a, b in rep.items():
        t = t.replace(a, b)
    t = re.sub(r'^\[', '{[}', t)
    return t


def ref_tex(r):
    r = tex_inline(r)
    return r


def latex():
    out = []
    for kind, d in BLOCKS:
        if kind == 'h1':
            out.append(f'\\section{{{tex_inline(d)}}}')
        elif kind == 'h2':
            out.append(f'\\subsection{{{tex_inline(d)}}}')
        elif kind == 'p':
            out.append(tex_inline(d) + '\n')
        elif kind == 'ul':
            out.append('\\begin{itemize}\n' + '\n'.join('\\item ' + tex_inline(x) for x in d) + '\n\\end{itemize}')
        elif kind == 'fig':
            f, w, cap = d
            env = 'figure*' if w == 'full' else 'figure'
            width = '0.95\\textwidth' if w == 'full' else '\\columnwidth'
            if f == 'fig_pipeline.png':
                width = '0.8\\textwidth'
            pos = '!t' if w == 'full' else '!ht'
            out.append(f'\\begin{{{env}}}[{pos}]\n\\centering\n\\includegraphics[width={width}]{{figures/{f}}}\n'
                       f'\\caption{{{tex_inline(cap)}}}\n\\label{{fig:{f[4:-4]}}}\n\\end{{{env}}}')
        elif kind == 'table':
            cap, head, rows = d
            n = len(head)
            wide = sum(max(len(r[j]) for r in [head] + rows) for j in range(n)) > 62
            env = 'table*' if wide else 'table'
            num = [all(re.fullmatch(r'[\d.,%±–\-\s*]+|–|', r[j]) for r in rows) for j in range(n)]
            spec = ''.join('c' if num[j] and j else 'l' for j in range(n))
            body = ' \\\\\n'.join(' & '.join(tex_inline(c) for c in r) for r in rows)
            pos = '!t' if wide else '!ht'
            out.append(f'\\begin{{{env}}}[{pos}]\n\\caption{{{tex_inline(cap)}}}\n\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}\n'
                       f'\\begin{{tabular}}{{{spec}}}\n\\hline\n' + ' & '.join('\\textbf{' + tex_inline(h) + '}' for h in head) +
                       ' \\\\\n\\hline\n' + body + ' \\\\\n\\hline\n\\end{tabular}\n' + f'\\end{{{env}}}')
    blocks_ = []
    for a in META['authors']:
        name, *aff = a
        lines = [('\\textit{' + tex_inline(x) + '}') if 'Dept.' in x or 'Institute' in x else tex_inline(x) for x in aff]
        blocks_.append('\\parbox[t]{0.32\\textwidth}{\\centering ' + tex_inline(name) + '\\\\[3pt]\n\\footnotesize ' +
                       '\\\\\n'.join(lines) + '}')
    row1 = '\\hspace{0.02\\textwidth}'.join(blocks_[:3])
    row2 = '\\hspace{0.04\\textwidth}'.join(blocks_[3:])
    authors_tex = ('\\IEEEauthorblockN{\\makebox[\\textwidth][c]{' + row1 + '}\\\\[10pt]\n'
                   '\\makebox[\\textwidth][c]{' + row2 + '}}')
    refs = '\n'.join(f'\\bibitem{{r{i + 1}}} {ref_tex(r)}' for i, r in enumerate(META['refs']))
    return r"""\documentclass[conference,a4paper]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{url}
\makeatletter
\newcommand{\linebreakand}{\end{@IEEEauthorhalign}\hfill\mbox{}\par\mbox{}\hfill\begin{@IEEEauthorhalign}}
\makeatother
\begin{document}

\title{""" + tex_inline(META['title']) + r"""}

\author{""" + authors_tex + r"""}

\maketitle

\begin{abstract}
""" + tex_inline(META['abstract']) + r"""
\end{abstract}

\begin{IEEEkeywords}
""" + tex_inline(META['keywords'].rstrip('.')) + r"""
\end{IEEEkeywords}

""" + '\n'.join(out) + r"""

\section*{Acknowledgment}
""" + tex_inline(META['ack']) + r"""

\begin{thebibliography}{00}
""" + refs + r"""
\end{thebibliography}

\end{document}
"""


open(os.path.join(HERE, 'transitvet_ieee.tex'), 'w', encoding='utf-8').write(latex())
print('tex ok')


# ---------------- Word (IEEE conference template) ----------------
def build_docx(template, out):
    import docx
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    d = docx.Document(template)
    body = d.element.body
    sects = [copy.deepcopy(s) for s in body.iter(qn('w:sectPr'))]
    s_title, s_auth3, s_body2, s_final = sects[0], sects[1], sects[-2], sects[-1]
    for ch in list(body):
        body.remove(ch)
    body.append(s_final)

    def para(style=None, align=None):
        p = d.add_paragraph(style=style)
        if align is not None:
            p.alignment = align
        return p

    def runs(p, text, size=None):
        # **bold**, *italic*
        for part in re.split(r'(\*\*.+?\*\*|\*.+?\*)', text):
            if not part:
                continue
            if part.startswith('**'):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith('*'):
                r = p.add_run(part[1:-1]); r.italic = True
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
        return p

    def end_section(p, sect, ncols=None):
        s = copy.deepcopy(sect)
        if ncols is not None:
            s.find(qn('w:cols')).set(qn('w:num'), str(ncols))
            s.find(qn('w:cols')).set(qn('w:space'), '160')
        p._p.get_or_add_pPr().append(s)

    t = runs(para('papertitle'), META['title'])
    end_section(t, s_title)
    groups = [META['authors'][:3], META['authors'][3:]]
    for g in groups:
        p = para('Author')
        for i, a in enumerate(g):
            name, *aff = a
            if i:
                p.add_run().add_break(docx.enum.text.WD_BREAK.COLUMN)
            r = p.add_run(name); r.font.size = Pt(11)
            for x in aff:
                r = p.add_run(); r.add_break()
                r = p.add_run(x); r.font.size = Pt(8.5)
                r.italic = ('Dept.' in x or 'Institute' in x)
        if g is groups[-1]:
            p.paragraph_format.space_after = Pt(14)
        end_section(p, s_auth3, len(g))
    p = para('Abstract')
    r = p.add_run('Abstract—'); r.bold = True; r.italic = True
    r = p.add_run(META['abstract']); r.bold = True
    p = para('Keywords')
    r = p.add_run('Keywords—'); r.bold = True; r.italic = True
    r = p.add_run(META['keywords']); r.bold = True; r.italic = True

    figdir = os.path.join(HERE, 'figures')
    for kind, dd in BLOCKS:
        if kind == 'h1':
            runs(para('Heading1'), dd)
        elif kind == 'h2':
            runs(para('Heading2'), dd)
        elif kind == 'p':
            runs(para('BodyText'), dd)
        elif kind == 'ul':
            for x in dd:
                runs(para('bulletlist'), x)
        elif kind == 'fig':
            f, w, cap = dd
            p = para(None, WD_ALIGN_PARAGRAPH.CENTER)
            p.add_run().add_picture(os.path.join(figdir, f), width=Inches(3.45))
            runs(para('figurecaption'), cap)
        elif kind == 'table':
            cap, head, rows = dd
            runs(para('tablehead'), cap)
            tb = d.add_table(rows=1 + len(rows), cols=len(head))
            tb.style = 'Table Grid'
            tb.alignment = 1
            tb.autofit = False
            lens = [max(len(re.sub(r'\*', '', r_[j])) for r_ in [head] + rows) + 3 for j in range(len(head))]
            widths = [Inches(3.4 * L / sum(lens)) for L in lens]
            tblPr = tb._tbl.tblPr
            lay = tblPr.makeelement(qn('w:tblLayout'), {qn('w:type'): 'fixed'}); tblPr.append(lay)
            for row_ in tb.rows:
                for j, c in enumerate(row_.cells):
                    c.width = widths[j]
            for j, col in enumerate(tb.columns):
                col.width = widths[j]
            for j, h in enumerate(head):
                c = tb.cell(0, j); c.paragraphs[0].style = 'tablecolhead'
                runs(c.paragraphs[0], '**' + h + '**' if not h.startswith('**') else h, 8)
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    c = tb.cell(i + 1, j); c.paragraphs[0].style = 'tablecopy'
                    runs(c.paragraphs[0], v, 8)
                    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            para('BodyText')
    p = para('Heading5')
    p.add_run('Acknowledgment')
    runs(para('BodyText'), META['ack'])
    p = para('Heading5'); p.add_run('References')
    last = None
    for r_ in META['refs']:
        last = runs(para('references'), r_)
    end_section(last, s_body2)
    # drop images and other parts left over from the template's own body
    used = set(re.findall(r'r:(?:embed|id)="([^"]+)"', d.element.xml))
    for rid, rel in list(d.part.rels.items()):
        if 'image' in rel.reltype and rid not in used:
            d.part.drop_rel(rid)
    d.core_properties.author = 'Girinath A'
    d.core_properties.title = META['title']
    d.save(out)


if __name__ == '__main__':
    tpl = os.path.join(HERE, 'ieee_template.docx')
    if os.path.exists(tpl):
        build_docx(tpl, os.path.join(HERE, 'transitvet_ieee.docx'))
        print('docx ok')
