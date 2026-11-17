"""Build the MNRAS/RASTI LaTeX version of the TransitVet manuscript from ../manuscript.md.

Run from this folder:  python3 build_tex.py && pdflatex transitvet && pdflatex transitvet
Text, tables and figures all come from manuscript.md, so edit that file and re-run.
"""
import re, subprocess, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'manuscript.md')
s = open(SRC, encoding='utf-8').read()


def pandoc(md):
    return subprocess.run(['pandoc', '-f', 'markdown-auto_identifiers', '-t', 'latex', '--wrap=none'],
                          input=md, capture_output=True, text=True, check=True).stdout


def cells_to_tex(cells):
    """Convert many short markdown strings to LaTeX in one pandoc call."""
    mark = 'QQCELLSPLITQQ'
    out = pandoc(f'\n\n{mark}\n\n'.join(c if c.strip() else '\\-' for c in cells))
    parts = [p.strip() for p in out.split(mark)]
    return ['' if p == '\\-' else p for p in parts]


# ---------- pieces of the front matter ----------
abstract = s[s.index('# Abstract') + len('# Abstract'):s.index('**Keywords:**')].strip()
kw_line = s[s.index('**Keywords:**') + len('**Keywords:**'):].split('\n', 1)[0].strip()
keywords = ' -- '.join(k.strip() for k in kw_line.split(';'))
body = s[s.index('# 1 Introduction'):s.index('# References')]
appendix = s[s.index('# Appendix A'):]
refs = s[s.index('# References') + len('# References'):s.index('# Appendix A')].strip().splitlines()


# ---------- tables: "**Table N.** caption" followed by a pipe table ----------
def table_block(m):
    num, cap, tbl = m.group(1), m.group(2).strip(), m.group(3)
    rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in tbl.strip().splitlines()]
    head, data = rows[0], rows[2:]
    ncol = len(head)
    flat = cells_to_tex([cap] + head + [c for r in data for c in r])
    cap_t, head_t = flat[0], flat[1:1 + ncol]
    data_t = [flat[1 + ncol + i * ncol:1 + ncol + (i + 1) * ncol] for i in range(len(data))]
    width = [max(len(r[j]) for r in rows if j < len(r)) for j in range(ncol)]
    dwidth = [max([min(len(head[j]), 12)] + [len(r[j]) for r in data if j < len(r)]) for j in range(ncol)]
    wide = sum(width) > 75
    avail = 17.0 if wide else 8.2                       # cm, MNRAS text / column width
    if sum(width) * 0.17 > avail:                       # too long for plain columns: wrap the long ones
        long = [j for j in range(ncol) if dwidth[j] > 18 or width[j] > 30]
        short = sum(dwidth[j] for j in range(ncol) if j not in long) * 0.17 + 0.4 * ncol
        rest = max(avail - short, 2.5 * len(long))
        tot = sum(dwidth[j] for j in long)
        spec = ''.join(f'>{{\\raggedright\\arraybackslash}}p{{{rest * dwidth[j] / tot:.1f}cm}}' if j in long else ('l' if j == 0 else 'c') for j in range(ncol))
    else:
        spec = 'l' + 'c' * (ncol - 1)
    env = 'table*' if wide else 'table'
    colkind = re.findall(r'p\{[^}]*\}|[lc]', spec.replace('>{\\raggedright\\arraybackslash}', ''))
    def stack(h, k):
        if k.startswith('p') or len(h) <= 14:
            return h
        words, lines, cur = h.split(' '), [], ''
        for w_ in words:
            if cur and len(cur) + 1 + len(w_) > 14:
                lines.append(cur); cur = w_
            else:
                cur = (cur + ' ' + w_).strip()
        lines.append(cur)
        al = 'l' if k == 'l' else 'c'
        return '\\begin{tabular}[b]{@{}' + al + '@{}}' + ' \\\\ '.join(lines) + '\\end{tabular}'
    head_t = [stack(h, k) for h, k in zip(head_t, colkind)]
    label = 'tab:' + num.lower()
    lines = [f'\\begin{{{env}}}', '\\centering', f'\\caption{{{cap_t}}}', f'\\label{{{label}}}',
             f'\\begin{{tabular}}{{{spec}}}', '\\hline', ' & '.join(head_t) + ' \\\\', '\\hline']
    lines += [' & '.join(r) + ' \\\\' for r in data_t]
    lines += ['\\hline', '\\end{tabular}', f'\\end{{{env}}}']
    return '\n```{=latex}\n' + '\n'.join(lines) + '\n```\n'


TBL = re.compile(r'\*\*Table ([0-9A]+)\.\*\*(.*?)\n\n((?:\|[^\n]*\|\n?)+)')


# ---------- figures ----------
def fig_block(m):
    cap, path = m.group(1), m.group(2)
    name = os.path.basename(path)
    cap_t = cells_to_tex([cap])[0]
    narrow = name in ('fig_gaia_importance.png', 'fig_kepler90_timing.png')
    env = 'figure' if narrow else 'figure*'
    w = '\\columnwidth' if narrow else ('0.92\\textwidth' if name in ('fig_pipeline.png', 'fig_views.png') else '\\textwidth')
    lab = 'fig:' + name[4:-4]
    return (f'\n```{{=latex}}\n\\begin{{{env}}}\n\\centering\n\\includegraphics[width={w}]{{figures/{name}}}\n'
            f'\\caption{{{cap_t}}}\n\\label{{{lab}}}\n\\end{{{env}}}\n```\n')


FIG = re.compile(r'!\[(.*?)\]\((\.\./figures/[^)]+)\)(\{[^}]*\})?')


def convert(md):
    md = TBL.sub(table_block, md)
    md = FIG.sub(fig_block, md)
    md = re.sub(r'^(#+) (?:[0-9]+(?:\.[0-9]+)*|Appendix [A-Z]:) ', r'\1 ', md, flags=re.M)   # LaTeX numbers sections
    md = re.sub(r'(?<![(<])(https?://[^\s)]+?)([.,;]?)(?=[\s)])', r'<\1>\2', md)            # bare URLs -> links
    tex = pandoc(md)
    tex = re.sub(r'(?:×)?10⁻([⁰-⁹⁴⁸³²¹]+)', lambda m: ('$\\times10^{-' if m.group(0)[0]=='×' else '$10^{-') + m.group(1).translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹','0123456789')) + '}$', tex)
    return tex


body_t = convert(body)
app_md = appendix.split('\n', 1)[1]
app_t = convert('# Further tables and figures\n' + app_md)

# section headings for back matter: MNRAS uses unnumbered \section*
for h in ('Acknowledgements', 'Software', 'Data availability'):
    body_t = body_t.replace(f'\\section{{{h}}}', f'\\section*{{{h}}}')
body_t = body_t.replace('\\section*{Acknowledgements}', '\\section*{Acknowledgements}')

# ---------- references ----------
items = []
for r in refs:
    r = r.lstrip('- ').strip()
    if not r:
        continue
    m = re.match(r'(.+?),\s*((?:19|20)\d\d)\b', r)
    auth, year = m.group(1), m.group(2)
    surnames = re.findall(r"([A-Z][a-zA-Z'\u00c0-\u017f]+(?: [A-Z][a-z]+)?)\s+(?:[A-Z]\.\s*)+", auth)
    if 'et al' in auth:
        short = f'{surnames[0]} et al.'
    elif 'Collaboration' in auth:
        short = 'Gaia Collaboration'
    elif len(surnames) == 2:
        short = f'{surnames[0]} \\& {surnames[1]}'
    elif surnames:
        short = surnames[0] + (' et al.' if len(surnames) > 2 else '')
    else:
        short = auth
    key = re.sub(r'[^a-z]', '', (surnames[0] if surnames else auth).lower()) + year
    txt = cells_to_tex([r])[0]
    items.append(f'\\bibitem[\\protect\\citeauthoryear{{{short}}}{{{year}}}]{{{key}}}\n{txt}')

title = re.search(r'title: "(.*?)"', s).group(1)
authors_line = re.search(r'author:\n  - "(.*?)"', s).group(1)

tex = r"""% TransitVet manuscript, MNRAS / RASTI layout.
% Built by build_tex.py from ../manuscript.md. Edit the Markdown and re-run, or edit this file directly on Overleaf.
% To submit to RAS Techniques and Instruments, open the RASTI template on Overleaf and paste the body of this file
% into it; the RASTI class follows the MNRAS commands used here [CHECK when the venue is fixed].
\documentclass[fleqn,usenatbib]{mnras}

\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{array}
\usepackage{url}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2009}{\,}
\DeclareUnicodeCharacter{00A0}{~}
\DeclareUnicodeCharacter{2299}{\ensuremath{\odot}}
\DeclareUnicodeCharacter{2295}{\ensuremath{\oplus}}
\DeclareUnicodeCharacter{00B5}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{00B3}{\ensuremath{^3}}
\DeclareUnicodeCharacter{2609}{\ensuremath{\odot}}

\title[TransitVet: calibrated vetting of Kepler and TESS signals]{<<TITLE>>}

% Author list and affiliation from the team. [CHECK affiliation wording, ORCIDs and the corresponding e-mail.]
\author[Girinath A et al.]{
Girinath A,$^{1}$\thanks{E-mail: corresponding author e-mail [CHECK]}
Akilan S,$^{1}$
Misham Samuel Daystar,$^{1}$
Sachin Sarlin K R S$^{1}$
and R. A. Rakhi$^{1}$
\\
$^{1}$Department of Electronics and Communication Engineering, Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu, India
}

\date{Accepted XXX. Received YYY; in original form ZZZ}
\pubyear{2026}

\begin{document}
\label{firstpage}
\pagerange{\pageref{firstpage}--\pageref{lastpage}}
\maketitle

\begin{abstract}
<<ABSTRACT>>
\end{abstract}

\begin{keywords}
<<KEYWORDS>>
\end{keywords}

<<BODY>>

\begin{thebibliography}{99}

<<REFS>>
\end{thebibliography}

\clearpage
\appendix
<<APPENDIX>>

\bsp
\label{lastpage}
\end{document}
"""
tex = (tex.replace('<<TITLE>>', title).replace('<<ABSTRACT>>', pandoc(abstract).strip())
       .replace('<<KEYWORDS>>', keywords).replace('<<BODY>>', body_t).replace('<<REFS>>', '\n\n'.join(items))
       .replace('<<APPENDIX>>', app_t))
open(os.path.join(HERE, 'transitvet.tex'), 'w', encoding='utf-8').write(tex)
print('wrote transitvet.tex', len(tex), 'chars;', len(items), 'references')
