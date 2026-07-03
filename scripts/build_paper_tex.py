#!/usr/bin/env python3
"""Convert ai_detection_at_scale.md to ICLR LaTeX and compile PDF."""
import re, os, shutil, subprocess

PROJECT_DIR = '/Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale'
SRC = os.path.join(PROJECT_DIR, 'ai_detection_at_scale.md')
STY_SRC = os.path.join(PROJECT_DIR, '..', 'linguistic-markers-paper', 'latex_build', 'iclr2024_conference.sty')
OUT_DIR = os.path.join(PROJECT_DIR, 'latex_build')
FIGURES_SRC = os.path.join(PROJECT_DIR, 'results', 'figures')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

# Copy .sty file
if os.path.exists(STY_SRC):
    shutil.copy2(STY_SRC, OUT_DIR)
    print(f"Copied {STY_SRC}")
else:
    print(f"Warning: {STY_SRC} not found - will rely on tectonic's bundled sty")

# Copy figures
if os.path.exists(FIGURES_SRC):
    for f in os.listdir(FIGURES_SRC):
        if f.endswith('.png') or f.endswith('.pdf'):
            shutil.copy2(os.path.join(FIGURES_SRC, f), os.path.join(OUT_DIR, 'figures'))
    print(f"Copied figures from {FIGURES_SRC}")
else:
    print(f"Warning: figures directory {FIGURES_SRC} not found")

with open(SRC, 'r', encoding='utf-8') as f:
    md = f.read()

# Extract abstract
abstract_match = re.search(r'## Abstract\n\n(.*?)\n\n_\*\*Keywords\*\*_', md, re.DOTALL)
if abstract_match:
    abstract = abstract_match.group(1).strip()
else:
    print("Warning: Abstract not found!")
    abstract = ""

# Extract body
body_match = re.search(r'(## 1\. Introduction.*?)## References', md, re.DOTALL)
if body_match:
    body = body_match.group(1).strip()
else:
    print("Warning: Body not found!")
    body = ""

# Extract references
refs_match = re.search(r'## References\n\n(.*)', md, re.DOTALL)
if refs_match:
    refs_text = refs_match.group(1).strip()
else:
    print("Warning: References not found!")
    refs_text = ""


def escape_latex_text(text):
    text = text.replace('\u2014', '---')
    text = text.replace('\u2013', '--')
    text = text.replace('\u2018', '`')
    text = text.replace('\u2019', "'")
    text = text.replace('\u201C', '``')
    text = text.replace('\u201D', "''")
    text = text.replace('\u00b1', r'\textpm{}')
    text = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', text)
    text = re.sub(r'(?<!\\)%', r'\\%', text)
    text = re.sub(r'(?<!\\)&', r'\\&', text)
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')
    text = text.replace('→', r'$\rightarrow$')
    text = text.replace('×', r'$\times$')
    def make_breakable(match):
        t = match.group(1)
        t = t.replace(',', ',\\allowbreak{}')
        t = t.replace('.', '.\\allowbreak{}')
        t = t.replace('\\_', '\\_\\allowbreak{}')
        t = t.replace(':', ':\\allowbreak{}')
        t = t.replace('|', '|\\allowbreak{}')
        return r'\texttt{' + t + '}'
    text = re.sub(r'`([^`]+)`', make_breakable, text)
    return text


def escape_latex_cell(text):
    text = text.replace('\u00b1', r'\textpm{}')
    text = re.sub(r'\$(\d)', r'\\$\1', text)
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    text = text.replace('&', '\\&')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('_', '\\_')
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    return text


def convert_citations(text):
    def cite_replace(m):
        inner = m.group(1)
        parts = [p.strip() for p in re.split(r'[,;]\s*', inner)]
        if all(p.isdigit() for p in parts):
            refs = ','.join(f'ref{p}' for p in parts)
            return f'\\cite{{{refs}}}'
        return m.group(0)
    return re.sub(r'\[(\d+(?:\s*[,;]\s*\d+)*)\]', cite_replace, text)


def convert_inline_formatting(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\\textit{\1}', text)
    text = re.sub(r'_(.+?)_', r'\\textit{\1}', text)
    return text


def convert_quotes(text):
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    return text


def process_paragraph(text):
    text = escape_latex_text(text)
    text = convert_inline_formatting(text)
    text = convert_citations(text)
    text = convert_quotes(text)
    return text


def render_table(rows, caption=None):
    header_cells = [c.strip() for c in rows[0].split('|')[1:-1]]
    ncols = len(header_cells)
    data_rows = []
    for row in rows[2:]:
        cells = [c.strip() for c in row.split('|')[1:-1]]
        while len(cells) < ncols:
            cells.append('')
        data_rows.append(cells[:ncols])

    # Use l for first col, c for rest
    col_spec = 'l' + 'c' * (ncols - 1)

    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    h_cells = [f'\\textbf{{{escape_latex_cell(c)}}}' for c in header_cells]
    lines.append(' & '.join(h_cells) + ' \\\\')
    lines.append('\\midrule')
    for cells in data_rows:
        d_cells = [escape_latex_cell(c) for c in cells]
        lines.append(' & '.join(d_cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    if caption:
        cap_escaped = escape_latex_text(caption)
        lines.append(f'\\caption{{{cap_escaped}}}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


def md_to_latex(text):
    lines = text.split('\n')
    output_blocks = []
    current_para = []
    in_table = False
    table_rows = []
    table_caption = None

    def flush_para():
        nonlocal current_para
        if current_para:
            para_text = '\n'.join(current_para)
            stripped = para_text.strip()
            if stripped.startswith('- ') or stripped.startswith('* '):
                items = re.split(r'\n[-*]\s+', stripped)
                list_lines = ['\\begin{itemize}']
                for item in items:
                    item_clean = re.sub(r'^[-*]\s+', '', item.strip()).strip()
                    if item_clean:
                        list_lines.append(f'  \\item {process_paragraph(item_clean)}')
                list_lines.append('\\end{itemize}')
                output_blocks.append('\n'.join(list_lines))
            elif re.match(r'^\d+\.\s', stripped):
                items = re.split(r'\n\d+\.\s+', stripped)
                list_lines = ['\\begin{enumerate}']
                for item in items:
                    item_clean = item.strip().lstrip('1234567890.').strip()
                    if item_clean:
                        list_lines.append(f'  \\item {process_paragraph(item_clean)}')
                list_lines.append('\\end{enumerate}')
                output_blocks.append('\n'.join(list_lines))
            else:
                rendered = process_paragraph(para_text)
                output_blocks.append(rendered)
            current_para = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            if in_table:
                in_table = False
                output_blocks.append(render_table(table_rows, table_caption))
                table_caption = None
                table_rows = []
            else:
                flush_para()
            output_blocks.append('')
            i += 1
            continue

        # Section headings
        if re.match(r'^## [^#]', line) and 'References' not in line:
            flush_para()
            title = line[3:].strip()
            title = re.sub(r'^([IVXLC]+|\d+)\.\s+', '', title)
            title = escape_latex_text(title)
            output_blocks.append(f'\\section{{{title}}}')
            i += 1
            continue

        if re.match(r'^### [^#]', line):
            flush_para()
            title = line[4:].strip()
            title = re.sub(r'^([A-Z]\.|\d+\.\d+)\s+', '', title)
            title = escape_latex_text(title)
            output_blocks.append(f'\\subsection{{{title}}}')
            i += 1
            continue

        # Table caption line (bold starting with Table)
        if re.match(r'^\*\*Table \d+', line):
            flush_para()
            table_caption = re.sub(r'^\*\*(.+)\*\*$', r'\1', line.strip())
            table_caption = re.sub(r'^Table \d+\.\s*', '', table_caption)
            i += 1
            continue

        # Table rows
        if line.strip().startswith('|') and '|' in line[1:]:
            if not in_table:
                flush_para()
                in_table = True
                table_rows = []
            table_rows.append(line)
            i += 1
            continue

        if in_table:
            in_table = False
            output_blocks.append(render_table(table_rows, table_caption))
            table_caption = None
            table_rows = []
            continue

        # Figure
        fig_match = re.match(r'^!\[(.+?)\]\((.+?)\)$', line.strip())
        if fig_match:
            flush_para()
            caption = fig_match.group(1)
            img_path = fig_match.group(2)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            src_img = os.path.join(PROJECT_DIR, img_path)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(OUT_DIR, 'figures'))
            caption_tex = escape_latex_text(caption)
            fig_tex = (
                '\\begin{figure}[ht]\n'
                '\\centering\n'
                '\\vspace{-0.5em}\n'
                f'\\includegraphics[width=0.82\\textwidth,clip]{{figures/{img_basename}}}\n'
                '\\vspace{-0.5em}\n'
                f'\\caption{{{caption_tex}}}\n'
                '\\end{figure}'
            )
            output_blocks.append(fig_tex)
            i += 1
            continue

        current_para.append(line)
        i += 1

    if in_table:
        output_blocks.append(render_table(table_rows, table_caption))
    flush_para()

    return '\n'.join(output_blocks)


def build_bibliography(refs_text):
    entries = re.split(r'\n\n+', refs_text.strip())
    bib_items = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        match = re.match(r'^\[(\d+)\]\s*(.*)', entry, re.DOTALL)
        if match:
            num = match.group(1)
            text = match.group(2).strip()
            # Strip markdown link syntax - extract text inside [text](url)
            # e.g. [Author (2020). "Title." *Venue*. URL](https://...)
            link_match = re.match(r'^\[(.+?)\]\((.+?)\)$', text, re.DOTALL)
            if link_match:
                text = link_match.group(1).strip()
                url = link_match.group(2).strip()
            else:
                # Extract URL from end of line
                url_match = re.search(r'(https?://\S+)$', text)
                url = url_match.group(1) if url_match else ''
                if url:
                    text = text[:text.rfind(url)].strip()

            text = text.replace('&', '\\&')
            text = text.replace('%', '\\%')
            text = re.sub(r'(?<!\\)\$', r'\\$', text)
            text = text.replace('_', '\\_')
            text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)

            if url:
                raw_url = url
                display = re.sub(r'^https?://(www\.)?', '', url).rstrip('/')
                display_escaped = (display.replace('_', '\\_')
                                   .replace('&', '\\&')
                                   .replace('%', '\\%')
                                   .replace('#', '\\#'))
                text += f'. \\href{{{raw_url}}}{{{display_escaped}}}'

            bib_items.append(f'\\bibitem{{ref{num}}}\n{text}\n')

    return ('\\small\n\\begin{thebibliography}{22}\n\\raggedright\n\n'
            + '\n'.join(bib_items)
            + '\n\\end{thebibliography}')


abstract_tex = process_paragraph(abstract)
body_tex = md_to_latex(body)
bib_tex = build_bibliography(refs_text)

PAPER_TITLE = 'Domain-Invariant vs Register-Dependent Stylometric Features for AI Text Detection: A Benchmark Study at Web Scale'
PAPER_TITLE_SHORT = 'Domain-Invariant Stylometric Features for AI Text Detection'

template = r"""\documentclass{article}
\usepackage{iclr2024_conference,times}
\usepackage{amsmath}
\usepackage{textcomp}
\usepackage[htt]{hyphenat}

\tolerance=1500
\emergencystretch=2em
\hyphenpenalty=10000
\exhyphenpenalty=10000
\frenchspacing
\sloppy
\widowpenalty=150
\clubpenalty=150
\predisplaypenalty=0

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black]{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=it}
\captionsetup[table]{skip=6pt}
\captionsetup[figure]{skip=6pt}

\iclrfinalcopy
\setcitestyle{numbers,square}

\title{Domain-Invariant vs Register-Dependent\\Stylometric Features for AI Text Detection:\\A Benchmark Study at Web Scale}

\author{Vedang Ratan Vatsa \\
\href{https://veda.ng}{veda.ng} \textperiodcentered{} \href{mailto:vedangvats@gmail.com}{vedangvats@gmail.com}}

\begin{document}
\hypersetup{
  pdftitle={""" + PAPER_TITLE + r"""},
  pdfauthor={Vedang Ratan Vatsa},
  pdfsubject={Cross-domain AI text detection benchmark study},
  pdfkeywords={AI text detection, stylometrics, domain generalization, RAID benchmark, lexical diversity, register analysis},
  pdfcreator={Vedang Ratan Vatsa},
  pdfproducer={Vedang Ratan Vatsa}
}
\raggedbottom

\maketitle

\begin{abstract}
""" + abstract_tex + r"""
\end{abstract}

""" + body_tex + r"""

\clearpage
""" + bib_tex + r"""

\end{document}
"""

out_path = os.path.join(OUT_DIR, 'paper.tex')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(template)
print(f"Written {out_path} ({len(template)} chars)")

# Compile using tectonic
try:
    print("Compiling LaTeX to PDF using tectonic...")
    res = subprocess.run(['tectonic', 'paper.tex'], cwd=OUT_DIR,
                         check=True, capture_output=True, text=True)
    print("Compilation successful.")
    pdf_src = os.path.join(OUT_DIR, 'paper.pdf')
    pdf_dest = os.path.join(PROJECT_DIR, 'ai_detection_at_scale.pdf')
    shutil.copy2(pdf_src, pdf_dest)
    print(f"Copied PDF to {pdf_dest}")
    desktop = '/Users/vedang/Desktop'
    if os.path.exists(desktop):
        shutil.copy2(pdf_src, os.path.join(desktop, 'ai_detection_at_scale.pdf'))
        print("Copied PDF to Desktop")
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: exit code {e.returncode}")
    print("Stdout:", e.stdout[-3000:] if e.stdout else '')
    print("Stderr:", e.stderr[-3000:] if e.stderr else '')
except Exception as ex:
    print(f"Compilation error: {str(ex)}")
