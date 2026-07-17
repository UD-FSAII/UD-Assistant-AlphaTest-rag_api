#!/usr/bin/env python3
"""
Local PDF -> Markdown extraction, tuned for scientific papers with ruled data tables.

Why this exists:
  A general Markdown converter (pymupdf4llm) silently DROPPED every numbered data
  table in the target papers, leaving only prose. Since all the target numbers live
  in those tables, the LLM had nothing to extract from. This module instead uses
  pdfplumber's ruled-table detection to recover the tables as clean Markdown pipe
  tables, and pairs each table with its caption for context.

Output per PDF:
  - Page prose (for surrounding context), followed by
  - Each detected table rendered as a labeled Markdown pipe table.

Continuation rows (e.g. hyperfine F multiplets, where the State cell is blank on the
second row) are preserved as empty leading cells so the LLM can tell the term-symbol
row (which carries J) apart from its F sub-rows.
"""

import re
import pdfplumber


def _clean_cell(c) -> str:
    if c is None:
        return ""
    # pdfplumber puts subscripts/superscripts on their own lines; rejoin to one line.
    s = str(c).replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _table_to_markdown(rows) -> str:
    rows = [[_clean_cell(c) for c in r] for r in rows if r is not None]
    rows = [r for r in rows if any(cell for cell in r)]  # drop fully-empty rows
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * ncol) + "|",
    ]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _find_captions(page_text: str):
    """Return {table_number: caption_text} for 'Table N:' captions on this page."""
    caps = {}
    for m in re.finditer(r"Table\s+(\d+)\s*[:.]\s*([^\n]{0,120})", page_text):
        num = int(m.group(1))
        caps.setdefault(num, "Table %d: %s" % (num, m.group(2).strip()))
    return caps


def extract_pdf_to_markdown(pdf_path: str, include_figures: bool = False,
                            figure_client=None, figure_model=None) -> str:
    """
    Convert a PDF to Markdown: prose + clean pipe tables (with captions).

    Args:
      include_figures: if True, append a "## Figures" section containing the
        vision-model description/transcription of each figure found in the PDF.
        NOTE: requires a running vLLM vision endpoint; the figure machinery is
        imported lazily so text-only callers never need a model.
      figure_client / figure_model: optional overrides passed through to
        figure_extract (else taken from environment).
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""

            # Prose first (gives the model context around each table).
            if page_text.strip():
                parts.append(page_text.strip())

            # Then any ruled tables on this page.
            caps = _find_captions(page_text)
            cap_nums = sorted(caps.keys())

            tables = page.extract_tables()
            for idx, tbl in enumerate(tables):
                md = _table_to_markdown(tbl)
                if not md:
                    continue
                # Best-effort caption: match Nth table on page to Nth caption number.
                label = ""
                if idx < len(cap_nums):
                    label = caps[cap_nums[idx]]
                elif cap_nums:
                    label = caps[cap_nums[-1]]
                if label:
                    parts.append("\n**%s**\n\n%s" % (label, md))
                else:
                    parts.append("\n%s" % md)

    body = _clean_markdown("\n\n".join(parts))

    if include_figures:
        figures_md = _figures_section(pdf_path, client=figure_client,
                                      model=figure_model)
        if figures_md:
            body = body + "\n\n" + figures_md

    return body


def _figures_section(pdf_path: str, client=None, model=None) -> str:
    """
    Build a '## Figures' markdown section from the vision-model readings.
    Imported lazily so text-only extraction never requires the model stack.
    Returns "" if figure extraction fails (e.g. no server) rather than raising,
    so a missing endpoint degrades to text-only instead of breaking the pipeline.
    """
    try:
        from figure_extract import figures_markdown
    except Exception:
        return ""
    try:
        fig_md = figures_markdown(pdf_path, client=client, model=model)
    except Exception as e:
        return "## Figures\n\n_Figure extraction unavailable: %s_" % e
    if not fig_md.strip():
        return ""
    return "## Figures\n\n" + fig_md.strip()


def _clean_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = "\n".join(line.rstrip() for line in md.split("\n"))
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python pdf_extract.py <file.pdf> [--figures]")
        print("  --figures    also read figures (via vision model) and append them")
        raise SystemExit(0 if args else 1)
    pdf = args[0]
    include = "--figures" in args
    print(extract_pdf_to_markdown(pdf, include_figures=include))
