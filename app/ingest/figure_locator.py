#!/usr/bin/env python3
"""
Locate figures in a PDF, generically.

This finds figures by looking for pages that contain embedded raster images —
which works for ANY document, regardless of whether figures are captioned
"Figure 1", "Fig.", "Exhibit", or nothing at all. It optionally grabs a nearby
caption line as context, but a caption is NOT required for a figure to be found.

Returns one entry per page that contains image(s):
    {figure, page_index, caption}
  - figure: a sequential index (1, 2, 3, ...) in page order — NOT parsed from the
    document, since not all figures are numbered.
  - page_index: 0-indexed page number.
  - caption: best-effort nearby caption text, or "" if none found.
"""

import re
import pymupdf

# Loose caption detector: matches "Figure 3:", "Fig. 3 -", "FIGURE 3.", "Exhibit 3:",
# "Chart 3 —", etc. Used only to attach optional context; a match is not required.
_CAPTION_RE = re.compile(
    r"(?:figure|fig\.?|exhibit|chart|plot|diagram|scheme|table)\s*"
    r"(\d+)?\s*[:.\-–—]?\s*([^\n]{0,160})",
    re.IGNORECASE,
)

# Minimum image dimension (px) to count as a "figure" rather than an icon, bullet,
# logo, or rule. Tune via find_figures(min_dim=...).
_DEFAULT_MIN_DIM = 64


def _page_images(page, min_dim: int):
    """Return the embedded raster images on a page that are larger than min_dim."""
    imgs = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            w = info[2]
            h = info[3]
        except Exception:
            w = h = 0
        if w >= min_dim and h >= min_dim:
            imgs.append(xref)
    return imgs


def _nearby_caption(page_text: str) -> str:
    """Best-effort: return the first caption-like line on the page, else ''."""
    m = _CAPTION_RE.search(page_text or "")
    if not m:
        return ""
    num = m.group(1)
    tail = (m.group(2) or "").strip()
    if num and tail:
        return f"Figure {num}: {tail}"
    if tail:
        return tail
    return ""


def find_figures(pdf_path: str, min_dim: int = _DEFAULT_MIN_DIM):
    """
    Return a list of figures found in the PDF, one entry per page that contains a
    sufficiently large embedded image.

    Args:
      min_dim: ignore images smaller than this many pixels on a side (filters out
               logos, icons, bullet glyphs). Lower it if real figures are being
               missed; raise it if logos/icons are being picked up as figures.

    Returns: [{figure, page_index, caption}], ordered by page.
    """
    doc = pymupdf.open(pdf_path)
    results = []
    fig_index = 0
    for i in range(doc.page_count):
        page = doc[i]
        imgs = _page_images(page, min_dim)
        if not imgs:
            continue
        fig_index += 1
        caption = _nearby_caption(page.get_text())
        results.append({
            "figure": fig_index,
            "page_index": i,
            "caption": caption,
        })
    return results


def count_embedded_images(pdf_path: str):
    """Return total number of raster images embedded in the PDF (all pages)."""
    doc = pymupdf.open(pdf_path)
    return sum(len(doc[i].get_images()) for i in range(doc.page_count))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python figure_locator.py <file.pdf>")
        raise SystemExit(1)
    figs = find_figures(sys.argv[1])
    for e in figs:
        cap = e["caption"][:70] if e["caption"] else "(no caption found)"
        print(f"Fig {e['figure']:>2}  p{e['page_index']+1:<3}  {cap}")
    print(f"\nFigures found (pages with images): {len(figs)}")
    print(f"Total embedded raster images: {count_embedded_images(sys.argv[1])}")
