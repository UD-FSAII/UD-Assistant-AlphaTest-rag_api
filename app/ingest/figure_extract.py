#!/usr/bin/env python3
"""
Read FIGURES from a PDF using the vision-capable vLLM model (Qwen3-VL), generically.

Some information in documents exists only inside figures (diagrams, charts,
screenshots, annotated images). A vision model reads a rendered page and describes /
transcribes the figure. This version is DOMAIN-AGNOSTIC: one generic prompt, no
subject-specific classification.

Design:
  - Uses the SAME OpenAI-compatible vLLM endpoint and model as the text pipeline
    (Qwen3-VL is vision-capable). Set VLLM_MODEL (default Qwen/Qwen3-VL-30B-A3B-Instruct-FP8).
  - Renders each figure's page to a PNG at FIGURE_DPI and asks the model to describe
    the figure and transcribe any data/text printed in it.
  - Bounded per call (timeout, max tokens, retries) and post-processed to catch
    repetition loops and degenerate/hallucinated output.
"""

import os
import re
import base64

import pymupdf
from openai import OpenAI

from figure_locator import find_figures

RENDER_DPI = int(os.getenv("FIGURE_DPI", "160"))
FIGURE_TIMEOUT = float(os.getenv("FIGURE_TIMEOUT", "120"))   # seconds per call
FIGURE_MAX_TOKENS = int(os.getenv("FIGURE_MAX_TOKENS", "2500"))
FIGURE_RETRIES = int(os.getenv("FIGURE_RETRIES", "2"))
# Mild penalty: enough to nudge off repetition loops without pushing the model into
# hallucinated content.
FIGURE_FREQUENCY_PENALTY = float(os.getenv("FIGURE_FREQUENCY_PENALTY", "0.1"))
# Collapse a line that repeats consecutively more than this many times.
FIGURE_MAX_REPEAT = int(os.getenv("FIGURE_MAX_REPEAT", "2"))


def render_page_png(pdf_path: str, page_index: int, dpi: int = RENDER_DPI) -> bytes:
    """Render a single 0-indexed page to PNG bytes at the given DPI."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def _b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


# Signatures of degenerate / hallucinated VLM output on dense figures.
_GARBAGE_PATTERNS = [
    re.compile(r"\d{12,}"),                      # absurdly long digit run
    re.compile(r"\.\d{10,}"),                    # absurd decimal tail
    re.compile(r"(Z\?\?){3,}"),                  # Z??Z??Z?? runs
    re.compile(r"\?{6,}"),                       # ?????? runs
    re.compile(r"(\?\s*){8,}"),                  # spaced ? ? ? ? runs
    re.compile(r"(<br>\s*\*\*\?\*\*\s*){3,}"),   # **?**<br>**?** runs
    re.compile(r"(<sub>\?){3,}"),                # nested <sub>? runs
    re.compile(r"unintelligible", re.I),
]


def _looks_like_garbage(text: str) -> bool:
    """True if the transcription shows signs of VLM degeneration/hallucination."""
    return any(pat.search(text) for pat in _GARBAGE_PATTERNS)


def _collapse_repeats(text: str, max_repeat: int = FIGURE_MAX_REPEAT) -> str:
    """
    Salvage output where the VLM fell into a repetition loop (same line emitted many
    times until max_tokens). Keeps the first `max_repeat` occurrences of any
    consecutively-repeated line, drops the rest, appends a visible note, and trims a
    dangling final line cut off mid-token by the token cap.
    """
    lines = text.split("\n")
    out, prev, run, collapsed = [], None, 0, False
    for ln in lines:
        key = ln.strip()
        if key and key == prev:
            run += 1
            if run > max_repeat:
                collapsed = True
                continue
        else:
            prev = key
            run = 1
        out.append(ln)
    while out and (out[-1].rstrip().endswith("\\") or out[-1].strip() in ("|", "")):
        out.pop()
    result = "\n".join(out).rstrip()
    if collapsed:
        result += ("\n\n_[repeated lines collapsed; content may be incomplete "
                   "due to a model repetition loop on this dense figure]_")
    return result


def _sanitize_figure_output(text: str) -> str:
    """
    Post-process one figure reading: collapse repetition loops; if garbage signatures
    remain, drop the transcription and replace with a 'verify manually' note so
    fabricated content never lands silently in the output.
    """
    cleaned = _collapse_repeats(text)
    if _looks_like_garbage(cleaned):
        return ("_[figure transcription unreliable — the model produced "
                "degenerate/implausible output on this figure; verify against the "
                "PDF manually]_")
    return cleaned


# One generic, domain-agnostic prompt for any figure.
FIGURE_PROMPT = (
    "This is a page from a document that contains a figure (a chart, diagram, plot, "
    "screenshot, illustration, or annotated image).\n\n"
    "Describe the figure clearly and completely:\n"
    "- Say what the figure shows and its purpose.\n"
    "- If it contains data printed as text (numbers, labels, axis values, table-like "
    "content), transcribe those values into a clean Markdown table or list, copying "
    "them verbatim.\n"
    "- If it is a chart/plot, describe the axes (quantities, units, ranges) and any "
    "notable features (trends, peaks, comparisons), but do NOT invent exact numeric "
    "values you cannot read off the figure.\n"
    "- Mark anything you cannot read clearly as '?'.\n"
    "- Do not add commentary beyond describing the figure."
)


def read_figure(client: OpenAI, model: str, png_bytes: bytes,
                instruction: str = FIGURE_PROMPT) -> str:
    """
    Send one page image + instruction to the VLM and return its text reply.
    Bounded by a per-call timeout, capped output, frequency penalty, repetition
    collapse, and retries (after which the figure is skipped with a note).
    """
    data_uri = f"data:image/png;base64,{_b64(png_bytes)}"
    last_err = None
    for _ in range(FIGURE_RETRIES + 1):
        try:
            resp = client.with_options(timeout=FIGURE_TIMEOUT).chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }],
                temperature=0,
                max_tokens=FIGURE_MAX_TOKENS,
                frequency_penalty=FIGURE_FREQUENCY_PENALTY,
            )
            return _sanitize_figure_output(resp.choices[0].message.content or "")
        except Exception as e:  # timeout, connection reset, etc.
            last_err = e
    return f"_[figure skipped after {FIGURE_RETRIES + 1} attempts: {last_err}]_"


def extract_all_figures(pdf_path: str, client=None, model=None):
    """
    Find figures in the PDF and read each with the VLM.
    Yields dicts: {figure, page_index, caption, transcription}.
    """
    if client is None:
        client = OpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )
    if model is None:
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")

    for f in find_figures(pdf_path):
        png = render_page_png(pdf_path, f["page_index"])
        text = read_figure(client, model, png)
        yield {
            "figure": f["figure"],
            "page_index": f["page_index"],
            "caption": f["caption"],
            "transcription": text,
        }


def figures_markdown(pdf_path: str, client=None, model=None) -> str:
    """Render all figure readings into a single Markdown block for output."""
    parts = []
    for r in extract_all_figures(pdf_path, client=client, model=model):
        cap = f"*Caption:* {r['caption']}\n\n" if r["caption"] else ""
        parts.append(
            f"### Figure {r['figure']} (page {r['page_index']+1})\n"
            f"{cap}{r['transcription'].strip()}\n"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python figure_extract.py <file.pdf>")
        raise SystemExit(1)
    print(figures_markdown(sys.argv[1]))
