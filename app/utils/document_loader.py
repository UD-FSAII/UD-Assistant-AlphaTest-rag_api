# app/utils/document_loader.py

import os
import codecs
import tempfile

from typing import Iterator, List, Optional
import chardet

from langchain_core.documents import Document

from app.config import known_source_ext, PDF_EXTRACT_IMAGES, CHUNK_OVERLAP, logger
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)



import io
import urllib.request
import urllib.error
import json as _json
import base64 as _base64

# --- OCR fallback configuration (custom fork) ---
OCR_FALLBACK_ENABLED = os.getenv("OCR_FALLBACK_ENABLED", "true").lower() == "true"
OCR_FALLBACK_URL = os.getenv("OCR_FALLBACK_URL", "http://172.22.0.1:8003/v1/ocr")
OCR_MIN_CHARS = int(os.getenv("OCR_FALLBACK_MIN_CHARS", "20"))
OCR_TIMEOUT = int(os.getenv("OCR_FALLBACK_TIMEOUT", "120"))

# --- Structure-aware markdown extraction (custom fork) ---
# When enabled, PDFs are first run through the table-aware extractor
# (app/ingest/pdf_extract.py) which produces clean Markdown with pipe tables kept
# whole. This runs BEFORE the OCR fallback: text-bearing PDFs get structure-aware
# markdown; scanned/image-only PDFs (near-empty result) fall through to OCR.
# Figures (vision-model reading) are OFF by default here — they're slow and would
# block the upload request; enable per-deployment only if you accept the latency.
STRUCTURED_PDF_ENABLED = os.getenv("STRUCTURED_PDF_ENABLED", "true").lower() == "true"
STRUCTURED_PDF_FIGURES = os.getenv("STRUCTURED_PDF_FIGURES", "false").lower() == "true"
STRUCTURED_PDF_MIN_CHARS = int(os.getenv("STRUCTURED_PDF_MIN_CHARS", "40"))


def _run_structured_pdf(filepath: str):
    """Run the table-aware extractor; return clean markdown or "" on failure/empty.

    Returns "" (so the caller falls through to OCR) when the PDF has no extractable
    text — i.e. a scanned/image-only PDF — or if the extractor errors for any reason.
    """
    try:
        from app.ingest.pdf_extract import extract_pdf_to_markdown
    except Exception as e:  # module not present / import error -> disable silently
        logger.warning(f"Structured PDF extractor unavailable: {e}")
        return ""
    try:
        md = extract_pdf_to_markdown(filepath, include_figures=STRUCTURED_PDF_FIGURES)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Structured PDF extraction failed for {filepath}: {e}")
        return ""
    if len((md or "").strip()) < STRUCTURED_PDF_MIN_CHARS:
        logger.info(
            f"Structured extraction yielded {len((md or '').strip())} chars for "
            f"{filepath}; treating as scanned -> OCR fallback."
        )
        return ""
    logger.info(f"Structured PDF extraction produced {len(md)} chars for {filepath}.")
    return md


def _run_ocr_fallback(filepath: str) -> str:
    """POST the PDF to the Mistral-OCR-compatible service; return page text or ""."""
    try:
        with open(filepath, "rb") as fh:
            raw = fh.read()
        b64 = _base64.b64encode(raw).decode("ascii")
        payload = _json.dumps({
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64," + b64,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            OCR_FALLBACK_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=OCR_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        pages = data.get("pages", []) or []
        texts = [ (p.get("markdown") or "").strip() for p in pages ]
        texts = [ t for t in texts if t ]
        result = "\n\n".join(texts).strip()
        logger.info(f"OCR fallback produced {len(result)} chars from {len(pages)} page(s) for {filepath}")
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OCR fallback failed for {filepath}: {e}")
        return ""


# Extensions that identify binary file formats handled by dedicated loaders.
# Used to prevent a conflicting multipart Content-Type (e.g. ``text/markdown``)
# from hijacking these files into a text loader.
_BINARY_FILE_EXTENSIONS = frozenset(
    {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "epub"}
)


def detect_file_encoding(filepath: str) -> str:
    """
    Detect the encoding of a file using BOM markers and chardet for broader support.
    Returns the detected encoding or 'utf-8' as default.
    """
    with open(filepath, "rb") as f:
        raw = f.read(4096)  # Read a larger sample for better detection

    # Check for BOM markers first
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    elif raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    elif raw.startswith(codecs.BOM_UTF16):
        return "utf-16"
    elif raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    elif raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    elif raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"

    # Use chardet to detect encoding if no BOM is found
    result = chardet.detect(raw)
    encoding = result.get("encoding")
    if encoding:
        return encoding.lower()
    # Default to utf-8 if detection fails
    return "utf-8"


def cleanup_temp_encoding_file(loader) -> None:
    """
    Clean up temporary UTF-8 file if it was created for encoding conversion.

    :param loader: The document loader that may have created a temporary file
    """
    if hasattr(loader, "_temp_filepath") and loader._temp_filepath is not None:
        try:
            os.remove(loader._temp_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary UTF-8 file: {e}")


def get_loader(
    filename: str,
    file_content_type: str,
    filepath: str,
    raw_text: bool = False,
):
    """Get the appropriate document loader based on file type and\or content type.

    When ``raw_text`` is True, text-formatted files (e.g. Markdown) are loaded
    verbatim with :class:`TextLoader` so their original formatting is
    preserved. This is intended for the ``/text`` endpoint, where the caller
    wants the raw file contents. The embedding path should keep the default
    (``raw_text=False``) so semantic loaders continue to strip formatting for
    better vector search quality.
    """
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # File Content Type reference:
    # ref.: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    if file_ext == "pdf" or file_content_type == "application/pdf":
        loader = SafePyPDFLoader(filepath, extract_images=PDF_EXTRACT_IMAGES)
    elif file_ext == "csv" or file_content_type == "text/csv":
        # Detect encoding for CSV files
        encoding = detect_file_encoding(filepath)

        if encoding != "utf-8":
            # For non-UTF-8 encodings, convert to UTF-8 using streaming
            # to avoid holding the entire file in memory as a single string
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".csv", delete=False
                ) as temp_file:
                    with open(
                        filepath, "r", encoding=encoding, errors="replace"
                    ) as original_file:
                        while True:
                            chunk = original_file.read(64 * 1024)
                            if not chunk:
                                break
                            temp_file.write(chunk)

                    temp_filepath = temp_file.name

                loader = CSVLoader(temp_filepath)
                loader._temp_filepath = temp_filepath
            except Exception as e:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml" or file_content_type in [
        "application/xml",
        "text/xml",
        "application/xhtml+xml",
    ]:
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext in ["ppt", "pptx"] or file_content_type in [
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]:
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_ext == "md" or (
        file_content_type
        in [
            "text/markdown",
            "text/x-markdown",
            "application/markdown",
            "application/x-markdown",
        ]
        and file_ext not in _BINARY_FILE_EXTENSIONS
    ):
        if raw_text:
            loader = TextLoader(filepath, autodetect_encoding=True)
        else:
            loader = UnstructuredMarkdownLoader(filepath)
    elif file_ext == "epub" or file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(filepath)
    elif file_ext in ["doc", "docx"] or file_content_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        loader = Docx2txtLoader(filepath)
    elif file_ext in ["xls", "xlsx"] or file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        loader = UnstructuredExcelLoader(filepath)
    elif file_ext == "json" or file_content_type == "application/json":
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False

    return loader, known_type, file_ext


def clean_text(text: str) -> str:
    """
    Clean up text from PDF lopader

    :param text: The original text
    :return: Cleaned text
    """
    text = remove_null(text)
    text = remove_non_utf8(text)
    return text


def remove_null(text: str) -> str:
    """
    Remove NUL (0x00) characters from a string.

    :param text: The original text with potential NUL characters.
    :return: Cleaned text without NUL characters.
    """
    return text.replace("\x00", "")


def remove_non_utf8(text: str) -> str:
    """
    Remove invalid UTF-8 characters from a string, such as surrogate characters

    :param text: The original text with potential invalid utf-8 characters
    :return: Cleaned text without invalid utf-8 characters.
    """
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except UnicodeError:
        return text


def process_documents(documents: List[Document]) -> str:
    processed_text = ""
    last_page: Optional[int] = None
    doc_basename = ""

    for doc in documents:
        if "source" in doc.metadata:
            doc_basename = doc.metadata["source"].split("/")[-1]
            break

    processed_text += f"{doc_basename}\n"

    for doc in documents:
        current_page = doc.metadata.get("page")
        if current_page and current_page != last_page:
            processed_text += f"\n# PAGE {doc.metadata['page']}\n\n"
            last_page = current_page

        new_content = doc.page_content
        if processed_text.endswith(new_content[:CHUNK_OVERLAP]):
            processed_text += new_content[CHUNK_OVERLAP:]
        else:
            processed_text += new_content

    return processed_text.strip()


class SafePyPDFLoader:
    """
    A wrapper around PyPDFLoader that handles image extraction failures gracefully.
    Falls back to text-only extraction when image extraction fails.

    This is a workaround for issues with PyPDFLoader that can occur when extracting images
    from PDFs, which can lead to KeyError exceptions if the PDF is malformed or has unsupported
    image formats. This class attempts to load the PDF with image extraction enabled, and if it
    fails due to a KeyError related to image filters, it falls back to loading the PDF
    without image extraction.
    ref.: https://github.com/langchain-ai/langchain/issues/26652
    """

    def __init__(self, filepath: str, extract_images: bool = False):
        self.filepath = filepath
        self.extract_images = extract_images
        self._temp_filepath = None  # For compatibility with cleanup function

    def _raw_pages(self) -> List[Document]:
        """Run PyPDF extraction with the existing image-extraction KeyError fallback."""
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)
        if not self.extract_images:
            return list(loader.lazy_load())
        try:
            return list(loader.lazy_load())
        except KeyError as e:
            if "/Filter" in str(e):
                logger.warning(
                    f"PDF image extraction failed for {self.filepath}, falling back to text-only: {e}"
                )
                fallback_loader = PyPDFLoader(self.filepath, extract_images=False)
                return list(fallback_loader.lazy_load())
            raise

    def _pages_with_ocr(self) -> List[Document]:
        """Return extracted pages, in tiers:

        1. Structure-aware Markdown extraction (tables kept whole) — for text-bearing
           PDFs. This is the custom fork's preferred path.
        2. OCR fallback — for scanned/image-only PDFs (structure + PyPDF both empty).
        3. Raw PyPDF pages — final fallback.

        Applies to BOTH lazy_load() (embed path) and load().
        """
        # Tier 1: structure-aware markdown (tables preserved). Returns "" for
        # scanned/image-only PDFs so we fall through to OCR.
        if STRUCTURED_PDF_ENABLED:
            md = _run_structured_pdf(self.filepath)
            if md:
                return [Document(
                    page_content=md,
                    metadata={"source": self.filepath, "extractor": "structured_md"},
                )]

        # Tier 2/3: existing behavior — raw PyPDF, with OCR fallback if near-empty.
        pages = self._raw_pages()
        if OCR_FALLBACK_ENABLED:
            total_chars = sum(len((p.page_content or "").strip()) for p in pages)
            if total_chars < OCR_MIN_CHARS:
                logger.info(
                    f"PDF {self.filepath} yielded only {total_chars} chars; attempting OCR fallback."
                )
                ocr_text = _run_ocr_fallback(self.filepath)
                if ocr_text and len(ocr_text.strip()) >= OCR_MIN_CHARS:
                    return [Document(page_content=ocr_text, metadata={"source": self.filepath})]
                logger.warning(
                    f"OCR fallback produced insufficient text for {self.filepath}; returning original extraction."
                )
        return pages

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load PDF documents (with image-error fallback AND OCR fallback)."""
        yield from self._pages_with_ocr()

    def load(self) -> List[Document]:
        """Load PDF documents (with image-error fallback AND OCR fallback)."""
        return self._pages_with_ocr()