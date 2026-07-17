# rag_api — UD Assistant fork

Fork of [`danny-avila/rag_api`](https://github.com/danny-avila/rag_api) customized for
UD Assistant. This is the RAG / File Search backend for LibreChat.

## Our changes vs upstream
- **OCR fallback** — when a scanned/image PDF has no extractable text, rag_api calls a
  local OCR service (Tesseract, presented as an OpenAI-style "Mistral" endpoint) so scanned
  documents become searchable. See root README §13.
- **`app/ingest/`** — custom ingestion helpers (`pdf_extract.py`, `figure_extract.py`,
  `figure_locator.py`) for figure/table-aware extraction.
- **`Dockerfile` + `requirements.txt`** — adjusted for the above.

## Deploy
Runs as a container on the LibreChat Docker network; LibreChat's `.env` points at it via
`RAG_API_URL`. Embeddings can run in-process (`EMBEDDINGS_PROVIDER=huggingface`) to save VRAM.

## Config
- `.env` (gitignored) — DB connection, embedding provider, OCR endpoint. See upstream's
  `.env.example` plus the OCR variables documented in root README §13.

## Upstream / updates
Pinned to the commit we forked from + our changes. `upstream` remote points at
`danny-avila/rag_api`. Pull upstream fixes manually; we don't auto-track.

**Full detail:** parent project root `README.md` §13 (document File Search & OCR pipeline).
