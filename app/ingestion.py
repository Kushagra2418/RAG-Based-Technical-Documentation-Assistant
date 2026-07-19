

"""Document ingestion pipeline.

Turns raw documents into indexed, searchable chunks in three steps:

1. **Load** documents from local files (``.md``, ``.txt``, ``.html``) or from URLs.
2. **Chunk** them with a structure-aware strategy (see :func:`chunk_documents`).
3. **Index** the chunks into the vector store (embed + persist).

Chunking strategy
-----------------
Technical documentation is hierarchical, so a naive fixed-size split can cut a code block
in half or strip a paragraph from the heading that gives it meaning. This pipeline
therefore uses a two-stage strategy for Markdown:

* First, :class:`MarkdownHeaderTextSplitter` splits on ``#``/``##``/``###`` headers and
  records the heading path in each chunk's metadata (the ``section`` field). This keeps
  each chunk within a single logical section and preserves the context it belongs to.
* Then, :class:`RecursiveCharacterTextSplitter` enforces the configured ``chunk_size`` on
  any section that is still too large, splitting on paragraph and sentence boundaries
  first so splits fall in natural places.

Non-Markdown text uses the recursive splitter directly. ``chunk_size`` (default 1000) and
``chunk_overlap`` (default 150, ~15%) are configurable in :mod:`app.config`; the overlap
ensures a sentence spanning a boundary still appears intact in at least one chunk.
"""

from __future__ import annotations

import os
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings
from app.vectorstore import add_documents

# Header levels captured by the Markdown splitter, mapped to metadata keys.
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_file(path: str) -> Document:
    """Load a single local file into a Document, extracting text from HTML/PDF."""
    if path.lower().endswith(".pdf"):
        text = _extract_pdf_text(path)
        return Document(page_content=text, metadata={"source": os.path.basename(path)})

    # Text-based formats (.md, .txt, .html, .htm) are read as UTF-8 text.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    if path.lower().endswith((".html", ".htm")):
        text = BeautifulSoup(raw, "html.parser").get_text("\n")
    else:
        text = raw

    return Document(page_content=text, metadata={"source": os.path.basename(path)})


def _extract_pdf_text(path: str) -> str:
    """Extract text from a PDF, page by page, via pypdf.

    Note: this reads the embedded text layer only — it does not OCR scanned/image-only
    pages, so those will contribute little or no text.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_url(url: str, timeout: int = 30) -> Document:
    """Fetch a URL and load it into a Document, extracting text from HTML pages."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-doc-assistant"})
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")

    if "html" in content_type or url.lower().endswith((".html", ".htm")):
        text = BeautifulSoup(resp.text, "html.parser").get_text("\n")
    else:
        text = resp.text

    return Document(page_content=text, metadata={"source": url})


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


def _section_label(metadata: dict) -> str:
    """Build a human-readable section path like 'FastAPI Guide > Query Parameters'."""
    parts = [metadata[key] for key in ("h1", "h2", "h3") if metadata.get(key)]
    return " > ".join(parts)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split loaded documents into indexable chunks (see module docstring)."""
    recursive = _recursive_splitter()
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS, strip_headers=False
    )
    chunks: List[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        is_markdown = source.lower().endswith(".md") or "\n# " in ("\n" + doc.page_content)

        if is_markdown:
            # Stage 1: split on headers, capturing the section path as metadata.
            sections = md_splitter.split_text(doc.page_content)
            for section in sections:
                section.metadata["source"] = source
                label = _section_label(section.metadata)
                if label:
                    section.metadata["section"] = label
            # Stage 2: enforce size limits within each section.
            sized = recursive.split_documents(sections)
            chunks.extend(sized)
        else:
            chunks.extend(recursive.split_documents([doc]))

    # Keep only the metadata we care about, and drop empty chunks.
    cleaned: List[Document] = []
    for chunk in chunks:
        if not chunk.page_content.strip():
            continue
        meta = {"source": chunk.metadata.get("source", "unknown")}
        if chunk.metadata.get("section"):
            meta["section"] = chunk.metadata["section"]
        cleaned.append(Document(page_content=chunk.page_content, metadata=meta))
    return cleaned


# --------------------------------------------------------------------------- #
# High-level ingestion entry points
# --------------------------------------------------------------------------- #
def ingest_files(paths: List[str]) -> dict:
    """Load, chunk, and index a list of local files."""
    docs = [load_file(p) for p in paths]
    return _index(docs)


def ingest_urls(urls: List[str]) -> dict:
    """Load, chunk, and index a list of URLs."""
    docs = [load_url(u) for u in urls]
    return _index(docs)


def ingest_corpus_dir(directory: str | None = None) -> dict:
    """Ingest every supported file in the corpus directory (used at startup / by CLI)."""
    directory = directory or settings.corpus_dir
    paths = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith((".md", ".txt", ".html", ".htm", ".pdf"))
    ]
    return ingest_files(paths)


def _index(docs: List[Document]) -> dict:
    """Chunk and store loaded documents; return an ingestion summary."""
    chunks = chunk_documents(docs)
    added = add_documents(chunks)
    return {
        "ingested_sources": [d.metadata.get("source", "unknown") for d in docs],
        "documents_added": len(docs),
        "chunks_added": added,
    }
