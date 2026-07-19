"""Standalone ingestion script.

Ingest the bundled corpus (or a directory / URLs you pass) into the vector store without
starting the API. Useful for pre-building the index.

Examples
--------
Ingest the default bundled corpus (``data/corpus``)::

    python -m scripts.ingest_corpus

Ingest a different directory::

    python -m scripts.ingest_corpus --dir ./my_docs

Ingest one or more URLs::

    python -m scripts.ingest_corpus --url https://example.com/guide.md
"""

from __future__ import annotations

import argparse

from app import ingestion
from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--dir", help="Directory of documents to ingest.", default=None)
    parser.add_argument("--url", action="append", default=[], help="URL to ingest (repeatable).")
    args = parser.parse_args()

    if args.url:
        result = ingestion.ingest_urls(args.url)
    else:
        directory = args.dir or settings.corpus_dir
        print(f"Ingesting corpus from: {directory}")
        result = ingestion.ingest_corpus_dir(directory)

    print(
        f"Done. Documents: {result['documents_added']}, "
        f"chunks: {result['chunks_added']}\nSources: {result['ingested_sources']}"
    )


if __name__ == "__main__":
    main()
