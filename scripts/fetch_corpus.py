"""Optional: fetch real documentation into the corpus directory.

The bundled corpus (``data/corpus/*.md``) is a curated, offline set so the project works
out of the box. This script demonstrates building a corpus from live URLs instead — it
downloads a small set of public technical docs and saves them as Markdown into the corpus
directory, ready to be ingested with ``python -m scripts.ingest_corpus``.

Usage::

    python -m scripts.fetch_corpus            # fetch the default list
    python -m scripts.fetch_corpus --url URL  # add your own (repeatable)
"""

from __future__ import annotations

import argparse
import os
import re

import requests

from app.config import settings

# A few small, plain-text-friendly public docs. Edit freely.
DEFAULT_URLS = [
    "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/query-params.md",
    "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/first-steps.md",
]


def _slugify(url: str) -> str:
    name = url.rstrip("/").split("/")[-1] or "document"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name if name.endswith((".md", ".txt")) else name + ".md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch docs into the corpus directory.")
    parser.add_argument("--url", action="append", default=[], help="URL to fetch (repeatable).")
    parser.add_argument("--out", default=settings.corpus_dir, help="Output directory.")
    args = parser.parse_args()

    urls = args.url or DEFAULT_URLS
    os.makedirs(args.out, exist_ok=True)

    for url in urls:
        print(f"Fetching {url}")
        resp = requests.get(url, timeout=30, headers={"User-Agent": "rag-doc-assistant"})
        resp.raise_for_status()
        dest = os.path.join(args.out, _slugify(url))
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
        print(f"  saved -> {dest}")

    print("\nDone. Now run: python -m scripts.ingest_corpus")


if __name__ == "__main__":
    main()
