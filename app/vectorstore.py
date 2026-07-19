"""Vector store access layer.

Wraps a persistent ChromaDB collection and the embedding model behind a tiny API used by
the ingestion pipeline and the retrieval node:

* :func:`add_documents`      – embed and store chunks
* :func:`similarity_search`  – return the top-k chunks for a query
* :func:`list_sources`       – summarise what is indexed (for ``GET /documents``)

Both the embedding model and the Chroma client are created lazily and cached, so importing
this module is cheap and does not download the (~80 MB) sentence-transformers model until
it is actually needed. :func:`set_vectorstore` allows tests to inject a lightweight
in-memory store instead.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import settings

# Module-level override used by tests (see set_vectorstore / reset_vectorstore).
_override_store: Optional[VectorStore] = None


@lru_cache
def get_embeddings():
    """Return a cached embedding model.

    Uses a local sentence-transformers model via ``langchain-huggingface`` so the pipeline
    needs no embedding API key. The model is downloaded from the Hugging Face hub on first
    use and cached locally thereafter.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache
def _default_store() -> VectorStore:
    """Build (or open) the persistent Chroma collection."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def get_vectorstore() -> VectorStore:
    """Return the active vector store (a test override if set, else Chroma)."""
    return _override_store if _override_store is not None else _default_store()


def set_vectorstore(store: VectorStore) -> None:
    """Inject a custom vector store. Used by tests to avoid ChromaDB / model downloads."""
    global _override_store
    _override_store = store


def reset_vectorstore() -> None:
    """Clear any injected override and reset the cached Chroma client."""
    global _override_store
    _override_store = None
    _default_store.cache_clear()


def add_documents(chunks: List[Document]) -> int:
    """Embed and store the given chunks. Returns the number of chunks added."""
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, k: Optional[int] = None) -> List[Document]:
    """Return the top-k most similar chunks for a query."""
    return get_vectorstore().similarity_search(query, k=k or settings.top_k)


def list_sources() -> List[dict]:
    """Summarise indexed documents as ``[{"source": ..., "chunk_count": ...}, ...]``.

    Reads chunk metadata directly from the underlying collection. Works for both the
    Chroma store and the in-memory store used in tests.
    """
    store = get_vectorstore()
    metadatas: List[dict] = []

    # Chroma: read straight from the collection.
    if hasattr(store, "get"):
        try:
            metadatas = store.get(include=["metadatas"]).get("metadatas") or []
        except Exception:
            metadatas = []
    # InMemoryVectorStore (tests): iterate the internal store.
    elif hasattr(store, "store"):
        metadatas = [rec.get("metadata", {}) for rec in store.store.values()]

    counts: dict[str, int] = {}
    for meta in metadatas:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    return [
        {"source": source, "chunk_count": count}
        for source, count in sorted(counts.items())
    ]
