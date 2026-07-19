"""Smoke tests for the RAG pipeline.

These tests exercise the *real* ingestion, chunking, graph wiring, and API layers, but
substitute two heavy external dependencies with lightweight fakes:

* the vector store  -> an in-memory store with deterministic fake embeddings, so no
  ChromaDB persistence or sentence-transformers download is required;
* the Groq LLM       -> deterministic Python stubs, so no API key or network call is made.

This lets the graph's control flow (retrieval, grading, generation, and the
self-correction retry loop) be verified deterministically and offline. Run directly::

    python -m tests.test_smoke

or with pytest::

    pytest -q
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from app import ingestion, llm, vectorstore
from app.llm import QueryAnalysis


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _install_fakes(grade_relevant: bool = True, grounded: bool = True) -> None:
    """Swap in an in-memory vector store and deterministic LLM stubs."""
    vectorstore.reset_vectorstore()
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=64))
    vectorstore.set_vectorstore(store)

    llm.analyze_query = lambda q, history="": QueryAnalysis(
        rewritten_query=q, query_type="how-to"
    )
    llm.grade_document = lambda q, d: grade_relevant
    llm.rewrite_query = lambda q, query: f"{query} (rewritten)"
    llm.generate_answer = lambda q, ctx: "FastAPI validates request bodies with Pydantic [1]."
    llm.check_grounded = lambda a, ctx: grounded


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_ingestion_and_chunking() -> None:
    _install_fakes()
    result = ingestion.ingest_corpus_dir("./data/corpus")
    assert result["documents_added"] == 4, "expected 4 corpus documents"
    assert result["chunks_added"] > 4, "expected multiple chunks from the corpus"

    sources = vectorstore.list_sources()
    assert len(sources) == 4
    # Markdown-header-aware chunking should attach section metadata.
    hits = vectorstore.similarity_search("request body", k=3)
    assert hits, "similarity search returned no chunks"
    assert any(h.metadata.get("section") for h in hits), "no section metadata captured"
    print("✓ ingestion + markdown-aware chunking")


def test_graph_happy_path() -> None:
    _install_fakes(grade_relevant=True)
    ingestion.ingest_corpus_dir("./data/corpus")
    from app.graph.workflow import build_graph

    graph = build_graph()
    state = graph.invoke({"question": "How does FastAPI handle request bodies?", "max_retries": 2})

    assert state["generation"], "no answer produced"
    assert state["documents"], "no documents carried through"
    assert state["query_type"] == "how-to"
    assert state["retries"] == 0, "happy path should not retry"
    assert state["grounded"] is True
    print("✓ graph happy path (retrieve -> grade -> generate)")


def test_graph_self_correction_gives_up() -> None:
    """When nothing is ever relevant, the loop must retry up to the limit then answer
    with an honest 'I don't know' rather than looping forever."""
    _install_fakes(grade_relevant=False)
    ingestion.ingest_corpus_dir("./data/corpus")
    from app.graph.workflow import build_graph

    graph = build_graph()
    state = graph.invoke({"question": "What is the airspeed of a swallow?", "max_retries": 2})

    assert state["retries"] == 2, f"expected 2 retries, got {state['retries']}"
    assert not state["documents"], "no documents should survive grading"
    assert "couldn't find" in state["generation"].lower()
    print("✓ graph self-correction loop terminates at retry limit")


def test_api_endpoints() -> None:
    from fastapi.testclient import TestClient

    _install_fakes(grade_relevant=True)
    from app.main import app

    with TestClient(app) as client:  # triggers startup corpus seeding into the fake store
        assert client.get("/health").json()["status"] == "ok"

        docs = client.get("/documents").json()
        assert docs["total_sources"] == 4
        assert docs["total_chunks"] > 4

        resp = client.post("/query", json={"question": "How do I define a Pydantic model?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["sources"] and body["sources"][0]["id"] == 1
        assert body["query_type"] == "how-to"

        fb = client.post(
            "/feedback",
            json={"question": "q", "answer": "a", "rating": "up", "comment": "clear"},
        )
        assert fb.status_code == 200 and fb.json()["status"] == "recorded"

        # Ingest via URL-less form should 400; ingest via a text file part should work.
        bad = client.post("/ingest", data={})
        assert bad.status_code == 400

        files = {"files": ("extra.txt", b"# Extra\nSome extra note about testing.", "text/plain")}
        ing = client.post("/ingest", files=files)
        assert ing.status_code == 200 and ing.json()["chunks_added"] >= 1
    print("✓ FastAPI endpoints: /health /documents /query /feedback /ingest")


def _run_all() -> None:
    test_ingestion_and_chunking()
    test_graph_happy_path()
    test_graph_self_correction_gives_up()
    test_api_endpoints()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _run_all()
