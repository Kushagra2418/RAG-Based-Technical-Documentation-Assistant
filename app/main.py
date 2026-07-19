"""FastAPI application.

Exposes the RAG workflow over HTTP:

======  ==========  ================================================
Method  Endpoint    Purpose
======  ==========  ================================================
POST    /query      Ask a question; returns an answer with sources.
POST    /ingest     Ingest new documents (file uploads and/or URLs).
GET     /documents  List what is indexed in the corpus.
POST    /feedback   Record thumbs up/down on an answer.
GET     /health     Liveness probe.
GET     /           Service metadata.
======  ==========  ================================================

On startup the app auto-ingests the bundled corpus if the vector store is empty, so a
fresh clone answers questions immediately.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from app import feedback as feedback_store
from app import ingestion, memory, vectorstore
from app.config import settings
from app.graph.workflow import get_graph
from app.models import (
    DocumentInfo,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Source,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph and seed the corpus on startup."""
    get_graph()  # Compile once so the first request is fast.
    try:
        if not vectorstore.list_sources() and os.path.isdir(settings.corpus_dir):
            result = ingestion.ingest_corpus_dir()
            print(f"[startup] Seeded corpus: {result}")
    except Exception as exc:  # pragma: no cover - startup best effort
        print(f"[startup] Corpus seeding skipped: {exc}")
    yield


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG over technical docs, built with LangGraph + Groq.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_sources(documents) -> List[Source]:
    """Convert the graph's final documents into citation objects.

    The 1-based index matches the ``[n]`` markers the generation prompt uses and the
    numbering in :func:`app.llm.format_context`.
    """
    sources: List[Source] = []
    for i, doc in enumerate(documents, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        sources.append(
            Source(
                id=i,
                source=doc.metadata.get("source", "unknown"),
                section=doc.metadata.get("section"),
                snippet=(snippet[:240] + "…") if len(snippet) > 240 else snippet,
                origin="web" if doc.metadata.get("origin") == "web" else "corpus",
            )
        )
    return sources


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root() -> dict:
    """Service metadata and effective configuration."""
    return {
        "service": "RAG Technical Documentation Assistant",
        "model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "web_search_enabled": settings.web_search_enabled,
        "hallucination_check_enabled": settings.enable_hallucination_check,
        "endpoints": ["/query", "/ingest", "/documents", "/feedback", "/health", "/docs"],
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using the self-corrective RAG workflow."""
    initial_state = {
        "question": request.question,
        "history": memory.render_history(request.session_id),
        "max_retries": settings.max_retries,
    }
    if request.top_k:
        # Per-request override of retrieval breadth.
        settings.top_k = request.top_k

    try:
        final_state = get_graph().invoke(initial_state)
    except RuntimeError as exc:
        # Most commonly a missing GROQ_API_KEY.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")

    answer = final_state.get("generation", "")
    documents = final_state.get("documents", [])
    memory.record_turn(request.session_id, request.question, answer)

    return QueryResponse(
        answer=answer,
        sources=_build_sources(documents),
        query_type=final_state.get("query_type", "conceptual"),
        rewritten_query=final_state.get("query", request.question),
        retries=final_state.get("retries", 0),
        grounded=final_state.get("grounded"),
        web_search_used=final_state.get("web_search_used", False),
        session_id=request.session_id,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(
    files: List[UploadFile] = File(default=[]),
    urls: List[str] = Form(default=[]),
) -> IngestResponse:
    """Ingest new documents from uploaded files and/or URLs.

    Send as ``multipart/form-data``. Provide ``files`` (one or more file parts),
    ``urls`` (one or more form fields), or both. At least one is required.
    """
    if not files and not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file upload or one URL.",
        )

    total = {"ingested_sources": [], "documents_added": 0, "chunks_added": 0}

    # Uploaded files: write to a temp dir preserving the original filename, then ingest.
    if files:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for upload in files:
                if not upload.filename:
                    continue
                dest = os.path.join(tmp, os.path.basename(upload.filename))
                with open(dest, "wb") as fh:
                    fh.write(await upload.read())
                paths.append(dest)
            if paths:
                result = ingestion.ingest_files(paths)
                _merge(total, result)

    # URLs
    clean_urls = [u.strip() for u in urls if u.strip()]
    if clean_urls:
        try:
            result = ingestion.ingest_urls(clean_urls)
            _merge(total, result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch a URL: {exc}")

    return IngestResponse(
        ingested_sources=total["ingested_sources"],
        documents_added=total["documents_added"],
        chunks_added=total["chunks_added"],
        message=f"Indexed {total['chunks_added']} chunk(s) from "
        f"{total['documents_added']} document(s).",
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["rag"])
def documents() -> DocumentsResponse:
    """List the source documents currently indexed and their chunk counts."""
    sources = vectorstore.list_sources()
    infos = [DocumentInfo(**s) for s in sources]
    return DocumentsResponse(
        documents=infos,
        total_sources=len(infos),
        total_chunks=sum(s["chunk_count"] for s in sources),
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["rag"])
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record thumbs up/down feedback on an answer."""
    feedback_id = feedback_store.save_feedback(request)
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


def _merge(total: dict, result: dict) -> None:
    """Accumulate one ingestion result into the running total."""
    total["ingested_sources"].extend(result["ingested_sources"])
    total["documents_added"] += result["documents_added"]
    total["chunks_added"] += result["chunks_added"]
