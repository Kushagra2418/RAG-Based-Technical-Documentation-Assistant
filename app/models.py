"""Pydantic schemas for API requests and responses.

These models define the public contract of the FastAPI service. FastAPI uses them for
automatic request validation, response serialization, and OpenAPI documentation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# /query
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    question: str = Field(..., min_length=1, description="The user's natural-language question.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id. When provided, previous turns in the same "
        "session are used to resolve follow-up questions (conversation memory).",
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override the number of chunks to retrieve."
    )


class Source(BaseModel):
    """A single cited source chunk returned alongside an answer."""

    id: int = Field(..., description="Citation number referenced in the answer, e.g. [1].")
    source: str = Field(..., description="Origin of the chunk (file name or URL).")
    section: Optional[str] = Field(default=None, description="Section / heading, if known.")
    snippet: str = Field(..., description="A short preview of the chunk text.")
    origin: Literal["corpus", "web"] = Field(
        default="corpus", description="Whether the chunk came from the corpus or web search."
    )


class QueryResponse(BaseModel):
    """Response for ``POST /query``."""

    answer: str
    sources: List[Source]
    query_type: str = Field(..., description="Detected query type (conceptual/how-to/etc.).")
    rewritten_query: str = Field(..., description="The query actually used for retrieval.")
    retries: int = Field(..., description="Number of query-rewrite retries performed.")
    grounded: Optional[bool] = Field(
        default=None,
        description="Result of the hallucination check: True if the answer is supported "
        "by the retrieved context, False if not, None if the check was disabled.",
    )
    web_search_used: bool = Field(default=False)
    session_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# /ingest
# --------------------------------------------------------------------------- #
class IngestResponse(BaseModel):
    """Response for ``POST /ingest``."""

    ingested_sources: List[str] = Field(..., description="Names/URLs that were ingested.")
    documents_added: int = Field(..., description="Number of source documents processed.")
    chunks_added: int = Field(..., description="Number of chunks written to the vector store.")
    message: str


# --------------------------------------------------------------------------- #
# /documents
# --------------------------------------------------------------------------- #
class DocumentInfo(BaseModel):
    """Summary of one indexed source document."""

    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """Response for ``GET /documents``."""

    documents: List[DocumentInfo]
    total_sources: int
    total_chunks: int


# --------------------------------------------------------------------------- #
# /feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    """Body for ``POST /feedback``."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or down.")
    comment: Optional[str] = Field(default=None, description="Optional free-text comment.")
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response for ``POST /feedback``."""

    status: str
    feedback_id: str
