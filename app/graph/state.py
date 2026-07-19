"""LangGraph state schema.

:class:`GraphState` is the single data structure threaded through every node. Each node
receives it and returns a partial dict; LangGraph merges the returned keys into the state.
The design is deliberately explicit — the fields below document exactly what data flows
through the pipeline and how self-correction is tracked.

Key design decisions
---------------------
* ``question`` is the immutable original; ``query`` is the mutable, possibly-rewritten
  string actually used for retrieval. Separating them means the answer and citations
  always reference the user's true intent while retrieval can be improved on retries.
* ``retries`` / ``max_retries`` bound the rewrite→re-retrieve loop so a query that never
  finds relevant docs terminates instead of cycling forever.
* ``grounded`` records the Self-RAG hallucination-check outcome so it can be surfaced to
  the caller for transparency.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """State passed between all nodes of the RAG workflow."""

    # --- Input ---
    question: str          # Original user question (never mutated).
    history: str           # Rendered prior conversation turns (empty if none).

    # --- Query analysis ---
    query: str             # Current retrieval query (rewritten on retries).
    query_type: str        # conceptual / how-to / troubleshooting / api-reference

    # --- Retrieval + grading ---
    documents: List[Document]   # Chunks kept after grading (the working context).

    # --- Self-correction bookkeeping ---
    retries: int           # Number of rewrite→re-retrieve attempts performed so far.
    max_retries: int       # Upper bound on retries before falling back / giving up.
    web_search_used: bool  # True if the web-search fallback contributed documents.

    # --- Generation + verification ---
    generation: str            # The final answer text.
    grounded: Optional[bool]   # Hallucination-check result (None if check disabled).
