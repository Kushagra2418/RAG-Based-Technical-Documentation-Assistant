"""Conditional routing functions (the graph's decision points).

Each function inspects the state after a node runs and returns a string key that the graph
maps to the next node. This is where self-correction lives: the routing logic decides
whether to generate, retry with a rewritten query, fall back to web search, or give up.
"""

from __future__ import annotations

from app.config import settings
from app.graph.state import GraphState


def decide_after_grading(state: GraphState) -> str:
    """Route after document grading.

    * Relevant documents found            → ``"generate"``
    * None found, retries remaining        → ``"transform_query"`` (rewrite + re-retrieve)
    * None found, retries exhausted, web on → ``"web_search"``
    * None found, retries exhausted, web off → ``"generate"`` (produces "I don't know")
    """
    if state.get("documents"):
        return "generate"

    if state.get("retries", 0) < state.get("max_retries", settings.max_retries):
        return "transform_query"

    if settings.web_search_enabled and not state.get("web_search_used", False):
        return "web_search"

    return "generate"


def decide_after_generation(state: GraphState) -> str:
    """Route after generation, based on the Self-RAG groundedness check.

    * Check disabled, answer grounded, or no context to improve on → ``"end"``
    * Answer not grounded and a retry budget remains               → ``"transform_query"``
      (re-retrieve for better supporting context, then regenerate)
    """
    grounded = state.get("grounded")
    has_retries = state.get("retries", 0) < state.get("max_retries", settings.max_retries)

    if grounded is False and state.get("documents") and has_retries:
        return "transform_query"
    return "end"
