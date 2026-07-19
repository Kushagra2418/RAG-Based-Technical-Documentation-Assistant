"""Workflow assembly.

Wires the nodes and conditional edges into a compiled LangGraph ``StateGraph``. The
compiled graph is cached so the FastAPI app builds it once at startup.

Graph shape::

        START
          │
    query_analysis
          │
       retrieve ◄─────────────┐
          │                   │
    grade_documents           │
          │ (conditional)     │
   ┌──────┼───────────┐       │
   ▼      ▼           ▼       │
generate  web_search  transform_query
   │      │                   │
   │      └──► generate        └──(loop back to retrieve)
   │ (conditional: grounded?)
   ├──► END
   └──► transform_query ──► retrieve  (regenerate with better context)
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph import edges, nodes
from app.graph.state import GraphState


def build_graph():
    """Construct and compile the RAG StateGraph."""
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("query_analysis", nodes.query_analysis)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade_documents", nodes.grade_documents)
    builder.add_node("transform_query", nodes.transform_query)
    builder.add_node("web_search", nodes.web_search)
    builder.add_node("generate", nodes.generate)

    # Fixed edges
    builder.add_edge(START, "query_analysis")
    builder.add_edge("query_analysis", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("transform_query", "retrieve")   # the self-correction loop
    builder.add_edge("web_search", "generate")

    # Conditional edge: what to do after grading the retrieved documents.
    builder.add_conditional_edges(
        "grade_documents",
        edges.decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    # Conditional edge: after generation, accept the answer or retry (Self-RAG).
    builder.add_conditional_edges(
        "generate",
        edges.decide_after_generation,
        {
            "end": END,
            "transform_query": "transform_query",
        },
    )

    return builder.compile()


@lru_cache
def get_graph():
    """Return the compiled graph, building it once and caching it."""
    return build_graph()
