"""LangGraph nodes.

Each function is one node in the workflow. A node reads the current
:class:`~app.graph.state.GraphState` and returns a dict of the keys it wants to update.
Nodes are deliberately thin: they orchestrate calls to :mod:`app.llm`,
:mod:`app.vectorstore`, and :mod:`app.ingestion`, which keeps them easy to read and unit
test.

Pipeline order (see :mod:`app.graph.workflow`)::

    query_analysis → retrieve → grade_documents → (route) → generate → (route) → END
                                                     ↘ transform_query → retrieve (loop)
                                                     ↘ web_search → generate
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app import llm, vectorstore
from app.config import settings
from app.graph.state import GraphState


def query_analysis(state: GraphState) -> dict:
    """Node 1 — rewrite and classify the question to improve retrieval."""
    analysis = llm.analyze_query(state["question"], history=state.get("history", ""))
    return {
        "query": analysis.rewritten_query,
        "query_type": analysis.query_type,
        # Initialise self-correction bookkeeping on the first pass.
        "retries": state.get("retries", 0),
        "max_retries": state.get("max_retries", settings.max_retries),
        "web_search_used": state.get("web_search_used", False),
    }


def retrieve(state: GraphState) -> dict:
    """Node 2 — similarity search over the vector store using the current query."""
    docs = vectorstore.similarity_search(state["query"], k=settings.top_k)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    """Node 3 — the self-corrective check: keep only chunks graded relevant.

    Web-search results (if any) are trusted and not re-graded here.
    """
    kept: List[Document] = []
    for doc in state["documents"]:
        if doc.metadata.get("origin") == "web":
            kept.append(doc)
        elif llm.grade_document(state["question"], doc.page_content):
            kept.append(doc)
    return {"documents": kept}


def transform_query(state: GraphState) -> dict:
    """Fallback node — rewrite the query and record the retry."""
    new_query = llm.rewrite_query(state["question"], state["query"])
    return {"query": new_query, "retries": state.get("retries", 0) + 1}


def web_search(state: GraphState) -> dict:
    """Optional fallback node — augment context with Tavily web-search results.

    Only reached when the web-search feature is enabled and configured. Results are added
    to whatever (possibly empty) set of graded documents already exists.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    tool = TavilySearchResults(max_results=settings.top_k, tavily_api_key=settings.tavily_api_key)
    try:
        results = tool.invoke({"query": state["query"]})
    except Exception:
        results = []

    web_docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "web-search"), "origin": "web"},
        )
        for r in results
        if r.get("content")
    ]
    return {"documents": state.get("documents", []) + web_docs, "web_search_used": True}


def generate(state: GraphState) -> dict:
    """Node 4 — generate the grounded, cited answer.

    If no context survived grading (and no fallback produced any), return an explicit
    "I don't know" rather than letting the model invent an answer.
    """
    documents = state.get("documents", [])
    if not documents:
        return {
            "generation": (
                "I couldn't find anything in the available documentation to answer that "
                "question. You may want to rephrase it or ingest more relevant documents."
            ),
            "grounded": True,  # An honest "I don't know" is trivially grounded.
        }

    context = llm.format_context(documents)
    answer = llm.generate_answer(state["question"], context)

    grounded = None
    if settings.enable_hallucination_check:
        grounded = llm.check_grounded(answer, context)

    return {"generation": answer, "grounded": grounded}
