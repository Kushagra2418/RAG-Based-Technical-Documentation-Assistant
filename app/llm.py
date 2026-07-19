"""LLM access layer.

Wraps the Groq chat model and exposes small, focused helpers that each node in the
LangGraph workflow calls:

* :func:`analyze_query`      – rewrite + classify the question (Query Analysis node)
* :func:`grade_document`     – judge one chunk relevant / irrelevant (Grading node)
* :func:`generate_answer`    – write the grounded, cited answer (Generation node)
* :func:`rewrite_query`      – produce a better query for re-retrieval
* :func:`check_grounded`     – verify the answer is supported by context (Self-RAG)

The Groq client is created lazily and cached, so importing this module never requires an
API key (the key is only needed when a node actually calls the model). All structured
calls degrade gracefully: if the model returns something unexpected, a sensible default
is used instead of crashing the request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings


# --------------------------------------------------------------------------- #
# Lazy Groq client
# --------------------------------------------------------------------------- #
@lru_cache
def get_llm():
    """Return a cached ChatGroq client.

    Imported lazily so that importing this module (and therefore building the graph or
    the FastAPI app) does not require ``GROQ_API_KEY`` to be set.
    """
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment before "
            "calling the LLM. See .env.example."
        )
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


def _structured(schema: type[BaseModel]):
    """Return the LLM bound to a structured (Pydantic) output schema."""
    return get_llm().with_structured_output(schema)


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #
class QueryAnalysis(BaseModel):
    """Structured result of the Query Analysis node."""

    rewritten_query: str = Field(
        description="A clarified, expanded search query optimised for vector retrieval."
    )
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference"] = Field(
        description="The category of the question."
    )


class GradeDocument(BaseModel):
    """Binary relevance grade for a single retrieved chunk."""

    relevant: bool = Field(description="True if the chunk helps answer the question.")


class GroundedGrade(BaseModel):
    """Binary groundedness grade for a generated answer."""

    grounded: bool = Field(
        description="True if every claim in the answer is supported by the provided context."
    )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You optimise questions for retrieval over a corpus of technical "
            "documentation. Rewrite the user's question into a concise, keyword-rich "
            "search query (expand abbreviations, add likely synonyms, remove chit-chat) "
            "and classify it as one of: conceptual, how-to, troubleshooting, "
            "api-reference. Preserve the original intent; do not answer the question.",
        ),
        (
            "human",
            "Conversation so far (may be empty):\n{history}\n\n"
            "User question:\n{question}",
        ),
    ]
)

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance grader for a retrieval system. Decide whether the "
            "document chunk contains information that helps answer the question. Judge "
            "relevance to the question's topic, not writing quality. Answer only with the "
            "structured boolean.",
        ),
        ("human", "Question:\n{question}\n\nChunk:\n{document}"),
    ]
)

_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical documentation assistant. Answer the question using ONLY "
            "the numbered context passages below. Follow these rules strictly:\n"
            "1. Ground every statement in the context. Do not use outside knowledge.\n"
            "2. Cite the passages you use inline with bracketed numbers, e.g. [1], [2].\n"
            "3. If the context does not contain the answer, say you don't know based on "
            "the available documentation — do not invent an answer.\n"
            "4. Be clear and concise; prefer short paragraphs and code blocks where useful.\n"
            "5. Write ALL math using LaTeX with dollar-sign delimiters, never plain brackets "
            "or parentheses: use $...$ for inline math (e.g. $\\theta_0$) and $$...$$ on its "
            "own line for standalone equations (e.g. $$J(\\theta) = -\\frac{{1}}{{m}}\\sum...$$). "
            "Never wrap math in [ ] or ( ) — those are reserved for citations and prose.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext passages:\n{context}\n\n"
            "Write the answer with inline [n] citations.",
        ),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The previous search query returned no relevant documentation. Rewrite it to "
            "improve retrieval: try different terminology, break compound questions into "
            "core keywords, and use terms likely to appear in technical docs. Return only "
            "the improved query text.",
        ),
        ("human", "Original question:\n{question}\n\nQuery that failed:\n{query}"),
    ]
)

_GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You verify that an answer is fully supported by the provided context "
            "(a hallucination / groundedness check). Return grounded=true only if every "
            "factual claim in the answer can be traced to the context. An honest "
            "'I don't know' answer counts as grounded.",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


# --------------------------------------------------------------------------- #
# Helpers used by the graph nodes
# --------------------------------------------------------------------------- #
def analyze_query(question: str, history: str = "") -> QueryAnalysis:
    """Rewrite and classify a question. Falls back to the raw question on error."""
    try:
        chain = _QUERY_ANALYSIS_PROMPT | _structured(QueryAnalysis)
        result = chain.invoke({"question": question, "history": history or "(none)"})
        # Guard against an empty rewrite.
        if not result.rewritten_query.strip():
            result.rewritten_query = question
        return result
    except Exception:
        return QueryAnalysis(rewritten_query=question, query_type="conceptual")


def grade_document(question: str, document: str) -> bool:
    """Grade a single chunk. On error, keep the chunk (fail open) to avoid dropping
    potentially useful context."""
    try:
        chain = _GRADE_PROMPT | _structured(GradeDocument)
        return bool(chain.invoke({"question": question, "document": document}).relevant)
    except Exception:
        return True


def rewrite_query(question: str, query: str) -> str:
    """Produce an improved retrieval query. Falls back to the original question."""
    try:
        chain = _REWRITE_PROMPT | get_llm()
        text = chain.invoke({"question": question, "query": query}).content
        return text.strip() or question
    except Exception:
        return question


def generate_answer(question: str, context: str) -> str:
    """Generate the final grounded, cited answer."""
    chain = _GENERATE_PROMPT | get_llm()
    return chain.invoke({"question": question, "context": context}).content.strip()


def check_grounded(answer: str, context: str) -> bool:
    """Return True if the answer is supported by the context. Fail open on error."""
    try:
        chain = _GROUNDED_PROMPT | _structured(GroundedGrade)
        return bool(chain.invoke({"context": context, "answer": answer}).grounded)
    except Exception:
        return True


def format_context(documents: List[Document]) -> str:
    """Render retrieved chunks into a numbered context block for the generation prompt.

    The numbering here matches the ``[n]`` citation numbers the model is asked to use and
    the ``id`` field of the :class:`~app.models.Source` objects returned to the client.
    """
    blocks = []
    for i, doc in enumerate(documents, start=1):
        src = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section")
        header = f"[{i}] source: {src}" + (f" | section: {section}" if section else "")
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)