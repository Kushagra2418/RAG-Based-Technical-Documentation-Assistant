"""Conversation memory (bonus feature).

Maintains a short, in-memory chat history per ``session_id`` so the Query Analysis node can
resolve follow-up questions such as "how do I override it?". Only the last few turns are
kept, to bound prompt size. Because it is a process-local dict, history is not shared
across workers or restarts — a production system would use Redis or a database, but this is
enough to demonstrate multi-turn behaviour.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

# session_id -> deque of (question, answer) pairs, newest last.
_MAX_TURNS = 4
_history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=_MAX_TURNS))


def render_history(session_id: str | None) -> str:
    """Render prior turns for a session into a compact prompt string."""
    if not session_id or session_id not in _history:
        return ""
    lines = []
    for question, answer in _history[session_id]:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def record_turn(session_id: str | None, question: str, answer: str) -> None:
    """Store a completed turn for later follow-ups."""
    if session_id:
        _history[session_id].append((question, answer))
