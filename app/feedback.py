"""Feedback storage.

Persists thumbs-up/down feedback to a JSON Lines file (one JSON object per line). JSONL is
append-only and easy to analyse later, which suits an evaluation signal that accumulates
over time. In production this would move to a database, but a file keeps the assignment
self-contained and dependency-free.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.models import FeedbackRequest


def save_feedback(feedback: FeedbackRequest) -> str:
    """Append one feedback record and return its generated id."""
    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **feedback.model_dump(),
    }

    os.makedirs(os.path.dirname(settings.feedback_path) or ".", exist_ok=True)
    with open(settings.feedback_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return feedback_id
