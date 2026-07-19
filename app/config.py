"""Application configuration.

All runtime settings are read from environment variables (or a local ``.env`` file)
using :class:`pydantic_settings.BaseSettings`. This keeps secrets such as the Groq API
key out of the source code and makes the whole pipeline tunable without code changes.

Import the singleton :data:`settings` object anywhere in the application::

    from app.config import settings
    print(settings.groq_model)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) ---------------------------------------------------------
    # Groq deprecated the Llama chat models in mid-2026; the openai/gpt-oss family
    # is the current production-grade default. Override with GROQ_MODEL if Groq's
    # catalogue changes again (see https://console.groq.com/docs/models).
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0

    # --- Embeddings ---------------------------------------------------------
    # A local sentence-transformers model. Runs offline after the first download
    # and needs no API key, which keeps the whole pipeline free to run locally.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store -------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "tech_docs"

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval / workflow ----------------------------------------------
    top_k: int = 4
    # Maximum number of query-rewrite + re-retrieve attempts before giving up.
    max_retries: int = 2

    # --- Optional / bonus features -----------------------------------------
    # Web-search fallback (Tavily). Only used when a key is present AND the flag is on.
    tavily_api_key: str = ""
    enable_web_search: bool = False
    # Groundedness / hallucination check on the generated answer (Self-RAG style).
    enable_hallucination_check: bool = True

    # --- Paths --------------------------------------------------------------
    corpus_dir: str = "./data/corpus"
    feedback_path: str = "./data/feedback.jsonl"

    @property
    def web_search_enabled(self) -> bool:
        """Web search is only active when explicitly enabled and a key is set."""
        return self.enable_web_search and bool(self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (read once per process)."""
    return Settings()


# Convenient module-level singleton.
settings = get_settings()
