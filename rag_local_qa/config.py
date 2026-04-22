"""Application configuration for the local RAG assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    """Centralized configuration used across the whole project."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    upload_dir: Path = field(init=False)
    vector_store_dir: Path = field(init=False)

    # File handling
    allowed_extensions: Tuple[str, ...] = (".pdf", ".txt")

    # Text splitting
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    min_chunk_length: int = field(default_factory=lambda: int(os.getenv("MIN_CHUNK_LENGTH", "40")))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    max_answer_sentences: int = field(default_factory=lambda: int(os.getenv("MAX_ANSWER_SENTENCES", "4")))

    # Embedding model
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )

    def __post_init__(self) -> None:
        data_dir = self.base_dir / "data"
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "upload_dir", data_dir / "uploads")
        object.__setattr__(self, "vector_store_dir", data_dir / "vector_store")

    def ensure_directories(self) -> None:
        """Create runtime directories when the app starts."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Create a new configuration object."""
    return AppConfig()
