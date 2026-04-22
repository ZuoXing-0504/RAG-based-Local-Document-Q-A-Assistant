"""Core package for the local RAG document Q&A assistant."""

from .config import AppConfig, get_config
from .document_processor import DocumentProcessor
from .qa_engine import QAEngine
from .vector_store import VectorStoreManager

__all__ = [
    "AppConfig",
    "DocumentProcessor",
    "QAEngine",
    "VectorStoreManager",
    "get_config",
]
