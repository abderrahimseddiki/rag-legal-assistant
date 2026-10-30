"""Accès à la base vectorielle ChromaDB (persistée sur disque)."""

from __future__ import annotations

from langchain_chroma import Chroma

from .config import Settings
from .embeddings import get_embeddings


def get_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings.embedding_model),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
