"""Pipeline d'ingestion : fichiers -> pages -> chunks -> ChromaDB."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from .chunking import chunk_id, chunk_page
from .config import Settings
from .loaders import load_directory
from .store import get_store

log = logging.getLogger(__name__)


def ingest(settings: Settings, batch_size: int = 128) -> int:
    pages = load_directory(settings.raw_dir)
    if not pages:
        raise SystemExit(f"Aucun document trouvé dans {settings.raw_dir}. Voir docs/DATA.md.")

    chunks = []
    for page in pages:
        chunks.extend(
            chunk_page(page.text, page.metadata, settings.chunk_max_chars, settings.chunk_overlap)
        )
    # dédoublonnage par identifiant stable
    unique = {chunk_id(c): c for c in chunks}
    log.info("%d pages -> %d chunks uniques", len(pages), len(unique))

    store = get_store(settings)
    ids = list(unique)
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        docs = [
            Document(page_content=unique[i].text, metadata=unique[i].metadata) for i in batch_ids
        ]
        store.add_documents(docs, ids=batch_ids)
    return len(ids)
