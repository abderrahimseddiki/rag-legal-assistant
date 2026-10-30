"""Recherche en deux étapes : vectorielle large, puis reranking par cross-encoder.

La recherche vectorielle est rapide mais approximative ; le cross-encoder lit la
question et le passage ensemble et les reclasse beaucoup plus finement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

from .config import Settings
from .store import get_store


@dataclass
class Retrieved:
    document: Document
    score: float


@lru_cache
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = get_store(settings)

    def retrieve(self, question: str) -> list[Retrieved]:
        s = self.settings
        hits = self.store.similarity_search_with_relevance_scores(question, k=s.top_k_retrieve)
        candidates = [Retrieved(doc, float(score)) for doc, score in hits]
        if not candidates:
            return []
        if s.use_reranker:
            reranker = _get_reranker(s.reranker_model)
            scores = reranker.predict([(question, c.document.page_content) for c in candidates])
            candidates = [Retrieved(c.document, float(sc)) for c, sc in zip(candidates, scores)]
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: s.top_k_final]
