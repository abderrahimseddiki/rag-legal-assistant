"""Embeddings locaux (sentence-transformers), gratuits et multilingues.

Les modèles E5 attendent les préfixes « query: » et « passage: » : sans eux,
la qualité de recherche chute nettement. On les ajoute ici automatiquement.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.is_e5:
            text = f"query: {text}"
        return self._encode([text])[0]


@lru_cache
def get_embeddings(model_name: str) -> LocalEmbeddings:
    return LocalEmbeddings(model_name)
