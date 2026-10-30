"""Configuration centralisée, chargée depuis les variables d'environnement / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "mistral"] = "openai"
    openai_api_key: str = ""
    mistral_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    mistral_model: str = "mistral-small-latest"
    temperature: float = 0.0

    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True

    chroma_dir: str = "data/chroma"
    collection_name: str = "legal_docs"
    raw_dir: str = "data/raw"

    chunk_max_chars: int = 1500
    chunk_overlap: int = 200
    top_k_retrieve: int = 20
    top_k_final: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
