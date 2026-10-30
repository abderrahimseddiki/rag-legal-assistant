"""Fabrique de LLM : OpenAI ou Mistral selon la configuration."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquante (voir .env.example)")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )
    if settings.llm_provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        if not settings.mistral_api_key:
            raise RuntimeError("MISTRAL_API_KEY manquante (voir .env.example)")
        return ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=settings.temperature,
        )
    raise ValueError(f"Fournisseur inconnu : {settings.llm_provider}")
