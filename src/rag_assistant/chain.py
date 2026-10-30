"""Chaîne question -> recherche -> réponse citée."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from .citations import NOT_FOUND, Source, build_context, validate
from .config import Settings, get_settings
from .llm import get_llm
from .retriever import Retriever

SYSTEM_PROMPT = f"""Tu es un assistant juridique. Tu réponds UNIQUEMENT à partir des extraits fournis.
Règles :
- Chaque affirmation doit être suivie de la citation de son extrait, au format [1], [2]...
- N'invente jamais d'article, de numéro ou de règle absente des extraits.
- Si les extraits ne permettent pas de répondre, réponds exactement : « {NOT_FOUND} »
- Réponds dans la langue de la question, de façon concise et précise."""


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]
    checks: dict


class RAGAssistant:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.retriever = Retriever(self.settings)
        self.llm = get_llm(self.settings)

    def ask(self, question: str) -> Answer:
        hits = self.retriever.retrieve(question)
        sources = [
            Source(
                index=i,
                source=h.document.metadata.get("source", "?"),
                page=h.document.metadata.get("page"),
                article=h.document.metadata.get("article"),
                excerpt=h.document.page_content,
            )
            for i, h in enumerate(hits, start=1)
        ]
        if not sources:
            return Answer(question, NOT_FOUND, [], validate(NOT_FOUND, 0))

        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(f"Extraits :\n{build_context(sources)}\n\nQuestion : {question}"),
        ]
        text = self.llm.invoke(messages).content.strip()
        checks = validate(text, len(sources))
        return Answer(question, text, [asdict(s) | {"label": s.label()} for s in sources], checks)
