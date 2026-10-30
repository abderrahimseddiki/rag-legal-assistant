"""Formatage du contexte numéroté et vérification des citations [n] dans la réponse."""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[(\d+)\]")
NOT_FOUND = "Je ne trouve pas la réponse dans les documents fournis."


@dataclass
class Source:
    index: int
    source: str
    page: int | None
    article: str | None
    excerpt: str

    def label(self) -> str:
        parts = [self.source]
        if self.article:
            parts.append(f"art. {self.article}")
        if self.page:
            parts.append(f"p. {self.page}")
        return ", ".join(parts)


def build_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.index}] ({s.label()})\n{s.excerpt}" for s in sources)


def extract_citations(answer: str) -> list[int]:
    seen: list[int] = []
    for m in CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate(answer: str, n_sources: int) -> dict:
    """Une réponse est « ancrée » si elle cite au moins une source existante et aucune inventée."""
    cited = extract_citations(answer)
    invalid = [n for n in cited if n < 1 or n > n_sources]
    abstained = NOT_FOUND.lower() in answer.lower()
    return {
        "cited": [n for n in cited if n not in invalid],
        "invalid_citations": invalid,
        "abstained": abstained,
        "grounded": abstained or (bool(cited) and not invalid),
    }
