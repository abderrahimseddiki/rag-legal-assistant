"""Découpage « sémantique » adapté aux textes juridiques.

Stratégie en deux temps :
1. On coupe d'abord sur les frontières naturelles d'un texte de loi : les articles
   (« Art. 1er. », « Article 12 », « Art. 1382/1 »...). Un article est l'unité qu'on veut citer.
2. Si un article est trop long, on le redécoupe par paragraphes puis phrases,
   avec un chevauchement, sans jamais couper au milieu d'un mot.

Chaque chunk garde ses métadonnées (source, page, article) pour pouvoir citer précisément.
Module en Python pur : testable sans dépendance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ARTICLE_RE = re.compile(
    r"^\s*(?P<label>(?:Art\.|Article)\s*(?P<num>\d+(?:er)?(?:[/.-]\d+)*(?:\s?(?:bis|ter|quater))?))\b",
    re.IGNORECASE | re.MULTILINE,
)
WHITESPACE_RE = re.compile(r"[ \t]+")
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"-\n(?=\w)", "", text)  # mots coupés en fin de ligne
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_article(text: str) -> list[tuple[str | None, str]]:
    """Retourne une liste (numéro d'article ou None, texte)."""
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [(None, text)]
    parts: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        parts.append((None, preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((m.group("num").replace(" ", ""), text[m.start() : end].strip()))
    return parts


def _units(text: str) -> list[str]:
    """Paragraphes, eux-mêmes redécoupés en phrases s'ils sont trop longs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for para in _units(text):
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(s for s in SENTENCE_RE.split(para) if s)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:  # phrase géante : coupe sur un espace
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        candidate = f"{current}\n{piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            space = tail.find(" ")
            tail = tail[space + 1 :] if space != -1 else tail
            current = f"{tail} {piece}".strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_page(text: str, metadata: dict, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    out: list[Chunk] = []
    for article, body in split_by_article(normalize(text)):
        for j, piece in enumerate(split_long(body, max_chars, overlap)):
            meta = dict(metadata)
            if article:
                meta["article"] = article
            meta["part"] = j
            out.append(Chunk(piece, meta))
    return out


def chunk_id(chunk: Chunk) -> str:
    """Identifiant stable : relancer l'ingestion ne crée pas de doublons."""
    import hashlib

    m = chunk.metadata
    key = f"{m.get('source')}|{m.get('page')}|{m.get('article')}|{m.get('part')}|{chunk.text[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
