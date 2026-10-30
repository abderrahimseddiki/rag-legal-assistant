"""Chargement des documents sources (PDF, HTML, TXT/MD) en texte brut + métadonnées."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass
class RawPage:
    text: str
    metadata: dict = field(default_factory=dict)


def _load_pdf(path: Path) -> list[RawPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(RawPage(text, {"source": path.name, "page": i}))
    return pages


def _load_html(path: Path) -> list[RawPage]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [RawPage(text, {"source": path.name})]


def _load_text(path: Path) -> list[RawPage]:
    return [RawPage(path.read_text(encoding="utf-8", errors="ignore"), {"source": path.name})]


def load_file(path: Path) -> list[RawPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".html", ".htm"}:
        return _load_html(path)
    if suffix in {".txt", ".md"}:
        return _load_text(path)
    raise ValueError(f"Format non supporté : {path.suffix}")


def load_directory(directory: str | Path) -> list[RawPage]:
    directory = Path(directory)
    pages: list[RawPage] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            pages.extend(load_file(path))
    return pages
