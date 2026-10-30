"""API REST : POST /ask, GET /health."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chain import RAGAssistant

app = FastAPI(title="RAG Legal Assistant", version="0.1.0")
_assistant: RAGAssistant | None = None


def assistant() -> RAGAssistant:
    global _assistant
    if _assistant is None:
        _assistant = RAGAssistant()
    return _assistant


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return asdict(assistant().ask(req.question))
    except RuntimeError as exc:  # clé API manquante, etc.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
