"""Ligne de commande : `python -m rag_assistant.cli ingest | ask "question"`."""

from __future__ import annotations

import argparse
import logging

from .config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="rag-assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Indexer les documents de data/raw")
    ask_p = sub.add_parser("ask", help="Poser une question")
    ask_p.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    if args.cmd == "ingest":
        from .ingest import ingest

        n = ingest(settings)
        print(f"{n} chunks indexés dans {settings.chroma_dir}")
    elif args.cmd == "ask":
        from .chain import RAGAssistant

        result = RAGAssistant(settings).ask(args.question)
        print(result.answer, "\n")
        for s in result.sources:
            print(f"  [{s['index']}] {s['label']}")
        print(f"\nAncrée : {result.checks['grounded']}")


if __name__ == "__main__":
    main()
