"""Évaluation automatique du RAG sur un jeu de questions annoté.

Métriques produites (toutes calculées, jamais saisies à la main) :
- retrieval_hit_rate : la bonne source (et le bon article si précisé) figure dans le top-k ;
- accuracy          : réponse jugée correcte vs la réponse de référence (LLM juge) ;
- hallucination_rate: part des réponses (hors abstentions) contenant une affirmation
                      non soutenue par les extraits cités (LLM juge) ;
- abstention_accuracy : sur les questions hors corpus, le système refuse bien de répondre ;
- citation_validity : part des réponses qui ne citent aucune source inexistante.

Usage : python eval/evaluate.py --questions eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from rag_assistant.chain import RAGAssistant  # noqa: E402
from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.llm import get_llm  # noqa: E402

JUDGE_CORRECT = """Tu évalues une réponse. Compare la RÉPONSE à la RÉFÉRENCE.
Réponds uniquement par un JSON : {"correct": true|false, "reason": "..."}.
"correct" = la réponse contient l'information essentielle de la référence, sans erreur."""

JUDGE_FAITHFUL = """Tu vérifies la fidélité d'une réponse à ses sources.
Réponds uniquement par un JSON : {"faithful": true|false, "unsupported": ["..."]}.
"faithful" = chaque affirmation de la réponse est soutenue par les EXTRAITS."""


def judge(llm, system: str, content: str) -> dict:
    raw = llm.invoke([SystemMessage(system), HumanMessage(content)]).content
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


def hit(item: dict, sources: list[dict]) -> bool:
    for s in sources:
        if s["source"] != item["expected_source"]:
            continue
        if item.get("expected_article") and s.get("article") != item["expected_article"]:
            continue
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.questions).read_text("utf-8").splitlines() if line.strip()]
    settings = get_settings()
    rag = RAGAssistant(settings)
    judge_llm = get_llm(settings)

    rows = []
    for item in items:
        t0 = time.perf_counter()
        res = rag.ask(item["question"])
        latency = time.perf_counter() - t0
        row = {
            "id": item["id"],
            "answerable": item["answerable"],
            "abstained": res.checks["abstained"],
            "citation_valid": not res.checks["invalid_citations"],
            "latency_s": round(latency, 2),
        }
        if item["answerable"]:
            row["retrieval_hit"] = hit(item, res.sources) if item.get("expected_source") else None
            row["correct"] = bool(
                judge(judge_llm, JUDGE_CORRECT,
                      f"RÉFÉRENCE : {item['reference_answer']}\n\nRÉPONSE : {res.answer}").get("correct")
            )
        if not res.checks["abstained"]:
            excerpts = "\n\n".join(f"[{s['index']}] {s['excerpt']}" for s in res.sources)
            row["faithful"] = bool(
                judge(judge_llm, JUDGE_FAITHFUL,
                      f"EXTRAITS :\n{excerpts}\n\nRÉPONSE : {res.answer}").get("faithful")
            )
        rows.append(row)
        print(f"{item['id']}: {row}")

    def rate(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    summary = {
        "n_questions": len(rows),
        "provider": settings.llm_provider,
        "reranker": settings.use_reranker,
        "retrieval_hit_rate": rate("retrieval_hit", answerable),
        "accuracy": rate("correct", answerable),
        "hallucination_rate": round(1 - rate("faithful", answered), 3) if answered else None,
        "abstention_accuracy": rate("abstained", unanswerable),
        "citation_validity": rate("citation_valid", rows),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"results-{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), "utf-8"
    )
    md = ["| Métrique | Valeur |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]
    (out / f"report-{stamp}.md").write_text("\n".join(md) + "\n", "utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
