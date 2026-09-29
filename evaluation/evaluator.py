"""Evaluation harness for KnowledgeOps AI.

Runs the RAG system against a hand-crafted test set and computes:
  - Retrieval hit rate
  - Answer accuracy (keyword match)
  - Faithfulness (LLM-as-judge)
  - Not-found accuracy
  - Citation presence

Usage:
    python -m evaluation.evaluator               # full run with judge
    python -m evaluation.evaluator --no-judge    # skip LLM-as-judge (saves API calls)
    python -m evaluation.evaluator --limit 5     # run only first 5 cases
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

from rag.answer import answer_question
from rag.embedder import Embedder
from rag.llm import get_client
from rag.vector_store import VectorStore


TEST_SET = Path("evaluation/test_set.json")
REPORT_JSON = Path("evaluation/report.json")


# ============================================================
# Result shape
# ============================================================

@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool

    # Per-case outcomes
    retrieved_source: str | None
    retrieved_pages: list[int]
    retrieval_hit: bool
    answer: str
    found: bool
    keyword_match: bool | None       # None for unanswerable
    faithful: bool | None            # None if answer empty or judge skipped
    correctly_not_found: bool | None # None for answerable
    has_citation: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    total: int
    answerable: int
    unanswerable: int
    metrics: dict
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "metrics": self.metrics,
            "cases": [c.to_dict() for c in self.cases],
        }


# ============================================================
# Metrics
# ============================================================

def retrieval_hit(sources: list, expected_source: str, expected_page: int | None) -> bool:
    """Did any cited source match the expected source (and page, if given)?"""
    for s in sources:
        if s.source_name == expected_source:
            if expected_page is None or s.page_number == expected_page:
                return True
    return False


def keyword_match(answer: str, keywords: list[str]) -> bool:
    """Did the answer contain all expected keywords (case-insensitive)?"""
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in keywords)


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


def judge_faithfulness(question: str, answer: str, contexts: list[str], llm_client) -> bool:
    """LLM-as-judge: does the answer contain only claims supported by context?"""
    context_block = "\n\n".join(contexts)
    prompt = f"""You are a strict fact-checker.

CONTEXT (source material):
{context_block}

ANSWER (to be judged):
{answer}

QUESTION: {question}

Does EVERY factual claim in the ANSWER come directly from the CONTEXT?
- If yes, respond starting with "YES".
- If the answer adds facts not in the context, respond starting with "NO".
Respond with YES or NO on the first line, then one short sentence explaining."""

    try:
        response = llm_client.complete(
            system="You are a strict fact-checker. Answer only YES or NO on the first line.",
            user=prompt,
        )
        first_line = response.strip().split("\n")[0].strip().upper()
        return first_line.startswith("YES")
    except Exception as e:
        print(f"  [judge error] {e}")
        return False


# ============================================================
# Runner
# ============================================================

def run_case(
    case: dict,
    embedder: Embedder,
    store: VectorStore,
    llm_client,
    run_judge: bool,
) -> CaseResult:
    """Run a single test case through the RAG pipeline."""
    question = case["question"]
    answerable = case["answerable"]

    result = answer_question(
        question,
        top_k=5,
        embedder=embedder,
        store=store,
        llm_client=llm_client,
    )

    # Retrieved sources
    retrieved_source = result.sources[0].source_name if result.sources else None
    retrieved_pages = [s.page_number for s in result.sources]

    # Retrieval hit (only meaningful for answerable questions with expected source)
    if answerable:
        r_hit = retrieval_hit(
            result.sources,
            case["expected_source"],
            case.get("expected_page"),
        )
    else:
        r_hit = False

    # Keyword match
    if answerable:
        km = keyword_match(result.answer, case.get("expected_keywords", []))
    else:
        km = None

    # Faithfulness (only for answers that found something)
    if run_judge and result.found and result.sources:
        contexts = [s.text for s in result.sources]
        faithful = judge_faithfulness(question, result.answer, contexts, llm_client)
    else:
        faithful = None

    # Not-found correctness
    correctly_not_found = None
    if not answerable:
        correctly_not_found = (not result.found)

    # Citation presence
    has_cite = has_citation(result.answer)

    return CaseResult(
        id=case["id"],
        question=question,
        answerable=answerable,
        retrieved_source=retrieved_source,
        retrieved_pages=retrieved_pages,
        retrieval_hit=r_hit,
        answer=result.answer,
        found=result.found,
        keyword_match=km,
        faithful=faithful,
        correctly_not_found=correctly_not_found,
        has_citation=has_cite,
    )


def compute_metrics(cases: list[CaseResult]) -> dict:
    """Aggregate per-case results into summary metrics."""
    answerable = [c for c in cases if c.answerable]
    unanswerable = [c for c in cases if not c.answerable]

    def ratio(num: int, denom: int) -> float:
        return round(num / denom, 3) if denom else 0.0

    retrieval_hits = sum(1 for c in answerable if c.retrieval_hit)
    answer_hits = sum(1 for c in answerable if c.keyword_match)
    faithful_judged = [c for c in answerable if c.faithful is not None]
    faithful_hits = sum(1 for c in faithful_judged if c.faithful)
    not_found_hits = sum(1 for c in unanswerable if c.correctly_not_found)
    citation_hits = sum(1 for c in answerable if c.has_citation)

    return {
        "retrieval_hit_rate": ratio(retrieval_hits, len(answerable)),
        "retrieval_hits": f"{retrieval_hits}/{len(answerable)}",
        "answer_accuracy": ratio(answer_hits, len(answerable)),
        "answer_hits": f"{answer_hits}/{len(answerable)}",
        "faithfulness": ratio(faithful_hits, len(faithful_judged)) if faithful_judged else None,
        "faithfulness_hits": f"{faithful_hits}/{len(faithful_judged)}" if faithful_judged else "skipped",
        "not_found_accuracy": ratio(not_found_hits, len(unanswerable)) if unanswerable else None,
        "not_found_hits": f"{not_found_hits}/{len(unanswerable)}" if unanswerable else "n/a",
        "citation_presence": ratio(citation_hits, len(answerable)),
        "citation_hits": f"{citation_hits}/{len(answerable)}",
    }


# ============================================================
# Reporting
# ============================================================

def print_report(report: Report) -> None:
    m = report.metrics
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"\nTotal cases:   {report.total}")
    print(f"Answerable:    {report.answerable}")
    print(f"Unanswerable:  {report.unanswerable}\n")

    print(f"Retrieval hit rate:    {m['retrieval_hits']:>6}   ({m['retrieval_hit_rate']:.0%})")
    print(f"Answer accuracy:       {m['answer_hits']:>6}   ({m['answer_accuracy']:.0%})")
    if m["faithfulness"] is not None:
        print(f"Faithfulness:          {m['faithfulness_hits']:>6}   ({m['faithfulness']:.0%})")
    else:
        print(f"Faithfulness:          skipped (--no-judge)")
    if m["not_found_accuracy"] is not None:
        print(f"Not-found accuracy:    {m['not_found_hits']:>6}   ({m['not_found_accuracy']:.0%})")
    print(f"Citation presence:     {m['citation_hits']:>6}   ({m['citation_presence']:.0%})")

    # Overall score (average of available metrics)
    scores = [m["retrieval_hit_rate"], m["answer_accuracy"], m["citation_presence"]]
    if m["faithfulness"] is not None:
        scores.append(m["faithfulness"])
    if m["not_found_accuracy"] is not None:
        scores.append(m["not_found_accuracy"])
    overall = sum(scores) / len(scores)
    print(f"\nOverall: {overall:.0%} across {len(scores)} metrics")
    print("=" * 60)

    # Failure detail
    failures = [c for c in report.cases if c.answerable and not (c.retrieval_hit and c.keyword_match)]
    failures += [c for c in report.cases if not c.answerable and not c.correctly_not_found]
    if failures:
        print("\nFailed cases:")
        for c in failures:
            print(f"  [{c.id}] {c.question}")
            print(f"       answer: {c.answer[:100]}...")
            print(f"       retrieved: {c.retrieved_source} pages {c.retrieved_pages}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge (saves API calls, faster)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N cases")
    args = parser.parse_args()

    if not TEST_SET.exists():
        print(f"Test set not found: {TEST_SET}", file=sys.stderr)
        return 1

    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.limit:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} test cases...")
    print(f"LLM-as-judge: {'off' if args.no_judge else 'on'}")

    embedder = Embedder()
    store = VectorStore()
    llm_client = get_client()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['id']}: {case['question'][:60]}")
        try:
            r = run_case(case, embedder, store, llm_client, run_judge=not args.no_judge)
            results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")

    metrics = compute_metrics(results)
    report = Report(
        total=len(results),
        answerable=sum(1 for c in results if c.answerable),
        unanswerable=sum(1 for c in results if not c.answerable),
        metrics=metrics,
        cases=results,
    )

    print_report(report)

    REPORT_JSON.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFull report saved to: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())