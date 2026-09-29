"""RAG answer generation: retrieve chunks, prompt the LLM, return answer + sources."""

import re
from dataclasses import dataclass, asdict
from typing import Any

from rag.embedder import Embedder
from rag.llm import get_client
from rag.vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful assistant answering questions about company documentation.

RULES:
1. Answer ONLY using the information in the CONTEXT provided by the user.
2. Every fact in your answer must come from the context. Do not use outside knowledge.
3. 3. Cite sources using ASCII square brackets like [1] or [2] matching the CONTEXT
   chunks. Use ASCII brackets only — never fullwidth brackets like 【1】.
4. If the context does not contain the answer, respond with exactly this sentence:
   "I couldn't find this information in the available documentation."
5. Do not invent policies, names, dates, numbers, or references.
6. Be concise. Prefer quoting the source over paraphrasing."""


NOT_FOUND_SENTENCE = "I couldn't find this information in the available documentation."

def _normalize_citations(text: str) -> str:
    """
    Convert fullwidth/curly citation brackets to ASCII square brackets.

    Some models (GPT-OSS, Qwen) produce 【1】 or ｢1｣ instead of [1].
    We normalize so citation parsing and display are consistent.
    """
    return (
        text.replace("【", "[").replace("】", "]")   # fullwidth
            .replace("｢", "[").replace("｣", "]")   # halfwidth corner
            .replace("〔", "[").replace("〕", "]")   # tortoise shell
    )
    

@dataclass
class Source:
    """A source cited in the answer."""
    index: int
    chunk_id: str
    source_name: str
    page_number: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """The full result of a RAG query."""
    question: str
    answer: str
    sources: list[Source]
    found: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "found": self.found,
            "sources": [s.to_dict() for s in self.sources],
        }


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks."""
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        header = f"[{i}] (source: {meta['source_name']}, page: {meta['page_number']})"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, context: str) -> str:
    return f"""CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _extract_cited_indices(answer_text: str) -> set[int]:
    """Find all [N] markers in the answer and return the set of indices."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}


def answer_question(
    question: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm_client: Any | None = None,
) -> Answer:
    """
    Full RAG pipeline:
      1. Embed the question
      2. Retrieve top_k chunks
      3. Format context
      4. Call LLM
      5. Parse cited sources
    """
    embedder = embedder or Embedder()
    store = store or VectorStore()
    llm_client = llm_client or get_client()

    # Step 1 + 2: retrieve
    query_vec = embedder.embed_one(question)
    retrieved = store.query(query_vec, top_k=top_k)

    if not retrieved:
        return Answer(
            question=question,
            answer=NOT_FOUND_SENTENCE,
            sources=[],
            found=False,
        )

    # Step 3: build prompt
    context = _format_context(retrieved)
    user_prompt = _build_user_prompt(question, context)

    # Step 4: generate
           # Step 4: generate + normalize citation brackets
    raw_answer = _normalize_citations(
        llm_client.complete(SYSTEM_PROMPT, user_prompt).strip()
    )
    
    # Did the model say it couldn't find it?
    if NOT_FOUND_SENTENCE.lower() in raw_answer.lower():
        return Answer(
            question=question,
            answer=NOT_FOUND_SENTENCE,
            sources=[],
            found=False,
        )

    # Step 5: map cited [N] back to sources
    cited = _extract_cited_indices(raw_answer)
    sources: list[Source] = []
    for i, r in enumerate(retrieved, start=1):
        if i in cited:
            meta = r["metadata"]
            sources.append(
                Source(
                    index=i,
                    chunk_id=r["id"],
                    source_name=meta["source_name"],
                    page_number=meta["page_number"],
                    text=r["text"],
                )
            )

    return Answer(
        question=question,
        answer=raw_answer,
        sources=sources,
        found=True,
    )


def format_answer(a: Answer) -> str:
    """Pretty-print an Answer for the terminal."""
    lines = [a.answer, ""]
    if a.sources:
        lines.append("Sources:")
        for s in a.sources:
            preview = s.text[:120].replace("\n", " ")
            lines.append(f"  [{s.index}] {s.source_name}, page {s.page_number}")
            lines.append(f'      "{preview}..."')
    else:
        lines.append("(no sources cited)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m rag.answer "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer_question(question)
    print(f"\nQ: {result.question}\n")
    print(format_answer(result))