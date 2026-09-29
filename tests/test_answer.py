"""Tests for RAG answer generation."""

from unittest.mock import MagicMock

from rag.answer import (
    NOT_FOUND_SENTENCE,
    _extract_cited_indices,
    _format_context,
    answer_question,
)


# ---------- Pure function tests ----------

def test_extract_cited_indices():
    assert _extract_cited_indices("Answer [1] and [2] here.") == {1, 2}
    assert _extract_cited_indices("No citations here.") == set()
    assert _extract_cited_indices("Multiple [1] [1] [3].") == {1, 3}


def test_format_context():
    results = [
        {"id": "a", "text": "hello", "metadata": {"source_name": "t.pdf", "page_number": 1}, "distance": 0.1},
        {"id": "b", "text": "world", "metadata": {"source_name": "t.pdf", "page_number": 2}, "distance": 0.2},
    ]
    ctx = _format_context(results)
    assert "[1] (source: t.pdf, page: 1)" in ctx
    assert "[2] (source: t.pdf, page: 2)" in ctx
    assert "hello" in ctx
    assert "world" in ctx


# ---------- Pipeline tests with mocked LLM ----------

def _fake_embedder():
    e = MagicMock()
    e.embed_one.return_value = [0.1] * 384
    return e


def _fake_store():
    s = MagicMock()
    s.query.return_value = [
        {
            "id": "employee_handbook.pdf:p2:c0",
            "text": "Employees receive 15 days of paid annual leave per calendar year.",
            "metadata": {"source_name": "employee_handbook.pdf", "page_number": 2},
            "distance": 0.2,
        },
        {
            "id": "employee_handbook.pdf:p2:c1",
            "text": "Leave accrues monthly at 1.25 days per month.",
            "metadata": {"source_name": "employee_handbook.pdf", "page_number": 2},
            "distance": 0.3,
        },
    ]
    return s


def _fake_llm(response: str):
    l = MagicMock()
    l.complete.return_value = response
    return l


def test_answer_returns_sources():
    llm = _fake_llm("Employees receive 15 days [1]. Accrues monthly [2].")
    result = answer_question(
        "How many vacation days?",
        embedder=_fake_embedder(),
        store=_fake_store(),
        llm_client=llm,
    )
    assert result.found is True
    assert len(result.sources) == 2
    assert {s.index for s in result.sources} == {1, 2}
    assert all(s.source_name == "employee_handbook.pdf" for s in result.sources)


def test_answer_only_cited_sources():
    llm = _fake_llm("Employees receive 15 days [1].")
    result = answer_question(
        "How many vacation days?",
        embedder=_fake_embedder(),
        store=_fake_store(),
        llm_client=llm,
    )
    assert len(result.sources) == 1
    assert result.sources[0].index == 1


def test_answer_handles_not_found():
    llm = _fake_llm(NOT_FOUND_SENTENCE)
    result = answer_question(
        "What is the maternity policy?",
        embedder=_fake_embedder(),
        store=_fake_store(),
        llm_client=llm,
    )
    assert result.found is False
    assert result.sources == []
    assert result.answer == NOT_FOUND_SENTENCE


def test_answer_with_empty_store():
    store = MagicMock()
    store.query.return_value = []
    result = answer_question(
        "Anything",
        embedder=_fake_embedder(),
        store=store,
        llm_client=_fake_llm("ignored"),
    )
    assert result.found is False
    assert result.answer == NOT_FOUND_SENTENCE
    assert result.sources == []