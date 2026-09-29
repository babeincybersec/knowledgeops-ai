"""Tests for the FastAPI backend."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_answer():
    """Patch answer_question and store so no LLM/DB is needed."""
    from rag.answer import Answer, Source

    fake_answer = Answer(
        question="How many vacation days?",
        answer="Employees receive 15 days [1].",
        found=True,
        sources=[
            Source(
                index=1,
                chunk_id="test:p2:c0",
                source_name="employee_handbook.pdf",
                page_number=2,
                text="Employees receive 15 days of paid annual leave.",
            )
        ],
    )

    with patch("backend.main.answer_question", return_value=fake_answer):
        with patch("backend.main.get_store") as mock_store:
            mock_store.return_value.count.return_value = 7
            with patch("backend.main.get_embedder"):
                with patch("backend.main.get_llm"):
                    yield


# ---------- Health ----------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store_count" in data
    assert "llm_provider" in data


# ---------- Query ----------

def test_query_returns_answer(client, mock_answer):
    response = client.post(
        "/query",
        json={"question": "How many vacation days do I get?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert "15 days" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page_number"] == 2


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422   # Pydantic validation error


def test_query_rejects_missing_question(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


# ---------- Documents ----------

def test_list_documents(client, mock_answer):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_chunks" in data