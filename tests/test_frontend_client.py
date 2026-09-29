"""Tests for the frontend's HTTP client functions.

We don't test Streamlit UI rendering (that's a separate tool).
We test the HTTP interactions the frontend performs.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests


# Import the functions we want to test
# They live inside frontend/app.py, but importing that runs Streamlit code.
# So we replicate the logic here for clarity, OR import carefully.

# Instead, we test the API contract the frontend relies on.
# This is an integration-style test that requires a running backend,
# skipped if unavailable.

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _backend_alive() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=1).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _backend_alive(),
    reason="FastAPI backend not running on localhost:8000",
)


def test_health_endpoint():
    r = requests.get(f"{API_URL}/health", timeout=3)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "vector_store_count" in data
    assert "llm_provider" in data


def test_query_endpoint_returns_expected_shape():
    r = requests.post(
        f"{API_URL}/query",
        json={"question": "How many vacation days do I get?", "top_k": 3},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "question" in data
    assert "answer" in data
    assert "found" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_query_endpoint_rejects_empty_question():
    r = requests.post(
        f"{API_URL}/query",
        json={"question": "", "top_k": 3},
        timeout=10,
    )
    assert r.status_code == 422