"""Tests for production hardening features."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------- Logging / request ID ----------

def test_response_includes_request_id(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 12


# ---------- Health probes ----------

def test_live_probe(client):
    r = client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_ready_probe(client):
    with patch("backend.main.get_store") as mock_store:
        mock_store.return_value.count.return_value = 7
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        assert r.json()["vector_store_count"] == 7


# ---------- Auth ----------

def test_auth_disabled_when_no_key(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    r = client.post("/query", json={"question": "test"})
    # If auth is disabled, we don't get a 401
    assert r.status_code != 401


def test_auth_rejects_missing_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.post("/query", json={"question": "test"})
    assert r.status_code == 401


def test_auth_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.post(
        "/query",
        json={"question": "test"},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401


# ---------- Upload validation ----------

def test_upload_rejects_unsupported_extension(client):
    files = {"file": ("malware.exe", b"\x00\x00", "application/octet-stream")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]