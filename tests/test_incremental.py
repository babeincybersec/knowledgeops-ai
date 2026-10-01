"""Tests for incremental indexing and change tracking."""

from pathlib import Path

import pytest

from ingestion.tracker import ChangeTracker


# ---------- Tracker tests ----------

def test_tracker_starts_empty(tmp_path):
    tracker = ChangeTracker(state_file=tmp_path / "state.json")
    assert tracker.known_sources() == []


def test_tracker_records_and_persists(tmp_path):
    state_file = tmp_path / "state.json"
    t1 = ChangeTracker(state_file=state_file)
    t1.record(
        source_name="doc.pdf",
        file_hash="abc123",
        processed_hash="def456",
        chunk_count=5,
    )
    assert "doc.pdf" in t1.known_sources()
    assert state_file.exists()

    # New tracker instance loads the persisted state
    t2 = ChangeTracker(state_file=state_file)
    assert "doc.pdf" in t2.known_sources()
    assert t2.has_changed("doc.pdf", "abc123") is False
    assert t2.has_changed("doc.pdf", "different") is True


def test_tracker_unknown_doc_is_changed(tmp_path):
    t = ChangeTracker(state_file=tmp_path / "state.json")
    assert t.has_changed("never_seen.pdf", "anyhash") is True


def test_tracker_forget(tmp_path):
    t = ChangeTracker(state_file=tmp_path / "state.json")
    t.record("a.pdf", "h1", "p1", 3)
    t.record("b.pdf", "h2", "p2", 4)
    assert len(t.known_sources()) == 2
    t.forget("a.pdf")
    assert t.known_sources() == ["b.pdf"]


def test_tracker_hash_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = ChangeTracker.hash_file(f)
    assert len(h1) == 64

    f.write_text("hello world!")
    h2 = ChangeTracker.hash_file(f)
    assert h1 != h2


def test_tracker_hash_text_stable():
    h1 = ChangeTracker.hash_text("some content")
    h2 = ChangeTracker.hash_text("some content")
    h3 = ChangeTracker.hash_text("different")
    assert h1 == h2
    assert h1 != h3


# ---------- End-to-end change detection ----------

@pytest.mark.skipif(
    not Path("data/raw/employee_handbook.pdf").exists(),
    reason="Sample PDF missing",
)
def test_second_run_skips_unchanged(tmp_path):
    """Full pipeline: first run indexes, second run skips."""
    from rag.incremental import run_incremental

    # Isolated state file so we don't touch the real one
    state_file = tmp_path / "state.json"

    # First run: fresh state file → should detect the PDF as new
    r1 = run_incremental(force=False, dry_run=False, state_file=state_file)
    assert len(r1["new_or_changed"]) >= 1, (
        f"Expected at least 1 change on first run, got: {r1}"
    )

    # Second run: same state file → hashes match → should skip
    r2 = run_incremental(force=False, dry_run=False, state_file=state_file)
    assert len(r2["new_or_changed"]) == 0, (
        f"Expected 0 changes on second run, got: {r2}"
    )
    assert len(r2["unchanged"]) >= 1