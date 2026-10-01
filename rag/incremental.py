"""Incremental reindexing: only process documents that changed.

Usage:
    python -m rag.incremental              # scan data/processed/, reindex changes
    python -m rag.incremental --force      # full rebuild regardless of hashes
    python -m rag.incremental --dry-run    # report what would change, don't do it
"""

import argparse
import json
import sys
from pathlib import Path

from ingestion.pdf_loader import extract_document
from ingestion.pipeline import process_pdf, save_document
from ingestion.tracker import ChangeTracker
from rag.chunker import chunk_document
from rag.embedder import Embedder
from rag.pipeline import index_document
from rag.vector_store import VectorStore


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def scan_raw_pdfs() -> list[Path]:
    """All PDFs in data/raw/ (excluding hidden files)."""
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.glob("*.pdf") if not p.name.startswith("."))


def ensure_processed(pdf_path: Path, tracker: ChangeTracker, force: bool = False) -> tuple[Path, bool]:
    """
    Make sure data/processed/<stem>.json exists and is up to date with the PDF.

    Returns (json_path, was_reprocessed).
    """
    json_path = PROCESSED_DIR / f"{pdf_path.stem}.json"
    file_hash = ChangeTracker.hash_file(pdf_path)

    needs_processing = force or not json_path.exists() or tracker.has_changed(pdf_path.name, file_hash)

    if not needs_processing:
        return json_path, False

    doc = process_pdf(pdf_path)
    save_document(doc, PROCESSED_DIR)
    return json_path, True


def delete_chunks_for_source(store: VectorStore, source_name: str) -> int:
    """
    Remove all chunks belonging to a source from the vector store.
    Returns the number of chunks deleted.
    """
    collection = store.collection
    existing = collection.get(where={"source_name": source_name})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def run_incremental(
    force: bool = False,
    dry_run: bool = False,
    state_file: Path | None = None,
) -> dict:
    """Main pipeline. Returns a summary dict.

    Args:
        force: reindex everything, ignoring stored hashes
        dry_run: report changes without modifying the store
        state_file: optional custom path for the tracker's state
                    (mainly for tests; defaults to data/processed/.index_state.json)
    """
    tracker = ChangeTracker(state_file=state_file) if state_file else ChangeTracker()
    embedder = Embedder()
    store = VectorStore()

    pdfs = scan_raw_pdfs()
    if not pdfs:
        return {"error": "No PDFs found in data/raw/", "processed": [], "skipped": []}

    summary = {
        "total_pdfs": len(pdfs),
        "new_or_changed": [],
        "unchanged": [],
        "deleted_chunks": 0,
        "indexed_chunks": 0,
    }

    for pdf in pdfs:
        file_hash = ChangeTracker.hash_file(pdf)
        changed = force or tracker.has_changed(pdf.name, file_hash)

        if not changed:
            summary["unchanged"].append(pdf.name)
            continue

        summary["new_or_changed"].append(pdf.name)
        if dry_run:
            continue

        # 1. Ensure processed JSON is up to date
        json_path, _ = ensure_processed(pdf, tracker, force=force)

        # 2. Remove stale chunks from the vector store
        n_deleted = delete_chunks_for_source(store, pdf.name)
        summary["deleted_chunks"] += n_deleted

        # 3. Re-index from the fresh JSON
        doc_dict = json.loads(json_path.read_text(encoding="utf-8"))
        chunks = chunk_document(doc_dict)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
        summary["indexed_chunks"] += len(chunks)

        # 4. Record the new hash
        processed_text = "\n".join(p["text"] for p in doc_dict["pages"])
        tracker.record(
            source_name=pdf.name,
            file_hash=file_hash,
            processed_hash=ChangeTracker.hash_text(processed_text),
            chunk_count=len(chunks),
        )

    return summary


def print_summary(s: dict) -> None:
    if "error" in s:
        print(f"ERROR: {s['error']}")
        return

    print("\n" + "=" * 60)
    print("Incremental Index Summary")
    print("=" * 60)
    print(f"Total PDFs scanned:     {s['total_pdfs']}")
    print(f"New or changed:         {len(s['new_or_changed'])}")
    for name in s["new_or_changed"]:
        print(f"  + {name}")
    print(f"Unchanged (skipped):    {len(s['unchanged'])}")
    for name in s["unchanged"]:
        print(f"  - {name}")
    print(f"Deleted old chunks:     {s['deleted_chunks']}")
    print(f"Indexed new chunks:     {s['indexed_chunks']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Reindex everything, ignoring stored hashes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes without modifying the store")
    args = parser.parse_args()

    result = run_incremental(force=args.force, dry_run=args.dry_run)
    print_summary(result)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())