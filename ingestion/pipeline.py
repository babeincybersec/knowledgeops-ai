"""Orchestrate: PDF → structured pages → cleaned → JSON."""

import json
from pathlib import Path

from ingestion.cleaner import clean_text
from ingestion.pdf_loader import Document, extract_document


def process_pdf(pdf_path: Path) -> Document:
    """Extract and clean a PDF, returning a Document."""
    doc = extract_document(pdf_path)
    for page in doc.pages:
        page.text = clean_text(page.text)
        page.char_count = len(page.text)
    return doc


def save_document(doc: Document, output_dir: Path) -> Path:
    """Save a processed Document as JSON. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use the same stem as the source PDF
    stem = Path(doc.source_name).stem
    output_path = output_dir / f"{stem}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    source = Path("data/raw/employee_handbook.pdf")
    output_dir = Path("data/processed")

    print(f"Processing: {source}")
    doc = process_pdf(source)
    print(f"  Pages: {doc.total_pages}")
    print(f"  Total chars after cleaning: {sum(p.char_count for p in doc.pages)}")

    out = save_document(doc, output_dir)
    print(f"Saved: {out}")

    # Show first 300 chars of page 2 as a sanity check
    if doc.total_pages >= 2:
        print("\n--- Page 2 preview (cleaned) ---")
        print(doc.pages[1].text[:300])