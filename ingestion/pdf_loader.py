"""Load raw text from PDF files, preserving per-page structure."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass
class Page:
    """A single page extracted from a PDF."""
    page_number: int      # 1-indexed (readers count from 1, not 0)
    text: str             # raw extracted text
    char_count: int       # length of text, for sanity checks

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return asdict(self)


@dataclass
class Document:
    """A PDF document split into pages."""
    source_path: str      # original file path (for citations)
    source_name: str      # just the filename (nicer for display)
    total_pages: int      # how many pages were extracted
    pages: list[Page]     # the actual content

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_name": self.source_name,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
        }


def extract_document(pdf_path: Path) -> Document:
    """
    Extract a PDF into a structured Document with per-page text.

    Raises:
        FileNotFoundError: if the PDF doesn't exist
        ValueError: if the PDF can't be read (corrupt, encrypted, etc.)
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF {pdf_path}: {e}") from e

    pages: list[Page] = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        # extract_text() can return None for image-only pages
        raw_text = pdf_page.extract_text() or ""
        pages.append(
            Page(
                page_number=i,
                text=raw_text,
                char_count=len(raw_text),
            )
        )

    return Document(
        source_path=str(pdf_path),
        source_name=pdf_path.name,
        total_pages=len(pages),
        pages=pages,
    )


if __name__ == "__main__":
    pdf = Path("data/raw/employee_handbook.pdf")
    doc = extract_document(pdf)

    print("=" * 60)
    print(f"Document: {doc.source_name}")
    print(f"Total pages: {doc.total_pages}")
    print("=" * 60)

    for page in doc.pages:
        print(f"\n--- Page {page.page_number} ({page.char_count} chars) ---")
        # Print first 200 chars so output stays readable
        preview = page.text[:200].replace("\n", "\\n")
        print(preview)

    total_chars = sum(p.char_count for p in doc.pages)
    print(f"\n{'=' * 60}")
    print(f"Total characters across all pages: {total_chars}")