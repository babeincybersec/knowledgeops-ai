"""Tests for the ingestion pipeline."""

from pathlib import Path

import pytest

from ingestion.pdf_loader import extract_document
from ingestion.cleaner import clean_text
from ingestion.pipeline import process_pdf, save_document


SAMPLE_PDF = Path("data/raw/employee_handbook.pdf")


# ---------- Cleaner tests ----------

def test_remove_page_footer():
    text = "Content here.\nPage 1 of 3\nMore content."
    result = clean_text(text)
    assert "Page 1 of 3" not in result
    assert "Content here." in result
    assert "More content." in result


def test_remove_standalone_number():
    text = "Some text.\n12\nMore text."
    result = clean_text(text)
    assert "\n12\n" not in result
    assert "Some text." in result
    assert "More text." in result


def test_fix_hyphenation():
    text = "This is a vaca-\ntion day."
    result = clean_text(text)
    assert "vacation" in result
    assert "vaca-" not in result


def test_preserve_content_numbers():
    """Real numbers in content must NOT be removed."""
    text = "Employees receive 15 days of leave."
    result = clean_text(text)
    assert "15 days" in result


# ---------- Loader tests ----------

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not present")
def test_extract_document_structure():
    doc = extract_document(SAMPLE_PDF)
    assert doc.total_pages == 3
    assert doc.source_name == "employee_handbook.pdf"
    assert len(doc.pages) == 3
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].char_count > 0


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not present")
def test_extract_contains_expected_content():
    """Spot-check the actual content of the PDF."""
    doc = extract_document(SAMPLE_PDF)
    full_text = "\n".join(p.text for p in doc.pages)

    assert "15 days" in full_text
    assert "1.25 days per month" in full_text
    assert "12 characters" in full_text
    assert "90 days" in full_text


def test_extract_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_document(Path("data/raw/does_not_exist.pdf"))


# ---------- Pipeline tests ----------

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not present")
def test_process_removes_footers():
    doc = process_pdf(SAMPLE_PDF)
    full_text = "\n".join(p.text for p in doc.pages)
    assert "Page 1 of 3" not in full_text
    assert "Page 2 of 3" not in full_text
    assert "Page 3 of 3" not in full_text


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not present")
def test_save_document(tmp_path):
    doc = process_pdf(SAMPLE_PDF)
    out = save_document(doc, tmp_path)
    assert out.exists()
    assert out.suffix == ".json"
    content = out.read_text(encoding="utf-8")
    assert '"source_name": "employee_handbook.pdf"' in content