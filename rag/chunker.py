"""Split documents into overlapping chunks for embedding."""

import re
from dataclasses import dataclass, asdict
from typing import Any


# Sentence boundary: after . ! or ? followed by whitespace
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""
    chunk_id: str          # unique ID, e.g. "employee_handbook.pdf:p2:c0"
    text: str              # the actual content
    source_name: str       # original filename
    page_number: int       # which page it came from
    chunk_index: int       # index within the page (0-based)
    char_count: int        # length of text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences."""
    parts = SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack_sentences(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Pack sentences into chunks of ~chunk_size chars with overlap.

    Strategy:
      - Add sentences to the current chunk until adding another would exceed chunk_size
      - Finalize the chunk
      - Start the next chunk with the last few sentences of the previous one
        (so we get overlap)
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # Would adding this sentence overflow the current chunk?
        if current and current_len + sent_len + 1 > chunk_size:
            chunks.append(" ".join(current))

            # Build overlap from the end of the current chunk
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_page(
    text: str,
    source_name: str,
    page_number: int,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Split one page of text into chunks."""
    # Split into paragraphs, then sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(_split_into_sentences(para))

    raw_chunks = _pack_sentences(sentences, chunk_size, overlap)

    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                chunk_id=f"{source_name}:p{page_number}:c{i}",
                text=chunk_text,
                source_name=source_name,
                page_number=page_number,
                chunk_index=i,
                char_count=len(chunk_text),
            )
        )
    return chunks


def chunk_document(doc_dict: dict, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """
    Chunk a full Document (as dict from ingestion pipeline).

    Args:
        doc_dict: serialized Document with 'source_name' and 'pages'
    """
    source_name = doc_dict["source_name"]
    all_chunks: list[Chunk] = []
    for page in doc_dict["pages"]:
        all_chunks.extend(
            chunk_page(
                text=page["text"],
                source_name=source_name,
                page_number=page["page_number"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return all_chunks


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load processed document
    doc_path = Path("data/processed/employee_handbook.json")
    doc_dict = json.loads(doc_path.read_text(encoding="utf-8"))

    chunks = chunk_document(doc_dict)
    print(f"Total chunks: {len(chunks)}")
    print("=" * 60)

    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        print(f"[{c.chunk_id}] ({c.char_count} chars) {preview}...")