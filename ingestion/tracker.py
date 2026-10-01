"""Track document hashes to detect changes between indexing runs."""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_STATE_FILE = Path("data/processed/.index_state.json")


@dataclass
class DocumentHash:
    source_name: str
    file_hash: str          # SHA-256 of raw PDF bytes
    processed_hash: str     # SHA-256 of extracted text (changes if cleaner changes)
    indexed_at: str         # ISO timestamp
    chunk_count: int


class ChangeTracker:
    """
    Persists per-document hashes so we can detect changes.

    State file is a simple JSON dict keyed by source_name.
    """

    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                self._state = {}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def hash_file(path: Path) -> str:
        """SHA-256 of raw file bytes."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def has_changed(self, source_name: str, file_hash: str) -> bool:
        """Return True if the file is new OR its hash differs from stored."""
        stored = self._state.get(source_name)
        if stored is None:
            return True
        return stored.get("file_hash") != file_hash

    def record(
        self,
        source_name: str,
        file_hash: str,
        processed_hash: str,
        chunk_count: int,
    ) -> None:
        from datetime import datetime, timezone

        self._state[source_name] = asdict(
            DocumentHash(
                source_name=source_name,
                file_hash=file_hash,
                processed_hash=processed_hash,
                indexed_at=datetime.now(timezone.utc).isoformat(),
                chunk_count=chunk_count,
            )
        )
        self._save()

    def forget(self, source_name: str) -> None:
        if source_name in self._state:
            del self._state[source_name]
            self._save()

    def known_sources(self) -> list[str]:
        return sorted(self._state.keys())

    def summary(self) -> dict[str, Any]:
        return {
            "tracked_documents": len(self._state),
            "sources": self.known_sources(),
        }


if __name__ == "__main__":
    t = ChangeTracker()
    print(json.dumps(t.summary(), indent=2))