"""Watch data/raw/ for changes and trigger incremental reindexing.

Usage:
    python -m scripts.watch_and_reindex

Press Ctrl+C to stop.
"""

import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from rag.incremental import run_incremental, print_summary


RAW_DIR = Path("data/raw")
DEBOUNCE_SECONDS = 2.0   # wait this long after last event before reindexing


class Handler(FileSystemEventHandler):
    def __init__(self):
        self._last_event = 0.0

    def _maybe_reindex(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = event.src_path
        if not src.lower().endswith(".pdf"):
            return

        now = time.time()
        if now - self._last_event < DEBOUNCE_SECONDS:
            return
        self._last_event = now

        print(f"\n[watcher] change detected: {src}")
        print("[watcher] running incremental reindex...")
        result = run_incremental(force=False, dry_run=False)
        print_summary(result)

    def on_created(self, event):
        self._maybe_reindex(event)

    def on_modified(self, event):
        self._maybe_reindex(event)

    def on_moved(self, event):
        self._maybe_reindex(event)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[watcher] watching {RAW_DIR.resolve()} for PDF changes")
    print("[watcher] press Ctrl+C to stop\n")

    observer = Observer()
    observer.schedule(Handler(), str(RAW_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[watcher] stopping...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()