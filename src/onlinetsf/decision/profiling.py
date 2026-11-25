"""Buffered, parent-only wall-clock diagnostics for hindsight search."""

import json
from pathlib import Path
from time import perf_counter


class SearchProfile:
    def __init__(self, path: Path) -> None:
        self.file = path.open("w", encoding="utf-8", buffering=64 * 1024)
        self.started = self.last_flush = perf_counter()

    def record(self, event: str, **values) -> None:
        now = perf_counter()
        self.file.write(json.dumps({"event": event, "elapsed_seconds": now - self.started, **values},
                                   ensure_ascii=False, allow_nan=False) + "\n")
        # Phase changes and the first item of a high-frequency stream are visible
        # immediately. Other items share a buffer; no fsync, worker logging or device barriers.
        buffered = event in {"segment", "fill_origin", "collect_origin"}
        first_item = values.get("segment") == 1 or values.get("step") == 1
        if not buffered or first_item or now - self.last_flush >= 30:
            self.file.flush()
            self.last_flush = now

    def close(self) -> None:
        self.file.close()
