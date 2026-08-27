"""Structured trace: every decision the agent makes, as JSONL.

Analysis code reads the trace and never parses natural language.  The trace is
also what makes replay affordable -- the oracle sweep and the regret
decomposition would be unaffordable if every re-run had to re-hit the API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


class TraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, kind: str, **fields: Any) -> None:
        self._fh.write(json.dumps({"kind": kind, **fields},
                                  ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self): return self
    def __exit__(self, *e): self.close(); return False


def read_trace(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


class ResponseCache:
    """Content-addressed cache of frontier responses, so re-running an
    identical (arm, preset, seed, budget) replays instead of re-billing."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def get(self, payload: Any) -> Any | None:
        p = self.root / f"{self.key(payload)}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, payload: Any, response: Any) -> None:
        (self.root / f"{self.key(payload)}.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8")
