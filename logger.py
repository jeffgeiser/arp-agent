from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DecisionLogger:
    """Append-only JSONL decision log.

    Each request writes one entry at routing time and a second entry with the
    same request_id when it finishes (timing, status, or error filled in).
    """

    MAX_PENDING = 10_000

    def __init__(self, log_file: str) -> None:
        self._path = log_file
        self._file_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, dict] = {}

    def _write(self, entry: dict) -> None:
        with self._file_lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def record(
        self,
        *,
        request_id: str,
        selected_node: Optional[str],
        score: float,
        reason: str,
        fallback_used: bool,
        node_scores: Dict[str, float],
        node_exclusions: Dict[str, str],
        primary_constraint: Optional[str] = None,
        factors: Optional[Dict[str, float]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "ts": _utcnow(),
            "request_id": request_id,
            "selected_node": selected_node,
            "score": score,
            "reason": reason,
            "fallback_used": fallback_used,
            "node_scores": node_scores,
            "node_exclusions": node_exclusions,
            "primary_constraint": primary_constraint,
            "factors": factors,
            "ttft_ms": None,
            "total_ms": None,
            "status_code": None,
            "error": None,
        }
        if selected_node is not None:
            with self._pending_lock:
                if len(self._pending) >= self.MAX_PENDING:
                    # Never grow without bound if completions are lost.
                    self._pending.pop(next(iter(self._pending)))
                self._pending[request_id] = entry
        self._write(entry)

    def complete(
        self,
        request_id: str,
        *,
        ttft_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._pending_lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return
        updated = dict(entry)
        updated["ts"] = _utcnow()
        updated["ttft_ms"] = round(ttft_ms, 2) if ttft_ms is not None else None
        updated["total_ms"] = round(total_ms, 2) if total_ms is not None else None
        updated["status_code"] = status_code
        updated["error"] = error
        self._write(updated)

    # Backwards-compatible name used by v0.1 callers.
    def update_timing(self, request_id: str, ttft_ms: Optional[float], total_ms: Optional[float]) -> None:
        self.complete(request_id, ttft_ms=ttft_ms, total_ms=total_ms)

    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)
