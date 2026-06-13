from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Dict, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DecisionLogger:
    def __init__(self, log_file: str) -> None:
        self._path = log_file
        self._file_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, dict] = {}

    def _write(self, entry: dict) -> None:
        with self._file_lock:
            with open(self._path, "a") as f:
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
    ) -> None:
        entry = {
            "ts": _utcnow(),
            "request_id": request_id,
            "selected_node": selected_node,
            "score": score,
            "reason": reason,
            "fallback_used": fallback_used,
            "node_scores": node_scores,
            "node_exclusions": node_exclusions,
            "ttft_ms": None,
            "total_ms": None,
        }
        with self._pending_lock:
            self._pending[request_id] = entry
        self._write(entry)

    def update_timing(
        self,
        request_id: str,
        ttft_ms: Optional[float],
        total_ms: Optional[float],
    ) -> None:
        with self._pending_lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return
        updated = dict(entry)
        updated["ttft_ms"] = round(ttft_ms, 2) if ttft_ms is not None else None
        updated["total_ms"] = round(total_ms, 2) if total_ms is not None else None
        self._write(updated)
