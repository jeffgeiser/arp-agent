from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


def score_node(health: dict, cfg: dict) -> float:
    thermal = health["thermal_headroom_pct"] / 100
    vram = health["vram_headroom_pct"] / 100
    tok_w = max(0, 1 + health["wes_delta_pct"] / 100)

    w = cfg["scoring"]
    score = (
        thermal * w["thermal_headroom_weight"]
        + vram * w["vram_headroom_weight"]
        + tok_w * w["tok_w_delta_weight"]
    )

    if health["wes_delta_pct"] < w["throttle_threshold_pct"]:
        return 0.0
    if health["status"] != "healthy":
        return 0.0
    if health["vram_headroom_pct"] < 10:
        return 0.0

    return round(score, 4)


def exclusion_reason(health: dict, cfg: dict) -> Optional[str]:
    w = cfg["scoring"]
    if health["status"] != "healthy":
        return "unhealthy"
    if health["wes_delta_pct"] < w["throttle_threshold_pct"]:
        return "throttling"
    if health["vram_headroom_pct"] < 10:
        return "vram_low"
    return None


@dataclass
class NodeState:
    node_id: str
    wicklee_url: str
    inference_url: str
    health: Optional[dict] = None
    score: float = 0.0
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def update(self, health: dict, cfg: dict) -> None:
        with self._lock:
            self.health = health
            self.score = score_node(health, cfg)
            self.last_error = None

    def set_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error
            self.score = 0.0

    def get_score(self) -> float:
        with self._lock:
            return self.score

    def get_health(self) -> Optional[dict]:
        with self._lock:
            return dict(self.health) if self.health else None
