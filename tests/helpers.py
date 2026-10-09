"""Shared fixtures: a mock Wicklee health payload and config."""
from __future__ import annotations

import copy

BASE_CFG = {
    "listen_port": 8090,
    "nodes": [],
    "scoring": {"poll_interval_ms": 100, "stale_after_ms": 2000},
    "fallback": "round_robin",
}


def health(**overrides):
    """A healthy node, shaped like Wicklee's /health (see examples/two_node_setup.md)."""
    h = {
        "node_id": "node-1",
        "model_loaded": "qwen2.5-7b-q4_k_m",
        "wes_score": 8.4,
        "wes_delta_pct": 0.0,
        "junction_temp_c": 60.0,
        "thermal_headroom_pct": 33.0,
        "vram_used_gb": 5.0,
        "vram_total_gb": 16.0,
        "vram_headroom_pct": 68.75,
        "status": "healthy",
    }
    h.update(overrides)
    return {k: v for k, v in h.items() if v is not ...}


def cfg(**scoring):
    c = copy.deepcopy(BASE_CFG)
    c["scoring"].update(scoring)
    return c
