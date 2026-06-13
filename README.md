# arp-agent

A reference implementation of the [ARP (Agentic Resource Protocol)](https://github.com/jeffgeiser/arp-spec) routing agent.

`arp-agent` is a lightweight Python proxy that:

- Polls each inference node's Wicklee `/health` endpoint every 100 ms
- Scores each node using the ARP scoring function (thermal headroom, VRAM headroom, tok/W delta)
- Routes incoming OpenAI-compatible requests to the highest-scoring eligible node
- Logs every routing decision as a JSONL entry for downstream analysis

## Quick Start

```bash
pip install -r requirements.txt
python agent.py config.json
```

Send any OpenAI-compatible request to `http://localhost:8090`:

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-q4_k_m", "messages": [{"role": "user", "content": "hello"}]}'
```

## Configuration

```json
{
  "listen_port": 8090,
  "nodes": [
    {
      "id": "node-1",
      "wicklee_url": "http://localhost:8080",
      "inference_url": "http://localhost:8000"
    }
  ],
  "scoring": {
    "thermal_headroom_weight": 0.4,
    "vram_headroom_weight": 0.3,
    "tok_w_delta_weight": 0.3,
    "throttle_threshold_pct": -15,
    "poll_interval_ms": 100
  },
  "fallback": "round_robin",
  "log_file": "decisions.jsonl"
}
```

| Field | Description |
|-------|-------------|
| `listen_port` | Port the agent listens on |
| `nodes[].wicklee_url` | URL of the Wicklee health daemon on that node |
| `nodes[].inference_url` | URL of the inference server (e.g. vLLM) on that node |
| `scoring.thermal_headroom_weight` | Weight for thermal headroom signal |
| `scoring.vram_headroom_weight` | Weight for VRAM headroom signal |
| `scoring.tok_w_delta_weight` | Weight for tok/W efficiency delta signal |
| `scoring.throttle_threshold_pct` | Nodes below this `wes_delta_pct` are hard-excluded |
| `scoring.poll_interval_ms` | Health poll interval in ms (default: 100) |
| `fallback` | `"round_robin"` — used when no node scores above zero |
| `log_file` | Path to the JSONL decision log |

## Scoring

Each node is scored as:

```
score = (thermal_headroom_pct / 100) * thermal_headroom_weight
      + (vram_headroom_pct / 100)    * vram_headroom_weight
      + max(0, 1 + wes_delta_pct/100) * tok_w_delta_weight
```

Nodes are hard-excluded (score = 0) when:
- `status != "healthy"`
- `wes_delta_pct < throttle_threshold_pct`
- `vram_headroom_pct < 10`

When all nodes score zero, the agent falls back to round-robin over nodes that have any health data.

## Decision Log

Every request appends one JSONL entry to `decisions.jsonl` at routing time, then a second entry with the same `request_id` after the response completes with timing filled in:

```json
{"ts":"2026-06-15T09:01:23Z","request_id":"req-abc123","selected_node":"node-1","score":0.847,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.847,"node-2":0.0},"node_exclusions":{"node-2":"throttling"},"ttft_ms":null,"total_ms":null}
{"ts":"2026-06-15T09:01:26Z","request_id":"req-abc123","selected_node":"node-1","score":0.847,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.847,"node-2":0.0},"node_exclusions":{"node-2":"throttling"},"ttft_ms":312.4,"total_ms":2841.7}
```

`reason` values:
- `highest_score` — normal path, best-scoring node selected
- `fallback_round_robin` — all nodes scored zero, fell back to round-robin
- `no_eligible_node` — no nodes had health data and fallback is disabled

## File Structure

```
agent.py        — startup: wires poller + proxy + logger, runs uvicorn
scorer.py       — ARP scoring function and NodeState
poller.py       — async background health polling loop
proxy.py        — OpenAI-compatible FastAPI proxy with score-based routing
logger.py       — thread-safe JSONL decision logger with timing updates
config.json     — example single-node configuration
requirements.txt
examples/
  two_node_setup.md
```

## Requirements

- Python 3.10+
- `httpx`, `fastapi`, `uvicorn`

## ARP Spec

This agent implements the routing layer described in the [ARP spec](https://github.com/jeffgeiser/arp-spec). The Reconcile flywheel (weight adaptation from outcomes) is v0.2 — this codebase writes the decision log that will train it.

## License

MIT
