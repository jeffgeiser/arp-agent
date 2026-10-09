# arp-agent

A reference implementation of the [ARP (Agentic Resource Protocol)](https://github.com/jeffgeiser/arp-spec) routing agent.

`arp-agent` is a lightweight Python proxy that:

- Polls each inference node's Wicklee `/health` endpoint (default every 100 ms, one independent loop per node)
- Scores each node with the ARP v0.1 Score function from [`spec/02-score.md`](https://github.com/jeffgeiser/arp-spec/blob/main/spec/02-score.md)
- Routes incoming OpenAI-compatible requests to the highest-scoring eligible node, passing the upstream status code, headers and body through unchanged (streaming included)
- Logs every routing decision as a JSONL entry for downstream analysis (the input to Reconcile)

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

The agent listens on `127.0.0.1` by default and has no authentication of its own; set `listen_host` to `0.0.0.0` only on a trusted network.

## Configuration

```json
{
  "listen_host": "127.0.0.1",
  "listen_port": 8090,
  "nodes": [
    {
      "id": "node-1",
      "wicklee_url": "http://localhost:8080",
      "inference_url": "http://localhost:8000"
    }
  ],
  "scoring": {
    "weights": {"thermal_headroom": 0.30, "vram_fit": 0.30, "wes_history": 0.25, "reliability": 0.15},
    "throttle_c": 90,
    "target_margin_c": 30,
    "vram_baseline_gb": 1.0,
    "throttle_threshold_pct": -15,
    "min_vram_headroom_pct": 10,
    "poll_interval_ms": 100,
    "stale_after_ms": 2000
  },
  "models": {},
  "fallback": "round_robin",
  "log_file": "decisions.jsonl"
}
```

| Field | Description |
|-------|-------------|
| `listen_host` / `listen_port` | Where the agent listens (default `127.0.0.1`) |
| `nodes[].wicklee_url` | URL of the Wicklee health daemon on that node |
| `nodes[].inference_url` | URL of the inference server (e.g. vLLM) on that node |
| `scoring.weights` | Factor weights. Defaults are the spec's v0.1 weights; change them only to experiment |
| `scoring.throttle_c` | Throttle threshold in °C (a node's `/health` may override it with `throttle_c`) |
| `scoring.target_margin_c` | Margin below throttle that counts as fully safe (spec default 30) |
| `scoring.vram_baseline_gb` | OS/driver VRAM overhead (spec default 1.0 GB for discrete GPUs, use 2.0 on Apple Silicon) |
| `scoring.throttle_threshold_pct` | Nodes below this `wes_delta_pct` are hard-excluded (agent policy) |
| `scoring.min_vram_headroom_pct` | Nodes below this VRAM headroom are hard-excluded (agent policy) |
| `scoring.poll_interval_ms` | Health poll interval per node |
| `scoring.stale_after_ms` | Health older than this is ignored; the node is excluded as `stale_health` |
| `scoring.reliability_window` | Recent proxied requests used for the reliability factor (default 50) |
| `models` | Optional `{"model name": required_gb}` used for `vram_fit` when a request names that model |
| `fallback` | `"round_robin"`: used when no node is eligible; only nodes with fresh health are candidates |
| `upstream_timeout_s` / `connect_timeout_s` | Upstream timeouts (defaults 120 s / 5 s) |
| `log_file` | Path to the JSONL decision log |

Configs from v0.1 that use `thermal_headroom_weight`, `vram_headroom_weight` and `tok_w_delta_weight` still load; those keys are ignored with a warning and the spec weights apply.

## Scoring

The score follows ARP v0.1 (`spec/02-score.md`):

```
score = 0.30 * thermal_headroom + 0.30 * vram_fit + 0.25 * wes_history + 0.15 * reliability
```

| Factor | How the agent computes it |
|--------|---------------------------|
| `thermal_headroom` | `clamp((throttle_c - junction_temp_c) / target_margin_c)` × the warming-trend penalty `clamp(1 - max(0, °C/min) / 5)`, with the trend measured over the last 60 s of polls. Falls back to `thermal_headroom_pct / 100` if no temperature is reported. |
| `vram_fit` | The spec's piecewise curve over `fill = required_gb / (vram_total_gb - vram_baseline_gb)`. `required_gb` comes from `models` when the request names a configured model, otherwise from the node's current `vram_used_gb` (the served model is already resident). Falls back to `1 - vram_headroom_pct / 100`. |
| `wes_history` | The spec's curve over `ratio = 1 + wes_delta_pct / 100` (Wicklee's delta against the node's baseline). Missing → `0.7`, the spec's cold-start value. |
| `reliability` | Fraction of the last `reliability_window` proxied requests that didn't fail (connection error, timeout or 5xx); `0.5` until there are 5. This stands in for Reconcile receipts until Reconcile exists. |

A factor whose inputs are missing gets a neutral `0.5` (`0.7` for `wes_history`) and is listed in the score's `missing` field instead of raising.

Nodes are hard-excluded (score = 0) when:
- health is missing (`no_health_data`) or older than `stale_after_ms` (`stale_health`)
- `status` is missing (`missing_status`) or not `"healthy"` (`unhealthy`)
- `wes_delta_pct < throttle_threshold_pct` (`throttling`)
- `vram_headroom_pct < min_vram_headroom_pct` (`vram_low`)
- `vram_fit` is 0, i.e. the model doesn't fit (`vram_full`), or `thermal_headroom` is 0, i.e. at or over the throttle threshold (`thermal_limit`). The spec defines a factor of 0.0 as infeasible on its own.

When no node is eligible, the agent falls back to round-robin over nodes with fresh health data; if there are none it returns 503 with the per-node exclusion reasons.

## Proxy behavior

- One shared `httpx.AsyncClient` (connection pooling) for all upstream requests.
- Upstream status codes and headers are passed through for streaming and non-streaming responses; hop-by-hop headers are dropped; the body is relayed byte-for-byte (compressed bodies stay compressed).
- Responses carry `x-arp-request-id` and `x-arp-node`. An incoming `x-request-id` is reused and forwarded upstream.
- Upstream connection failures return `502 {"error": "upstream_unreachable"}` and timeouts return `504 {"error": "upstream_timeout"}`; both are logged and count against the node's reliability.

## Decision Log

Every request appends one JSONL entry at routing time, then a second entry with the same `request_id` when it completes:

```json
{"ts":"2026-06-15T09:01:23Z","request_id":"req-abc123","selected_node":"node-1","score":0.847,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.847,"node-2":0.0},"node_exclusions":{"node-2":"throttling"},"primary_constraint":"vram_fit","factors":{"thermal_headroom":0.94,"vram_fit":0.71,"wes_history":0.92,"reliability":0.94},"ttft_ms":null,"total_ms":null,"status_code":null,"error":null}
{"ts":"2026-06-15T09:01:26Z","request_id":"req-abc123","selected_node":"node-1","score":0.847,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.847,"node-2":0.0},"node_exclusions":{"node-2":"throttling"},"primary_constraint":"vram_fit","factors":{"thermal_headroom":0.94,"vram_fit":0.71,"wes_history":0.92,"reliability":0.94},"ttft_ms":312.4,"total_ms":2841.7,"status_code":200,"error":null}
```

`reason` values:
- `highest_score`: normal path, best-scoring node selected
- `fallback_round_robin`: no node was eligible, fell back to round-robin
- `no_eligible_node`: no node had fresh health data (or fallback is disabled)

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a mock Wicklee payload and mock inference servers (`httpx.MockTransport`); no hardware or network is needed.

## File Structure

```
agent.py        — startup: wires poller + proxy + logger, runs uvicorn
scorer.py       — ARP v0.1 Score function and NodeState (staleness, trend, reliability)
poller.py       — async background health polling loop
proxy.py        — OpenAI-compatible FastAPI proxy with score-based routing
logger.py       — thread-safe JSONL decision logger with completion updates
config.json     — example single-node configuration
requirements.txt
requirements-dev.txt
tests/          — pytest suite (scorer, proxy, poller)
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
