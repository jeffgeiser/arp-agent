# Two-Node Setup Example

This example shows how to run `arp-agent` against two local inference nodes,
each monitored by a Wicklee health daemon.

## Prerequisites

- Two vLLM instances running (e.g. on ports 8000 and 8001)
- Two Wicklee daemons running against each GPU (e.g. on ports 8080 and 8081)
- `arp-agent` installed: `pip install -r requirements.txt`

## Config

Create `config_two_nodes.json`:

```json
{
  "listen_port": 8090,
  "nodes": [
    {
      "id": "node-1",
      "wicklee_url": "http://localhost:8080",
      "inference_url": "http://localhost:8000"
    },
    {
      "id": "node-2",
      "wicklee_url": "http://localhost:8081",
      "inference_url": "http://localhost:8001"
    }
  ],
  "scoring": {
    "throttle_threshold_pct": -15,
    "poll_interval_ms": 100,
    "stale_after_ms": 2000
  },
  "fallback": "round_robin",
  "log_file": "decisions.jsonl"
}
```

## Start the Agent

```bash
python agent.py config_two_nodes.json
```

You should see:

```
2026-06-15 09:00:00 INFO     arp-agent  ARP agent starting — port=8090 nodes=2 log=decisions.jsonl
2026-06-15 09:00:00 INFO     arp-agent    node=node-1  wicklee=http://localhost:8080  inference=http://localhost:8000
2026-06-15 09:00:00 INFO     arp-agent    node=node-2  wicklee=http://localhost:8081  inference=http://localhost:8001
```

## Send a Test Request

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-q4_k_m",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 50
  }'
```

The agent routes to the higher-scoring node. Check `decisions.jsonl`:

```
{"ts":"2026-06-15T09:01:23Z","request_id":"req-abc123","selected_node":"node-2","score":0.812,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.7234,"node-2":0.812},"node_exclusions":{},"ttft_ms":null,"total_ms":null}
{"ts":"2026-06-15T09:01:26Z","request_id":"req-abc123","selected_node":"node-2","score":0.812,"reason":"highest_score","fallback_used":false,"node_scores":{"node-1":0.7234,"node-2":0.812},"node_exclusions":{},"ttft_ms":287.4,"total_ms":2914.2}
```

## Mock Wicklee Endpoint (for testing without hardware)

If you don't have real Wicklee daemons, mock them with this small FastAPI server:

```python
# mock_wicklee.py
import sys
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health(node_id: str = "node-1", thermal: float = 18.4, vram: float = 63.75, wes_delta: float = -12.4):
    return {
        "node_id": node_id,
        "ts": "2026-06-15T09:00:00Z",
        "model_loaded": "qwen2.5-7b-q4_k_m",
        "tok_s": 24.3,
        "power_w": 38.2,
        "tok_w": 0.636,
        "wes_score": 8.4,
        "wes_delta_pct": wes_delta,
        "cold_baseline_tok_w": 0.721,
        "junction_temp_c": 71.2,
        "thermal_headroom_pct": thermal,
        "vram_used_gb": 5.8,
        "vram_total_gb": 16.0,
        "vram_headroom_pct": vram,
        "status": "healthy"
    }

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run(app, port=port)
```

Run two mock nodes with different scores:

```bash
# node-1: 71.2C junction (18.8C below a 90C throttle), 5.8 GB of 16 GB VRAM
python mock_wicklee.py 8080 &

# node-2: run a copy with a cooler junction_temp_c (e.g. 55.0) so it scores higher.
# Tweak the /health defaults in mock_wicklee.py or add a second app variant.
python mock_wicklee.py 8081 &

python agent.py config_two_nodes.json
```

Requests will consistently route to node-2 while it holds a higher score.
Simulate throttling by setting `wes_delta_pct` below `-15` on node-2 and
watch the agent fall back to node-1.
