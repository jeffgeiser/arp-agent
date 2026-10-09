from __future__ import annotations

import asyncio
import json
import logging
import sys

import uvicorn

from logger import DecisionLogger
from poller import polling_loop
from proxy import build_proxy
from scorer import NodeState, scoring_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger("arp-agent")
# httpx logs every request at INFO; with 100 ms health polling that floods the log.
logging.getLogger("httpx").setLevel(logging.WARNING)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


async def _run(cfg_path: str) -> None:
    cfg = _load_cfg(cfg_path)

    nodes = [
        NodeState(
            node_id=n["id"],
            wicklee_url=n["wicklee_url"],
            inference_url=n["inference_url"],
        )
        for n in cfg["nodes"]
    ]
    for n in nodes:
        n.configure(cfg)

    sc = scoring_config(cfg)
    if sc["legacy_weight_keys"]:
        log.warning(
            "Ignoring legacy scoring keys %s: scoring now follows the ARP v0.1 spec "
            "(scoring.weights: thermal_headroom/vram_fit/wes_history/reliability)",
            ", ".join(sc["legacy_weight_keys"]),
        )
    log.info("Score weights: %s", sc["weights"])

    decision_logger = DecisionLogger(cfg.get("log_file", "decisions.jsonl"))
    app = build_proxy(nodes, cfg, decision_logger)

    host = cfg.get("listen_host", "127.0.0.1")
    log.info(
        "ARP agent starting — host=%s port=%d nodes=%d log=%s",
        host,
        cfg["listen_port"],
        len(nodes),
        cfg.get("log_file", "decisions.jsonl"),
    )
    for n in nodes:
        log.info(
            "  node=%s  wicklee=%s  inference=%s",
            n.node_id,
            n.wicklee_url,
            n.inference_url,
        )

    poll_task = asyncio.create_task(polling_loop(nodes, cfg))

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=cfg["listen_port"], log_level="warning")
    )
    try:
        await server.serve()
    finally:
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    asyncio.run(_run(cfg_path))
