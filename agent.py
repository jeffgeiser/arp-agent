from __future__ import annotations

import asyncio
import json
import logging
import sys

import uvicorn

from logger import DecisionLogger
from poller import polling_loop
from proxy import build_proxy
from scorer import NodeState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger("arp-agent")


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

    decision_logger = DecisionLogger(cfg.get("log_file", "decisions.jsonl"))
    app = build_proxy(nodes, cfg, decision_logger)

    log.info(
        "ARP agent starting — port=%d nodes=%d log=%s",
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
        uvicorn.Config(app, host="0.0.0.0", port=cfg["listen_port"], log_level="warning")
    )
    try:
        await server.serve()
    finally:
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    asyncio.run(_run(cfg_path))
