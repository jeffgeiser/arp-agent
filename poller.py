from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from scorer import NodeState

log = logging.getLogger(__name__)


async def _poll_node(node: NodeState, cfg: dict, client: httpx.AsyncClient) -> None:
    url = f"{node.wicklee_url}/health"
    try:
        resp = await client.get(url, timeout=2.0)
        resp.raise_for_status()
        health = resp.json()
        node.update(health, cfg)
        log.debug("polled node=%s score=%.4f", node.node_id, node.get_score())
    except Exception as exc:
        node.set_error(str(exc))
        log.warning("poll failed node=%s error=%s", node.node_id, exc)


async def polling_loop(nodes: List[NodeState], cfg: dict) -> None:
    interval = cfg["scoring"]["poll_interval_ms"] / 1000
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.gather(
                *[_poll_node(n, cfg, client) for n in nodes],
                return_exceptions=True,
            )
            await asyncio.sleep(interval)
