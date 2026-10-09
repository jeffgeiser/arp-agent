from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import httpx

from scorer import NodeState, scoring_config

log = logging.getLogger(__name__)


async def poll_node_once(node: NodeState, cfg: dict, client: httpx.AsyncClient) -> bool:
    """Fetch one /health sample. Returns True on success."""
    url = f"{node.wicklee_url.rstrip('/')}/health"
    try:
        resp = await client.get(url, timeout=2.0)
        resp.raise_for_status()
        health = resp.json()
        node.update(health, cfg)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # network error, bad status, bad JSON, non-object
        node.set_error(f"{type(exc).__name__}: {exc}")
        log.warning("poll failed node=%s error=%s", node.node_id, exc)
        return False


async def _node_loop(node: NodeState, cfg: dict, client: httpx.AsyncClient, interval: float) -> None:
    # One loop per node, so a slow or hung node never delays the others.
    while True:
        await poll_node_once(node, cfg, client)
        await asyncio.sleep(interval)


async def polling_loop(nodes: List[NodeState], cfg: dict, client: Optional[httpx.AsyncClient] = None) -> None:
    interval = float(scoring_config(cfg)["poll_interval_ms"]) / 1000.0
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        await asyncio.gather(*[_node_loop(n, cfg, client, interval) for n in nodes])
    finally:
        if own:
            await client.aclose()
