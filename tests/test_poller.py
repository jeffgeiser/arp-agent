import asyncio

import httpx
import pytest

from helpers import cfg, health
from poller import poll_node_once, polling_loop
from scorer import NodeState


def node():
    return NodeState("n", "http://wicklee", "http://infer")


@pytest.mark.asyncio
async def test_successful_poll_updates_health():
    n = node()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=health()))) as c:
        assert await poll_node_once(n, cfg(), c) is True
    assert n.health_status(cfg()) == "ok"
    assert n.last_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, json={}), httpx.Response(200, content=b"not json"), httpx.Response(200, json=[1, 2])],
)
async def test_bad_polls_record_an_error_and_keep_old_health(response):
    n = node()
    n.update(health(junction_temp_c=55.0), cfg())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response)) as c:
        assert await poll_node_once(n, cfg(), c) is False
    assert n.last_error
    assert n.health["junction_temp_c"] == 55.0


@pytest.mark.asyncio
async def test_slow_node_does_not_delay_the_others():
    fast, slow = NodeState("fast", "http://fast", "http://i"), NodeState("slow", "http://slow", "http://i")
    calls = {"fast": 0, "slow": 0}

    async def handler(req):
        calls[req.url.host] += 1
        if req.url.host == "slow":
            await asyncio.sleep(10)
        return httpx.Response(200, json=health())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    task = asyncio.create_task(polling_loop([fast, slow], cfg(poll_interval_ms=10), client))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    assert calls["slow"] == 1
    assert calls["fast"] >= 5
