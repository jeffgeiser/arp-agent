import gzip
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import proxy as proxy_mod
from helpers import cfg, health
from logger import DecisionLogger
from proxy import build_proxy, filter_headers
from scorer import NodeState


class _Raw(httpx.AsyncByteStream):
    """Unread body, like a real network response (httpx eagerly reads `content=`)."""

    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        if self.data:
            yield self.data


def resp(status, content=b"", json_body=None, headers=None):
    headers = dict(headers or {})
    if json_body is not None:
        content = json.dumps(json_body).encode()
        headers.setdefault("content-type", "application/json")
    return httpx.Response(status, stream=_Raw(content), headers=headers)


def make_nodes(*healths):
    nodes = []
    for i, h in enumerate(healths, start=1):
        n = NodeState(f"node-{i}", f"http://wicklee-{i}", f"http://infer-{i}")
        if h is not None:
            n.update(h, cfg())
        nodes.append(n)
    return nodes


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "decisions.jsonl"


def app_with(handler, nodes, log_path, config=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    logger = DecisionLogger(str(log_path))
    app = build_proxy(nodes, config or cfg(), logger, client=client)
    return app, logger


def test_routes_to_the_highest_scoring_node(log_path):
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return resp(200, json_body={"ok": True})

    nodes = make_nodes(health(junction_temp_c=85.0), health(junction_temp_c=50.0))
    app, _ = app_with(handler, nodes, log_path)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 200
    assert seen == ["infer-2"]
    assert r.headers["x-arp-node"] == "node-2"
    entries = read_log(log_path)
    assert [e["selected_node"] for e in entries] == ["node-2", "node-2"]
    assert entries[0]["reason"] == "highest_score"
    assert entries[0]["factors"]["thermal_headroom"] == 1.0
    assert entries[1]["status_code"] == 200 and entries[1]["total_ms"] is not None


def test_non_stream_preserves_status_headers_and_encoded_body(log_path):
    raw = gzip.compress(b'{"error":"not found"}')

    def handler(req):
        assert req.headers["host"] == "infer-1"  # the client's Host header is not forwarded
        assert req.headers["authorization"] == "Bearer abc"
        assert req.url.params.get_list("a") == ["1", "2"]
        return resp(
            404,
            content=raw,
            headers={"content-type": "application/json", "content-encoding": "gzip", "x-upstream": "yes", "connection": "close, x-hop", "x-hop": "drop-me"},
        )

    app, _ = app_with(handler, make_nodes(health()), log_path)
    with TestClient(app) as c:
        r = c.get("/v1/models?a=1&a=2", headers={"authorization": "Bearer abc"})
    assert r.status_code == 404
    assert r.headers["x-upstream"] == "yes"
    assert "x-hop" not in r.headers
    assert r.headers["content-encoding"] == "gzip"
    assert r.json() == {"error": "not found"}  # body passed through intact, decoded by the client
    assert int(r.headers["content-length"]) == len(raw)


def test_stream_preserves_upstream_status_and_headers(log_path):
    def handler(req):
        return resp(429, content=b"data: slow down\n\n", headers={"content-type": "text/event-stream", "retry-after": "3"})

    app, _ = app_with(handler, make_nodes(health()), log_path)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"stream": True, "messages": []})
    assert r.status_code == 429  # was always 200 before
    assert r.headers["retry-after"] == "3"
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text == "data: slow down\n\n"


def test_stream_relays_chunks_and_logs_ttft(log_path):
    chunks = [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for ch in chunks:
                yield ch

    def handler(req):
        return httpx.Response(200, stream=Body(), headers={"content-type": "text/event-stream"})

    app, logger = app_with(handler, make_nodes(health()), log_path)
    with TestClient(app) as c:
        with c.stream("POST", "/v1/chat/completions", json={"stream": True}) as r:
            body = b"".join(r.iter_bytes())
    assert r.status_code == 200
    assert body == b"".join(chunks)
    final = read_log(log_path)[-1]
    assert final["ttft_ms"] is not None and final["total_ms"] >= final["ttft_ms"]
    assert final["status_code"] == 200 and final["error"] is None
    assert logger.pending_count() == 0


def test_sse_is_streamed_even_without_stream_flag(log_path):
    def handler(req):
        return resp(200, content=b"data: x\n\n", headers={"content-type": "text/event-stream"})

    app, _ = app_with(handler, make_nodes(health()), log_path)
    with TestClient(app) as c:
        r = c.get("/events")
    assert r.status_code == 200 and r.text == "data: x\n\n"


@pytest.mark.parametrize(
    "exc,status,kind",
    [(httpx.ConnectError("refused"), 502, "upstream_unreachable"), (httpx.ReadTimeout("slow"), 504, "upstream_timeout")],
)
def test_upstream_errors_return_json_and_are_logged(log_path, exc, status, kind):
    def handler(req):
        raise exc

    nodes = make_nodes(health())
    app, logger = app_with(handler, nodes, log_path)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={})
    assert r.status_code == status
    assert r.json()["error"] == kind
    final = read_log(log_path)[-1]
    assert final["status_code"] == status and final["error"].startswith(kind)
    assert logger.pending_count() == 0  # no leaked pending entries
    assert list(nodes[0]._outcomes) == [False]


def test_5xx_counts_against_reliability(log_path):
    def handler(req):
        return resp(500, json_body={})

    nodes = make_nodes(health())
    app, _ = app_with(handler, nodes, log_path)
    with TestClient(app) as c:
        for _ in range(3):
            assert c.get("/x").status_code == 500
    assert list(nodes[0]._outcomes) == [False, False, False]


def test_stale_or_missing_health_is_excluded(log_path):
    def handler(req):
        return resp(200, json_body={})

    stale, fresh = make_nodes(health(), health(junction_temp_c=85.0))
    stale.updated_at -= 10.0  # older than stale_after_ms
    never = NodeState("node-3", "http://w3", "http://infer-3")
    app, _ = app_with(handler, [stale, fresh, never], log_path)
    with TestClient(app) as c:
        r = c.get("/x")
    assert r.headers["x-arp-node"] == "node-2"
    entry = read_log(log_path)[0]
    assert entry["node_exclusions"] == {"node-1": "stale_health", "node-3": "no_health_data"}


def test_no_fresh_node_returns_503(log_path):
    def handler(req):
        raise AssertionError("must not be called")

    (node,) = make_nodes(health())
    node.updated_at -= 10.0
    app, _ = app_with(handler, [node], log_path)
    with TestClient(app) as c:
        r = c.get("/x")
    assert r.status_code == 503
    assert r.json()["node_exclusions"] == {"node-1": "stale_health"}


def test_round_robin_fallback_skips_stale_nodes(log_path):
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return resp(200)

    a, b = make_nodes(health(status="degraded"), health(status="degraded"))
    b.updated_at -= 10.0
    app, _ = app_with(handler, [a, b], log_path)
    with TestClient(app) as c:
        for _ in range(3):
            c.get("/x")
    assert seen == ["infer-1"] * 3
    assert read_log(log_path)[0]["reason"] == "fallback_round_robin"


def test_per_model_vram_requirement_is_used(log_path):
    def handler(req):
        return resp(200)

    small, big = make_nodes(health(vram_total_gb=16.0, vram_used_gb=4.0), health(vram_total_gb=80.0, vram_used_gb=4.0))
    config = cfg()
    config["models"] = {"llama-70b": 40.0}
    app, _ = app_with(handler, [small, big], log_path, config)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"model": "llama-70b"})
    assert r.headers["x-arp-node"] == "node-2"
    assert read_log(log_path)[0]["node_exclusions"] == {"node-1": "vram_full"}


def test_one_shared_client_is_reused(monkeypatch, log_path):
    created = []
    real = httpx.AsyncClient

    class Counting(real):
        def __init__(self, *a, **kw):
            created.append(1)
            kw["transport"] = httpx.MockTransport(lambda req: resp(200, json_body={}))
            super().__init__(*a, **kw)

    monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", Counting)
    app = build_proxy(make_nodes(health()), cfg(), DecisionLogger(str(log_path)))
    with TestClient(app) as c:
        for _ in range(5):
            assert c.get("/x").status_code == 200
    assert len(created) == 1


def test_filter_headers_drops_hop_by_hop():
    h = httpx.Headers({"Connection": "keep-alive, X-Foo", "X-Foo": "1", "Keep-Alive": "5", "Transfer-Encoding": "chunked", "Content-Length": "3", "X-Keep": "y"})
    assert dict(filter_headers(h)) == {"x-keep": "y"}
