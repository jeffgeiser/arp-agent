from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from logger import DecisionLogger
from scorer import NodeState, ScoreResult

log = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 9110 §7.6.1) plus ones the proxy recomputes.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}



def filter_headers(headers: Any) -> List[Tuple[str, str]]:
    """Drop hop-by-hop headers (and any named in Connection)."""
    items = list(headers.items())
    extra = set()
    for k, v in items:
        if k.lower() == "connection":
            extra.update(t.strip().lower() for t in v.split(","))
    return [(k, v) for k, v in items if k.lower() not in HOP_BY_HOP and k.lower() not in extra]


def select_node(
    nodes: List[NodeState],
    cfg: dict,
    rr_state: dict,
    required_gb: Optional[float] = None,
) -> Tuple[Optional[NodeState], Dict[str, float], Dict[str, str], str, bool, Optional[ScoreResult]]:
    scores: Dict[str, float] = {}
    exclusions: Dict[str, str] = {}
    results: Dict[str, ScoreResult] = {}

    for node in nodes:
        status = node.health_status(cfg)
        if status != "ok":
            exclusions[node.node_id] = status
            scores[node.node_id] = 0.0
            continue
        res = node.score(cfg, required_gb=required_gb)
        if res is None:  # went stale between the two calls
            exclusions[node.node_id] = "stale_health"
            scores[node.node_id] = 0.0
            continue
        results[node.node_id] = res
        scores[node.node_id] = res.score
        if res.excluded or res.score <= 0.0:
            exclusions[node.node_id] = res.excluded or "score_zero"

    eligible = [n for n in nodes if n.node_id in results and n.node_id not in exclusions]
    if eligible:
        best = max(eligible, key=lambda n: scores[n.node_id])
        return best, scores, exclusions, "highest_score", False, results[best.node_id]

    if cfg.get("fallback") == "round_robin":
        # Only nodes with fresh health; stale or never-polled nodes are skipped.
        candidates = [n for n in nodes if n.node_id in results]
        if candidates:
            idx = rr_state["counter"] % len(candidates)
            rr_state["counter"] += 1
            chosen = candidates[idx]
            return chosen, scores, exclusions, "fallback_round_robin", True, results[chosen.node_id]

    return None, scores, exclusions, "no_eligible_node", False, None


def _parse_json(body: bytes) -> Optional[dict]:
    if not body or len(body) > 10_000_000:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def build_proxy(
    nodes: List[NodeState],
    cfg: dict,
    decision_logger: DecisionLogger,
    client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    """Build the routing proxy. Pass `client` to share one (e.g. in tests)."""
    own_client = client is None
    timeout = httpx.Timeout(float(cfg.get("upstream_timeout_s", 120.0)), connect=float(cfg.get("connect_timeout_s", 5.0)))
    model_vram: Dict[str, float] = {k: float(v) for k, v in (cfg.get("models") or {}).items()}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if own_client:
            app.state.client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            )
        try:
            yield
        finally:
            if own_client:
                await app.state.client.aclose()

    app = FastAPI(title="ARP Agent", lifespan=lifespan)
    if not own_client:
        app.state.client = client
    rr_state = {"counter": 0}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def forward(request: Request, path: str) -> Response:
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        t_start = time.monotonic()
        body = await request.body()
        payload = _parse_json(body)
        model = payload.get("model") if payload else None
        required_gb = model_vram.get(model) if isinstance(model, str) else None

        node, node_scores, node_exclusions, reason, fallback_used, result = select_node(nodes, cfg, rr_state, required_gb)

        if node is None:
            decision_logger.record(
                request_id=request_id,
                selected_node=None,
                score=0.0,
                reason=reason,
                fallback_used=fallback_used,
                node_scores=node_scores,
                node_exclusions=node_exclusions,
            )
            return JSONResponse(
                {"error": "no eligible node available", "request_id": request_id, "node_exclusions": node_exclusions},
                status_code=503,
                headers={"x-arp-request-id": request_id},
            )

        decision_logger.record(
            request_id=request_id,
            selected_node=node.node_id,
            score=node_scores.get(node.node_id, 0.0),
            reason=reason,
            fallback_used=fallback_used,
            node_scores=node_scores,
            node_exclusions=node_exclusions,
            primary_constraint=result.primary_constraint if result else None,
            factors=result.factors if result else None,
        )

        headers = filter_headers(request.headers)
        headers.append(("x-request-id", request_id))
        target_url = f"{node.inference_url.rstrip('/')}/{path}"
        http: httpx.AsyncClient = app.state.client
        upstream_req = http.build_request(
            request.method,
            target_url,
            content=body,
            headers=headers,
            params=list(request.query_params.multi_items()),
            timeout=timeout,
        )

        try:
            upstream = await http.send(upstream_req, stream=True)
        except httpx.TimeoutException as exc:
            return _upstream_error(decision_logger, node, request_id, t_start, 504, "upstream_timeout", exc)
        except httpx.HTTPError as exc:
            return _upstream_error(decision_logger, node, request_id, t_start, 502, "upstream_unreachable", exc)

        resp_headers = filter_headers(upstream.headers)
        resp_headers += [("x-arp-request-id", request_id), ("x-arp-node", node.node_id)]
        node.record_outcome(upstream.status_code < 500)

        wants_stream = bool(payload and payload.get("stream"))
        is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")

        if wants_stream or is_sse:
            async def relay() -> AsyncIterator[bytes]:
                ttft: Optional[float] = None
                error: Optional[str] = None
                try:
                    async for chunk in upstream.aiter_raw():
                        if ttft is None and chunk:
                            ttft = (time.monotonic() - t_start) * 1000
                        yield chunk
                except httpx.HTTPError as exc:
                    error = f"stream_interrupted: {type(exc).__name__}"
                    node.record_outcome(False)
                    log.warning("stream from node=%s interrupted: %s", node.node_id, exc)
                finally:
                    await upstream.aclose()
                    decision_logger.complete(
                        request_id,
                        ttft_ms=ttft,
                        total_ms=(time.monotonic() - t_start) * 1000,
                        status_code=upstream.status_code,
                        error=error,
                    )

            response = StreamingResponse(relay(), status_code=upstream.status_code)
            # Pass upstream headers through as-is (content-type, encoding, ...).
            response.raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in resp_headers]
            return response

        try:
            content = b"".join([chunk async for chunk in upstream.aiter_raw()])
        except httpx.HTTPError as exc:
            await upstream.aclose()
            node.record_outcome(False)
            return _upstream_error(decision_logger, node, request_id, t_start, 502, "upstream_read_failed", exc, record=False)
        await upstream.aclose()
        elapsed = (time.monotonic() - t_start) * 1000
        decision_logger.complete(request_id, ttft_ms=elapsed, total_ms=elapsed, status_code=upstream.status_code)
        response = Response(content=content, status_code=upstream.status_code)
        response.raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in resp_headers] + [
            (b"content-length", str(len(content)).encode())
        ]
        return response

    return app


def _upstream_error(
    decision_logger: DecisionLogger,
    node: NodeState,
    request_id: str,
    t_start: float,
    status: int,
    kind: str,
    exc: Exception,
    record: bool = True,
) -> Response:
    if record:
        node.record_outcome(False)
    log.warning("upstream error node=%s kind=%s error=%s", node.node_id, kind, exc)
    decision_logger.complete(
        request_id,
        total_ms=(time.monotonic() - t_start) * 1000,
        status_code=status,
        error=f"{kind}: {type(exc).__name__}",
    )
    return JSONResponse(
        {"error": kind, "node": node.node_id, "request_id": request_id},
        status_code=status,
        headers={"x-arp-request-id": request_id, "x-arp-node": node.node_id},
    )
