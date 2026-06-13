from __future__ import annotations

import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from logger import DecisionLogger
from scorer import NodeState, exclusion_reason


def _select_node(
    nodes: List[NodeState],
    cfg: dict,
    rr_state: dict,
) -> Tuple[Optional[NodeState], Dict[str, float], Dict[str, str], str, bool]:
    scores: Dict[str, float] = {}
    exclusions: Dict[str, str] = {}

    for node in nodes:
        health = node.get_health()
        if health is None:
            exclusions[node.node_id] = "no_health_data"
            scores[node.node_id] = 0.0
            continue
        s = node.get_score()
        scores[node.node_id] = s
        if s == 0.0:
            exclusions[node.node_id] = exclusion_reason(health, cfg) or "score_zero"

    eligible = [(n, scores[n.node_id]) for n in nodes if scores.get(n.node_id, 0.0) > 0.0]

    if eligible:
        best = max(eligible, key=lambda x: x[1])
        return best[0], scores, exclusions, "highest_score", False

    if cfg.get("fallback") == "round_robin":
        candidates = [n for n in nodes if n.get_health() is not None]
        if candidates:
            idx = rr_state["counter"] % len(candidates)
            rr_state["counter"] += 1
            return candidates[idx], scores, exclusions, "fallback_round_robin", True

    return None, scores, exclusions, "no_eligible_node", False


def build_proxy(nodes: List[NodeState], cfg: dict, decision_logger: DecisionLogger) -> FastAPI:
    app = FastAPI(title="ARP Agent")
    rr_state = {"counter": 0}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def forward(request: Request, path: str) -> Response:
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        t_start = time.monotonic()

        node, node_scores, node_exclusions, reason, fallback_used = _select_node(
            nodes, cfg, rr_state
        )

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
            return Response(
                content=json.dumps({"error": "no eligible node available"}),
                status_code=503,
                media_type="application/json",
            )

        score = node_scores.get(node.node_id, 0.0)
        decision_logger.record(
            request_id=request_id,
            selected_node=node.node_id,
            score=score,
            reason=reason,
            fallback_used=fallback_used,
            node_scores=node_scores,
            node_exclusions=node_exclusions,
        )

        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        target_url = f"{node.inference_url}/{path}"
        params = dict(request.query_params)

        is_stream = False
        try:
            if body:
                is_stream = bool(json.loads(body).get("stream", False))
        except (json.JSONDecodeError, TypeError):
            pass

        if is_stream:
            timing: Dict[str, Optional[float]] = {"ttft_ms": None}

            async def _stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        request.method,
                        target_url,
                        content=body,
                        headers=headers,
                        params=params,
                        timeout=120.0,
                    ) as upstream:
                        async for chunk in upstream.aiter_bytes():
                            if timing["ttft_ms"] is None and chunk:
                                timing["ttft_ms"] = (time.monotonic() - t_start) * 1000
                            yield chunk
                decision_logger.update_timing(
                    request_id,
                    timing["ttft_ms"],
                    (time.monotonic() - t_start) * 1000,
                )

            return StreamingResponse(_stream(), media_type="text/event-stream")

        async with httpx.AsyncClient() as client:
            upstream = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=params,
                timeout=120.0,
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000
        decision_logger.update_timing(request_id, elapsed_ms, elapsed_ms)

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    return app
