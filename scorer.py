"""ARP v0.1 Score function (spec/02-score.md in jeffgeiser/arp-spec) and node state.

score = 0.30 * thermal_headroom + 0.30 * vram_fit + 0.25 * wes_history + 0.15 * reliability

Each factor is in [0, 1]. Inputs come from the node's Wicklee /health payload;
where the agent can't observe what the spec asks for, it uses the closest
signal it has (documented per factor below and in the README).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

SPEC_WEIGHTS: Dict[str, float] = {
    "thermal_headroom": 0.30,
    "vram_fit": 0.30,
    "wes_history": 0.25,
    "reliability": 0.15,
}

DEFAULT_SCORING: Dict[str, Any] = {
    "weights": dict(SPEC_WEIGHTS),
    "throttle_c": 90.0,  # platform throttle threshold
    "target_margin_c": 30.0,  # margin the spec treats as "fully safe"
    "trend_window_s": 60.0,  # window for the warming-trend penalty
    "vram_baseline_gb": 1.0,  # OS/driver overhead (spec default for discrete GPUs)
    "throttle_threshold_pct": -15.0,  # agent policy: exclude below this wes_delta_pct
    "min_vram_headroom_pct": 10.0,  # agent policy: exclude below this headroom
    "poll_interval_ms": 100,
    "stale_after_ms": 2000,  # health older than this is ignored
    "reliability_window": 50,  # recent proxied requests counted for reliability
}

LEGACY_WEIGHT_KEYS = ("thermal_headroom_weight", "vram_headroom_weight", "tok_w_delta_weight")


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def num(health: Dict[str, Any], key: str) -> Optional[float]:
    """A finite number from the health payload, or None if missing/invalid."""
    v = health.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def scoring_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the user's "scoring" block over the spec defaults."""
    user = dict(cfg.get("scoring") or {})
    out = {**DEFAULT_SCORING, **{k: v for k, v in user.items() if k != "weights"}}
    weights = dict(SPEC_WEIGHTS)
    weights.update(user.get("weights") or {})
    out["weights"] = weights
    out["legacy_weight_keys"] = [k for k in LEGACY_WEIGHT_KEYS if k in user]
    return out


# --- Per-factor computation (spec/02-score.md "Per-Factor Computation") -------

def vram_fit_from_fill(fill_ratio: float) -> float:
    if fill_ratio <= 0.60:
        return 1.0
    if fill_ratio <= 0.90:
        return 1.0 - 0.75 * (fill_ratio - 0.60) / 0.30
    if fill_ratio <= 1.00:
        return 0.25 - 0.25 * (fill_ratio - 0.90) / 0.10
    return 0.0


def wes_history_from_ratio(ratio: float) -> float:
    """Continuous v0.1 curve: 0 at 0.50, 0.75 at 0.80, 1.0 at 1.05."""
    if ratio >= 1.05:
        return 1.0
    if ratio >= 0.80:
        return ratio - 0.05
    if ratio >= 0.50:
        return 2.5 * (ratio - 0.50)
    return 0.0


def thermal_factor(health: Dict[str, Any], sc: Dict[str, Any], trend_c_per_min: Optional[float]) -> Tuple[Optional[float], str]:
    current = num(health, "junction_temp_c")
    if current is not None:
        throttle = float(health.get("throttle_c") or sc["throttle_c"])
        headroom = clamp01((throttle - current) / float(sc["target_margin_c"]))
        pen = 1.0
        if trend_c_per_min is not None:
            pen = clamp01(1.0 - max(0.0, trend_c_per_min) / 5.0)
        detail = f"{current:.0f}C / {throttle:.0f}C throttle"
        if trend_c_per_min is not None and trend_c_per_min > 0:
            detail += f", warming {trend_c_per_min:.1f}C/min"
        return headroom * pen, detail
    pct = num(health, "thermal_headroom_pct")
    if pct is not None:  # fallback when the temperature isn't reported
        return clamp01(pct / 100.0), f"thermal headroom {pct:.0f}% (no temperature reported)"
    return None, "no thermal data"


def vram_factor(health: Dict[str, Any], sc: Dict[str, Any], required_gb: Optional[float]) -> Tuple[Optional[float], str]:
    total = num(health, "vram_total_gb")
    used = num(health, "vram_used_gb")
    if total is not None and total > 0:
        available = total - float(sc["vram_baseline_gb"])
        # The served model is already resident, so its footprint is part of
        # `used`; an explicit per-model requirement replaces that estimate.
        need = required_gb if required_gb is not None else used
        if need is not None and available > 0:
            fill = need / available
            return vram_fit_from_fill(fill), f"{need:.1f}GB of {available:.1f}GB available ({fill:.0%} fill)"
    pct = num(health, "vram_headroom_pct")
    if pct is not None:
        fill = 1.0 - pct / 100.0
        return vram_fit_from_fill(fill), f"{fill:.0%} VRAM fill (from headroom)"
    return None, "no VRAM data"


def wes_factor(health: Dict[str, Any]) -> Tuple[Optional[float], str]:
    delta = num(health, "wes_delta_pct")
    if delta is None:
        return None, "no WES baseline (cold start)"
    ratio = 1.0 + delta / 100.0
    return wes_history_from_ratio(ratio), f"WES {ratio:.2f}x baseline"


def reliability_factor(outcomes: Deque[bool]) -> Tuple[float, str]:
    total = len(outcomes)
    if total < 5:
        return 0.5, f"cold start ({total} recent requests)"
    clean = sum(1 for o in outcomes if o)
    return clean / total, f"{clean}/{total} recent requests succeeded"


@dataclass
class ScoreResult:
    score: float
    factors: Dict[str, float]
    details: Dict[str, str]
    excluded: Optional[str] = None
    primary_constraint: Optional[str] = None
    missing: List[str] = field(default_factory=list)


# Neutral value for a factor whose inputs are missing; recorded in `missing`.
MISSING_FACTOR_VALUE = 0.5


def score_health(
    health: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    trend_c_per_min: Optional[float] = None,
    outcomes: Optional[Deque[bool]] = None,
    required_gb: Optional[float] = None,
) -> ScoreResult:
    sc = scoring_config(cfg)
    w = sc["weights"]
    factors: Dict[str, float] = {}
    details: Dict[str, str] = {}
    missing: List[str] = []

    def put(name: str, value: Optional[float], detail: str, cold: Optional[float] = None) -> None:
        if value is None:
            missing.append(name)
            value = MISSING_FACTOR_VALUE if cold is None else cold
        factors[name] = round(clamp01(value), 4)
        details[name] = detail

    t, td = thermal_factor(health, sc, trend_c_per_min)
    put("thermal_headroom", t, td)
    v, vd = vram_factor(health, sc, required_gb)
    put("vram_fit", v, vd)
    h, hd = wes_factor(health)
    put("wes_history", h, hd, cold=0.7)  # spec cold-start default
    r, rd = reliability_factor(outcomes if outcomes is not None else deque())
    put("reliability", r, rd)

    total_w = sum(float(w.get(k, 0.0)) for k in SPEC_WEIGHTS) or 1.0
    score = sum(float(w.get(k, 0.0)) * factors[k] for k in SPEC_WEIGHTS) / total_w
    # Primary constraint: the factor costing the most weighted score.
    primary = max(SPEC_WEIGHTS, key=lambda k: float(w.get(k, 0.0)) * (1.0 - factors[k]))
    result = ScoreResult(round(score, 4), factors, details, None, primary, missing)

    reason = exclusion_reason(health, cfg, result)
    if reason:
        result.excluded = reason
        result.score = 0.0
    return result


def exclusion_reason(health: Dict[str, Any], cfg: Dict[str, Any], result: Optional[ScoreResult] = None) -> Optional[str]:
    """Hard gates. A factor at 0.0 means "infeasible on its own" in the spec."""
    sc = scoring_config(cfg)
    status = health.get("status")
    if status is None:
        return "missing_status"
    if status != "healthy":
        return "unhealthy"
    delta = num(health, "wes_delta_pct")
    if delta is not None and delta < float(sc["throttle_threshold_pct"]):
        return "throttling"
    headroom = num(health, "vram_headroom_pct")
    if headroom is not None and headroom < float(sc["min_vram_headroom_pct"]):
        return "vram_low"
    if result is not None:
        if result.factors.get("vram_fit") == 0.0 and "vram_fit" not in result.missing:
            return "vram_full"
        if result.factors.get("thermal_headroom") == 0.0 and "thermal_headroom" not in result.missing:
            return "thermal_limit"
    return None


@dataclass
class NodeState:
    node_id: str
    wicklee_url: str
    inference_url: str
    health: Optional[dict] = None
    last_error: Optional[str] = None
    updated_at: Optional[float] = None  # monotonic seconds of the last good poll

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._temps: Deque[Tuple[float, float]] = deque()
        self._outcomes: Deque[bool] = deque(maxlen=int(DEFAULT_SCORING["reliability_window"]))

    def configure(self, cfg: Dict[str, Any]) -> None:
        window = int(scoring_config(cfg)["reliability_window"])
        with self._lock:
            self._outcomes = deque(self._outcomes, maxlen=max(1, window))

    def update(self, health: dict, cfg: Dict[str, Any], now: Optional[float] = None) -> None:
        if not isinstance(health, dict):
            raise ValueError("health payload is not a JSON object")
        now = time.monotonic() if now is None else now
        window = float(scoring_config(cfg)["trend_window_s"])
        temp = num(health, "junction_temp_c")
        with self._lock:
            self.health = dict(health)
            self.updated_at = now
            self.last_error = None
            if temp is not None:
                self._temps.append((now, temp))
            while self._temps and now - self._temps[0][0] > window:
                self._temps.popleft()

    def set_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def record_outcome(self, ok: bool) -> None:
        with self._lock:
            self._outcomes.append(ok)

    def trend_c_per_min(self) -> Optional[float]:
        with self._lock:
            if len(self._temps) < 2:
                return None
            (t0, c0), (t1, c1) = self._temps[0], self._temps[-1]
        if t1 - t0 < 1.0:
            return None
        return (c1 - c0) / (t1 - t0) * 60.0

    def fresh_health(self, cfg: Dict[str, Any], now: Optional[float] = None) -> Optional[dict]:
        """The latest health, or None if there is none or it is stale."""
        now = time.monotonic() if now is None else now
        stale = float(scoring_config(cfg)["stale_after_ms"]) / 1000.0
        with self._lock:
            if self.health is None or self.updated_at is None:
                return None
            if now - self.updated_at > stale:
                return None
            return dict(self.health)

    def health_status(self, cfg: Dict[str, Any], now: Optional[float] = None) -> str:
        """'ok', 'no_health_data' or 'stale_health'."""
        with self._lock:
            has = self.health is not None
        if not has:
            return "no_health_data"
        return "ok" if self.fresh_health(cfg, now) is not None else "stale_health"

    def score(self, cfg: Dict[str, Any], required_gb: Optional[float] = None, now: Optional[float] = None) -> Optional[ScoreResult]:
        health = self.fresh_health(cfg, now)
        if health is None:
            return None
        with self._lock:
            outcomes = deque(self._outcomes)
        return score_health(health, cfg, trend_c_per_min=self.trend_c_per_min(), outcomes=outcomes, required_gb=required_gb)
