from collections import deque

import pytest

from helpers import cfg, health
from scorer import (
    SPEC_WEIGHTS,
    NodeState,
    score_health,
    scoring_config,
    vram_fit_from_fill,
    wes_history_from_ratio,
)


def test_default_weights_are_the_spec_weights():
    assert scoring_config(cfg())["weights"] == {"thermal_headroom": 0.30, "vram_fit": 0.30, "wes_history": 0.25, "reliability": 0.15}
    assert SPEC_WEIGHTS == scoring_config({})["weights"]


@pytest.mark.parametrize("fill,expected", [(0.5, 1.0), (0.60, 1.0), (0.75, 0.625), (0.90, 0.25), (0.95, 0.125), (1.0, 0.0), (1.2, 0.0)])
def test_vram_fit_curve_matches_spec(fill, expected):
    assert vram_fit_from_fill(fill) == pytest.approx(expected)


def test_wes_history_curve_is_continuous_and_matches_spec():
    eps = 1e-9
    for boundary, value in [(0.50, 0.0), (0.80, 0.75), (1.05, 1.0)]:
        assert wes_history_from_ratio(boundary - eps) == pytest.approx(value, abs=1e-6)
        assert wes_history_from_ratio(boundary) == pytest.approx(value, abs=1e-6)
    assert wes_history_from_ratio(0.95) == pytest.approx(0.90)
    assert wes_history_from_ratio(0.30) == 0.0
    assert wes_history_from_ratio(1.50) == 1.0


def test_score_is_the_weighted_sum_of_spec_factors():
    h = health(junction_temp_c=75.0, vram_used_gb=12.0, vram_total_gb=16.0, wes_delta_pct=-5.0)
    outcomes = deque([True] * 9 + [False])
    r = score_health(h, cfg(), outcomes=outcomes)
    thermal = (90 - 75) / 30  # 0.5
    vram = vram_fit_from_fill(12.0 / 15.0)  # 1 GB baseline
    wes = 0.95 - 0.05
    rel = 0.9
    expected = 0.30 * thermal + 0.30 * vram + 0.25 * wes + 0.15 * rel
    assert r.factors == pytest.approx({"thermal_headroom": thermal, "vram_fit": round(vram, 4), "wes_history": wes, "reliability": rel}, abs=1e-4)
    assert r.score == pytest.approx(expected, abs=1e-4)
    assert r.excluded is None
    assert r.primary_constraint == "thermal_headroom"
    assert r.missing == []


def test_warming_trend_penalizes_thermal():
    h = health(junction_temp_c=60.0)
    flat = score_health(h, cfg(), trend_c_per_min=0.0).factors["thermal_headroom"]
    warming = score_health(h, cfg(), trend_c_per_min=2.5).factors["thermal_headroom"]
    assert flat == pytest.approx(1.0)
    assert warming == pytest.approx(0.5)


def test_missing_fields_never_raise_and_are_reported():
    r = score_health({"status": "healthy"}, cfg())
    assert set(r.missing) == {"thermal_headroom", "vram_fit", "wes_history"}
    assert r.factors["wes_history"] == 0.7  # spec cold-start default
    assert 0.0 < r.score <= 1.0
    assert r.excluded is None


def test_non_numeric_fields_are_treated_as_missing():
    h = health(junction_temp_c="hot", thermal_headroom_pct=None, vram_total_gb=True, vram_used_gb=..., vram_headroom_pct="x", wes_delta_pct=float("nan"))
    r = score_health(h, cfg())
    assert set(r.missing) == {"thermal_headroom", "vram_fit", "wes_history"}


def test_missing_status_excludes_the_node():
    h = health(status=...)
    assert score_health(h, cfg()).excluded == "missing_status"
    assert score_health(h, cfg()).score == 0.0


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"status": "degraded"}, "unhealthy"),
        ({"wes_delta_pct": -20.0}, "throttling"),
        ({"vram_headroom_pct": 5.0}, "vram_low"),
        ({"vram_used_gb": 15.5, "vram_headroom_pct": 50.0}, "vram_full"),
        ({"junction_temp_c": 95.0}, "thermal_limit"),
    ],
)
def test_hard_exclusions(overrides, reason):
    r = score_health(health(**overrides), cfg())
    assert r.excluded == reason
    assert r.score == 0.0


def test_reliability_cold_start_and_window():
    node = NodeState("n", "http://w", "http://i")
    node.configure(cfg(reliability_window=10))
    node.update(health(), cfg(), now=0.0)
    assert node.score(cfg(), now=0.5).factors["reliability"] == 0.5  # < 5 outcomes
    for ok in [True] * 8 + [False] * 4:  # window keeps the last 10: 6 ok, 4 failed
        node.record_outcome(ok)
    assert node.score(cfg(), now=0.5).factors["reliability"] == pytest.approx(0.6)


def test_stale_health_is_ignored():
    node = NodeState("n", "http://w", "http://i")
    c = cfg(stale_after_ms=2000)
    assert node.health_status(c, now=0.0) == "no_health_data"
    node.update(health(), c, now=100.0)
    assert node.health_status(c, now=101.0) == "ok"
    assert node.score(c, now=101.0) is not None
    assert node.health_status(c, now=102.5) == "stale_health"
    assert node.fresh_health(c, now=102.5) is None
    assert node.score(c, now=102.5) is None


def test_poll_errors_do_not_refresh_health():
    node = NodeState("n", "http://w", "http://i")
    c = cfg(stale_after_ms=1000)
    node.update(health(), c, now=0.0)
    node.set_error("ConnectError")
    assert node.last_error == "ConnectError"
    assert node.health_status(c, now=2.0) == "stale_health"


def test_temperature_trend_is_measured_from_samples():
    node = NodeState("n", "http://w", "http://i")
    c = cfg()
    node.update(health(junction_temp_c=60.0), c, now=0.0)
    node.update(health(junction_temp_c=62.0), c, now=30.0)
    assert node.trend_c_per_min() == pytest.approx(4.0)


def test_non_object_health_is_rejected():
    node = NodeState("n", "http://w", "http://i")
    with pytest.raises(ValueError):
        node.update(["not", "a", "dict"], cfg())


def test_legacy_weight_keys_are_flagged():
    sc = scoring_config({"scoring": {"thermal_headroom_weight": 0.4, "tok_w_delta_weight": 0.3}})
    assert sc["legacy_weight_keys"] == ["thermal_headroom_weight", "tok_w_delta_weight"]
    assert sc["weights"] == SPEC_WEIGHTS
