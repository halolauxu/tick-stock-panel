from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_low_attention_forecast_drift_discovery.py"
    )
    spec = importlib.util.spec_from_file_location("p0_low_attention", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_attention_uses_only_turnover_strictly_before_each_date() -> None:
    start = date(2020, 1, 1)
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * 22,
            "date": [start + timedelta(days=index) for index in range(22)],
            "turnover_rate": [1.0] * 21 + [100.0],
        }
    )

    result = study.attach_prior_attention(panel).sort("date")

    assert result.get_column("prior_turnover_20d")[-1] == 1.0


def test_build_events_separates_low_high_and_negative_controls(monkeypatch) -> None:
    raw = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH", "000001.SZ"],
            "ann_date": [date(2020, 4, 1)] * 3,
        }
    )
    categorized = raw.with_columns(
        pl.Series("category", ["growth_0_50", "growth_50_100", "negative_control"])
    )
    monkeypatch.setattr(study.forecast, "categorize_events", lambda _: categorized)
    attention = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH", "000001.SZ"],
            "date": [date(2020, 4, 1)] * 3,
            "prior_turnover_20d": [1.0, 9.0, 1.5],
            "attention_percentile": [0.2, 0.8, 0.3],
        }
    )

    events = study.build_events(raw, attention)
    got = dict(events.select("symbol", "category").iter_rows())

    assert got == {
        "600001.SH": study.CANDIDATE,
        "600002.SH": study.HIGH_CONTROL,
        "000001.SZ": study.NEGATIVE_CONTROL,
    }


def test_gate_requires_low_attention_increment() -> None:
    def summary(excess: float) -> dict:
        return {
            "tradable_events": 800,
            "announcement_days": 300,
            "tradable_rate": 0.98,
            "unresolved_exits": 0,
            "mean_net_return": 0.012,
            "mean_excess_return": excess,
            "excess_daily_cluster_t": 3.2,
            "positive_excess_years": 6,
        }

    passing = {
        study.CANDIDATE: summary(0.008),
        study.HIGH_CONTROL: summary(0.004),
        study.NEGATIVE_CONTROL: summary(-0.002),
    }
    failed = {**passing, study.HIGH_CONTROL: summary(0.007)}

    assert study.evaluate(passing)["passed"] is True
    assert study.evaluate(failed)["passed"] is False
    assert study.evaluate(failed)["checks"]["beats_high_attention_by_25bp"] is False

