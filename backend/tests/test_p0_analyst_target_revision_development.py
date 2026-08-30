from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_analyst_target_revision_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_analyst_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _report(
    report_id: str,
    publish_date: date,
    broker: str,
    target: float,
) -> dict:
    return {
        "report_id": report_id,
        "publish_date": publish_date,
        "symbol": "600001.SH",
        "org_code": broker,
        "target_price_high": target,
        "target_price_low": target,
        "current_rating": "买入",
    }


def test_events_require_two_brokers_with_independent_target_revisions() -> None:
    reports = pl.DataFrame(
        [
            _report("a1", date(2020, 1, 1), "A", 10.0),
            _report("b1", date(2020, 1, 1), "B", 10.0),
            _report("a2", date(2020, 2, 1), "A", 12.0),
            _report("b2", date(2020, 2, 10), "B", 12.0),
        ]
    )

    events = study.build_events(reports)

    assert events.height == 1
    assert events["ann_date"].to_list() == [date(2020, 2, 10)]
    assert events["broker_count"].to_list() == [2]


def test_promotion_gate_requires_high_absolute_and_excess_return() -> None:
    metrics = {
        "tradable_events": 80,
        "announcement_days": 70,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_return": 0.04,
        "mean_excess_return": 0.03,
        "excess_daily_cluster_t": 2.5,
        "positive_excess_years": 3,
        "max_year_positive_excess_share": 0.50,
    }
    assert study.promotion_passed(metrics) is True

    metrics["mean_net_return"] = 0.039
    assert study.promotion_passed(metrics) is False
