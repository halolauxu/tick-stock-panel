from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_cb_call_condition_approach_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_cb_call_approach", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_call_count_signal_enters_at_ten_and_exits_when_count_reaches_fifteen() -> None:
    days = [date(2017, 1, 1) + timedelta(days=index) for index in range(45)]
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH"] * 45,
            "date": days,
            "cb_value": [100.0] * 30 + [130.0] * 15,
        }
    )
    master = pl.DataFrame(
        {"symbol": ["110001.SH"], "stk_code": ["600001.SH"]}
    )

    panel = study.prepare_call_panel(daily, master)
    events = study.build_events(panel)

    assert events.height == 1
    assert events["ann_date"].to_list() == [days[39]]
    assert events["holding_trading_days"].to_list() == [5]


def test_first_observable_window_is_not_misclassified_as_a_crossing() -> None:
    days = [date(2017, 1, 1) + timedelta(days=index) for index in range(35)]
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH"] * 35,
            "date": days,
            "cb_value": [100.0] * 20 + [130.0] * 15,
        }
    )
    master = pl.DataFrame(
        {"symbol": ["110001.SH"], "stk_code": ["600001.SH"]}
    )

    events = study.build_events(study.prepare_call_panel(daily, master))

    assert events.is_empty()
    assert events.schema["ann_date"] == pl.Date


def test_nonordinary_exchangeable_bond_is_excluded() -> None:
    days = [date(2017, 1, 1) + timedelta(days=index) for index in range(30)]
    daily = pl.DataFrame(
        {
            "symbol": ["132001.SH"] * 30,
            "date": days,
            "cb_value": [130.0] * 30,
        }
    )
    master = pl.DataFrame(
        {"symbol": ["132001.SH"], "stk_code": ["600001.SH"]}
    )

    assert study.prepare_call_panel(daily, master).is_empty()


def test_promotion_gate_requires_high_absolute_return() -> None:
    metrics = {
        "tradable_events": 80,
        "announcement_days": 60,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_return": 0.03,
        "mean_excess_return": 0.02,
        "excess_daily_cluster_t": 2.5,
        "positive_excess_years": 3,
        "max_year_positive_excess_share": 0.50,
    }
    assert study.promotion_passed(metrics) is True
    metrics["mean_net_return"] = 0.029
    assert study.promotion_passed(metrics) is False
