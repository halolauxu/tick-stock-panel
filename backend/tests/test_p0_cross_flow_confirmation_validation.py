from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_cross_flow_confirmation_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_cross_flow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _margin(symbol: str, event_date: date, growth: float = 0.10) -> dict:
    return {
        "symbol": symbol,
        "trade_date": event_date,
        "rzye": 110_000_000.0,
        "rzmre": 20_000_000.0,
        "previous_rzye": 100_000_000.0,
        "margin_balance_change": growth,
        "margin_dates_adjacent": True,
    }


def _flow(symbol: str, event_date: date, large_net: float = 20_000_000.0) -> dict:
    return {
        "symbol": symbol,
        "trade_date": event_date,
        "buy_lg_cny": max(large_net, 0.0),
        "buy_elg_cny": 0.0,
        "sell_lg_cny": max(-large_net, 0.0),
        "sell_elg_cny": 0.0,
    }


def test_cross_flow_requires_both_frozen_component_signals():
    event_date = date(2022, 1, 4)
    margin = pl.DataFrame(
        [
            _margin("000001.SZ", event_date),
            _margin("000002.SZ", event_date),
            _margin("000003.SZ", event_date, growth=0.01),
        ]
    )
    moneyflow = pl.DataFrame(
        [
            _flow("000001.SZ", event_date),
            _flow("000002.SZ", event_date, large_net=5_000_000.0),
            _flow("000003.SZ", event_date),
        ]
    )
    day = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": [event_date] * 3,
            "event_daily_amount": [100_000_000.0] * 3,
            "event_return": [-0.02] * 3,
        }
    )

    events, margin_count, flow_count = study.build_cross_flow_events(margin, moneyflow, day)

    assert margin_count == 2
    assert flow_count == 2
    assert events.select("symbol", "category").to_dicts() == [
        {"symbol": "000001.SZ", "category": study.CATEGORY}
    ]


def test_cross_flow_cooldown_is_applied_after_intersection():
    first = date(2022, 1, 4)
    second = first + timedelta(days=10)
    margin = pl.DataFrame([_margin("000001.SZ", first), _margin("000001.SZ", second)])
    moneyflow = pl.DataFrame([_flow("000001.SZ", first), _flow("000001.SZ", second)])
    day = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": [first, second],
            "event_daily_amount": [100_000_000.0, 100_000_000.0],
            "event_return": [-0.02, -0.02],
        }
    )

    events, _, _ = study.build_cross_flow_events(margin, moneyflow, day)

    assert events.height == 1
    assert events.row(0, named=True)["ann_date"] == first


def test_validation_gate_requires_zero_unresolved_and_two_positive_years():
    metrics = {
        "tradable_events": 300,
        "announcement_days": 150,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_return": 0.011,
        "mean_excess_return": 0.008,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 2,
        "max_year_positive_excess_share": 0.55,
    }

    assert study.validation_passed(metrics) is True
    assert study.validation_passed({**metrics, "unresolved_exits": 1}) is False
    assert study.validation_passed({**metrics, "positive_excess_years": 1}) is False
