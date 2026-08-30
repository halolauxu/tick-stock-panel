from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_large_order_flow_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_large_flow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _flow(symbol: str, trade_date: date, large_net: float) -> dict:
    positive = max(large_net, 0.0)
    negative = max(-large_net, 0.0)
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "buy_sm_cny": 0.0,
        "sell_sm_cny": 0.0,
        "buy_md_cny": 0.0,
        "sell_md_cny": 0.0,
        "buy_lg_cny": positive,
        "sell_lg_cny": negative,
        "buy_elg_cny": 0.0,
        "sell_elg_cny": 0.0,
        "net_mf_cny": large_net,
    }


def test_categorize_flow_price_divergence_and_continuation():
    event_date = date(2020, 1, 2)
    moneyflow = pl.DataFrame(
        [
            _flow("000001.SZ", event_date, 20_000_000.0),
            _flow("000002.SZ", event_date, 20_000_000.0),
        ]
    )
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "trade_date": [event_date, event_date],
            "event_daily_amount": [100_000_000.0, 100_000_000.0],
            "event_return": [-0.02, 0.02],
        }
    )

    events = study.categorize_events(moneyflow, panel)

    assert events.select("symbol", "category").to_dicts() == [
        {"symbol": "000001.SZ", "category": "flow_price_divergence"},
        {"symbol": "000002.SZ", "category": "flow_price_continuation"},
    ]


def test_categorize_uses_global_symbol_cooldown():
    first = date(2020, 1, 2)
    second = first + timedelta(days=10)
    moneyflow = pl.DataFrame(
        [
            _flow("000001.SZ", first, 20_000_000.0),
            _flow("000001.SZ", second, -20_000_000.0),
        ]
    )
    panel = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": [first, second],
            "event_daily_amount": [100_000_000.0, 100_000_000.0],
            "event_return": [-0.02, -0.02],
        }
    )

    events = study.categorize_events(moneyflow, panel)

    assert events.height == 1
    assert events.row(0, named=True)["ann_date"] == first


def test_promotion_requires_zero_unresolved_exits():
    metrics = {
        "tradable_events": 500,
        "announcement_days": 200,
        "tradable_rate": 0.95,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 1,
        "mean_net_return": 0.01,
        "mean_excess_return": 0.008,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.4,
    }

    assert study._promotion(metrics, True) is False
