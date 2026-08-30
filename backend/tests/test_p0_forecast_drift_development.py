from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_forecast_drift_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_forecast_drift", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _event(
    symbol: str,
    event_type: str,
    minimum_change: float | None,
    maximum_change: float | None,
    minimum_profit: float,
) -> dict:
    return {
        "symbol": symbol,
        "ann_date": date(2020, 1, 31),
        "period_end": date(2019, 12, 31),
        "type": event_type,
        "p_change_min": minimum_change,
        "p_change_max": maximum_change,
        "net_profit_min": minimum_profit,
        "net_profit_max": minimum_profit + 1,
        "last_parent_net": 1.0,
        "first_ann_date": date(2020, 1, 31),
    }


def test_categories_are_mutually_exclusive_and_turnaround_has_priority() -> None:
    frame = pl.DataFrame(
        [
            _event("A", "扭亏", 150.0, 200.0, 10.0),
            _event("A", "预增", 150.0, 200.0, 10.0),
            _event("B", "预增", 120.0, 150.0, 10.0),
            _event("C", "略增", 60.0, 80.0, 10.0),
            _event("D", "续盈", 20.0, 30.0, 10.0),
            _event("E", "预减", -30.0, -10.0, 10.0),
        ]
    )

    result = study.categorize_events(frame)
    categories = dict(zip(result["symbol"], result["category"], strict=True))

    assert result.height == 5
    assert categories == {
        "A": "turnaround",
        "B": "growth_ge_100",
        "C": "growth_50_100",
        "D": "growth_0_50",
        "E": "negative_control",
    }


def test_nonfirst_forecast_is_excluded() -> None:
    row = _event("A", "预增", 100.0, 120.0, 10.0)
    row["first_ann_date"] = date(2020, 1, 1)

    result = study.categorize_events(pl.DataFrame([row]))

    assert result.is_empty()


def test_entry_mapping_is_strictly_after_weekend_announcement() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["A"],
            "ann_date": [date(2020, 1, 4)],
            "category": ["turnaround"],
        }
    )
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)],
            "trade_index": [0, 1, 2],
        }
    )

    result = study.map_entry_indices(events, panel)

    assert result["mapped_entry_date"][0] == date(2020, 1, 6)
    assert result["trade_index"][0] == 1
    assert result["planned_exit_index"][0] == 11


def test_trade_builder_executes_next_open_and_fixed_ten_day_exit() -> None:
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(20)]
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * len(dates),
            "date": dates,
            "trade_index": list(range(len(dates))),
            "raw_open": [100.0] * len(dates),
            "raw_high": [101.0] * len(dates),
            "raw_low": [99.0] * len(dates),
            "raw_close": [100.0] * len(dates),
            "open": [100.0] * len(dates),
            "amount": [50_000_000.0] * len(dates),
            "volume": [1_000.0] * len(dates),
            "excluded_name": [False] * len(dates),
            "limit_up_price": [110.0] * len(dates),
            "limit_down_price": [90.0] * len(dates),
        }
    )
    events = pl.DataFrame(
        {
            "symbol": ["A"],
            "ann_date": [date(2020, 1, 2)],
            "category": ["turnaround"],
        }
    )

    result = study.build_trades(events, panel)

    assert result["entry_date"][0] == date(2020, 1, 3)
    assert result["actual_exit_date"][0] == date(2020, 1, 13)
    assert result["exit_delay"][0] == 0
    assert result["tradable"][0] is True
    assert result["net_return"][0] < 0
