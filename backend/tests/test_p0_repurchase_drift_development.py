from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_repurchase_drift_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_repurchase_drift", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _event(symbol: str, ann_date: date, process: str, amount: float = 1.0) -> dict:
    return {
        "symbol": symbol,
        "ann_date": ann_date,
        "end_date": ann_date,
        "proc": process,
        "exp_date": None,
        "repurchase_shares": 100.0,
        "repurchase_amount_cny": amount,
        "high_limit": 20.0,
        "low_limit": 10.0,
    }


def test_process_categories_are_frozen_and_termination_has_priority() -> None:
    events = pl.DataFrame(
        [
            _event("A", date(2020, 1, 1), "股东大会通过"),
            _event("F", date(2020, 1, 1), "预案"),
            _event("B", date(2020, 1, 1), "实施"),
            _event("C", date(2020, 1, 1), "完成"),
            _event("D", date(2020, 1, 1), "停止实施"),
            _event("E", date(2020, 1, 1), "未知状态"),
        ]
    )

    result = study.categorize_events(events)
    categories = dict(zip(result["symbol"], result["category"], strict=True))

    assert categories == {
        "A": "proposal_approved",
        "B": "implementation",
        "C": "completion",
        "D": "termination_control",
        "F": "proposal_approved",
    }


def test_category_cooldown_keeps_first_event_after_each_full_year() -> None:
    events = pl.DataFrame(
        [
            _event("A", date(2018, 1, 1), "实施"),
            _event("A", date(2018, 6, 1), "实施"),
            _event("A", date(2019, 1, 1), "实施"),
            _event("A", date(2020, 1, 1), "实施"),
        ]
    )

    result = study.categorize_events(events)

    assert result["ann_date"].to_list() == [
        date(2018, 1, 1),
        date(2019, 1, 1),
        date(2020, 1, 1),
    ]


def test_same_day_details_keep_largest_known_amount() -> None:
    events = pl.DataFrame(
        [
            _event("A", date(2020, 1, 1), "完成", 100.0),
            _event("A", date(2020, 1, 1), "完成", 200.0),
        ]
    )

    result = study.categorize_events(events)

    assert result.height == 1
    assert result["repurchase_amount_cny"][0] == 200.0


def test_market_benchmark_uses_fixed_twenty_day_open_return_median() -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "trade_index": [0, 20, 0, 20],
            "open": [10.0, 11.0, 20.0, 24.0],
            "raw_open": [10.0, 11.0, 20.0, 24.0],
            "amount": [50_000_000.0] * 4,
            "volume": [1_000.0] * 4,
            "excluded_name": [False] * 4,
        }
    )

    result = study.build_market_benchmark(panel)

    assert result.height == 1
    assert result["trade_index"][0] == 0
    assert abs(result["market_median_return"][0] - 0.15) < 1e-12
    assert result["market_symbols"][0] == 2
