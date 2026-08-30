from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_top_inst_flow_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_top_inst_flow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _row(
    symbol: str,
    trade_date: date,
    seat_name: str,
    net_buy: float,
    side: str = "0",
    reason: str = "日涨幅偏离值",
) -> dict:
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "seat_name": seat_name,
        "side": side,
        "buy": max(net_buy, 0.0) + 100.0,
        "buy_rate": 1.0,
        "sell": max(-net_buy, 0.0) + 100.0,
        "sell_rate": 1.0,
        "net_buy": net_buy,
        "reason": reason,
    }


def test_duplicate_seat_amount_on_both_rank_sides_counts_once() -> None:
    event_date = date(2020, 1, 1)
    details = pl.DataFrame(
        [
            _row("A", event_date, "机构专用", 1000.0, "0", "理由一"),
            _row("A", event_date, "机构专用", 1000.0, "1", "理由二"),
        ]
    )

    result = study.aggregate_events(details)

    assert result.height == 1
    assert result["category"][0] == "institution_buy"
    assert result["category_net_buy"][0] == 1000.0


def test_institution_and_northbound_directions_are_independent() -> None:
    event_date = date(2020, 1, 1)
    details = pl.DataFrame(
        [
            _row("A", event_date, "机构专用", 1000.0),
            _row("A", event_date, "沪股通专用", -500.0),
        ]
    )

    result = study.aggregate_events(details)

    assert set(result["category"]) == {
        "institution_buy",
        "northbound_sell_control",
    }


def test_twenty_day_cooldown_uses_last_kept_event() -> None:
    details = pl.DataFrame(
        [
            _row("A", date(2020, 1, 1), "机构专用", 1000.0),
            _row("A", date(2020, 1, 10), "机构专用", 1000.0),
            _row("A", date(2020, 1, 21), "机构专用", 1000.0),
        ]
    )

    result = study.aggregate_events(details)

    assert result["ann_date"].to_list() == [date(2020, 1, 1), date(2020, 1, 21)]
