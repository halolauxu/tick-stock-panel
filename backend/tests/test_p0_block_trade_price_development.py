from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_block_trade_price_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_block_trade_price", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _detail(symbol: str, event_date: date, price: float) -> dict:
    return {
        "symbol": symbol,
        "trade_date": event_date,
        "price": price,
        "volume_shares": 2_000_000.0,
        "notional_cny": price * 2_000_000.0,
        "buyer": "买方",
        "seller": "卖方",
    }


def _panel(symbols: list[str], event_date: date) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": symbols,
            "date": [event_date] * len(symbols),
            "event_raw_close": [10.0] * len(symbols),
            "event_daily_amount": [100_000_000.0] * len(symbols),
        }
    )


def test_aggregate_classifies_fixed_premium_discount_bands() -> None:
    event_date = date(2020, 1, 1)
    details = pl.DataFrame(
        [
            _detail("A", event_date, 10.2),
            _detail("B", event_date, 9.8),
            _detail("C", event_date, 10.0),
        ]
    )

    result = study.aggregate_events(details, _panel(["A", "B", "C"], event_date))
    categories = dict(zip(result["symbol"], result["category"], strict=True))

    assert categories == {
        "A": "premium_block",
        "B": "discount_block",
        "C": "near_close_control",
    }


def test_aggregate_filters_small_or_immaterial_trades() -> None:
    event_date = date(2020, 1, 1)
    small = _detail("A", event_date, 10.2)
    small["volume_shares"] = 10_000.0
    small["notional_cny"] = 102_000.0

    result = study.aggregate_events(pl.DataFrame([small]), _panel(["A"], event_date))

    assert result.is_empty()


def test_thirty_day_cooldown_uses_last_kept_event() -> None:
    dates = [date(2020, 1, 1), date(2020, 1, 20), date(2020, 1, 31)]
    details = pl.DataFrame([_detail("A", value, 10.2) for value in dates])
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * 3,
            "date": dates,
            "event_raw_close": [10.0] * 3,
            "event_daily_amount": [100_000_000.0] * 3,
        }
    )

    result = study.aggregate_events(details, panel)

    assert result["ann_date"].to_list() == [date(2020, 1, 1), date(2020, 1, 31)]
