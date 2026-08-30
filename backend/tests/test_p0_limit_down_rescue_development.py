from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_limit_down_rescue_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_limit_down_rescue", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _row(day: date, *, symbol: str = "000001.SZ", touch: bool = False) -> dict:
    return {
        "symbol": symbol,
        "date": day,
        "raw_low": 9.0 if touch else 9.5,
        "raw_close": 9.30 if touch else 10.0,
        "limit_down_price": 9.0,
        "amount": 100_000_000.0,
        "excluded_name": False,
    }


def test_rescue_requires_first_limit_down_touch_in_sixty_trading_days() -> None:
    start = date(2020, 1, 2)
    rows = [_row(start + timedelta(days=index)) for index in range(10)]
    rows[2] = _row(start + timedelta(days=2), touch=True)
    rows[6] = _row(start + timedelta(days=6), touch=True)

    result = study.build_rescue_events(pl.DataFrame(rows))

    assert result.height == 1
    assert result["ann_date"][0] == start + timedelta(days=2)


def test_rescue_rejects_close_without_two_percent_reopening() -> None:
    row = _row(date(2020, 1, 2), touch=True)
    row["raw_close"] = 9.10

    assert study.build_rescue_events(pl.DataFrame([row])).is_empty()


def test_rescue_rejects_non_main_board_and_risk_warning() -> None:
    day = date(2020, 1, 2)
    growth = _row(day, symbol="300001.SZ", touch=True)
    risk = _row(day, symbol="000001.SZ", touch=True)
    risk["excluded_name"] = True

    result = study.build_rescue_events(pl.DataFrame([growth, risk]))

    assert result.is_empty()
