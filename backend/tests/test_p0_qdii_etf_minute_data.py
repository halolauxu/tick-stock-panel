from __future__ import annotations

import importlib.util
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = Path(__file__).resolve().parents[2] / "research" / "collect_p0_qdii_etf_minute_data.py"
    spec = importlib.util.spec_from_file_location("p0_qdii_etf_minute_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _session_times(trade_date: date) -> list[datetime]:
    return [
        datetime.combine(trade_date, time(9, 30)) + timedelta(minutes=offset)
        for offset in range(121)
    ] + [
        datetime.combine(trade_date, time(13, 1)) + timedelta(minutes=offset)
        for offset in range(120)
    ]


def test_normalize_keeps_etf_shares_and_removes_non_session_rows() -> None:
    rows = [
        {
            "ts_code": "513100.SH", "trade_time": "2026-08-28 09:30:00",
            "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.05,
            "vol": 12_300.0, "amount": 25_215.0,
        },
        {
            "ts_code": "513100.SH", "trade_time": "2026-08-28 12:00:00",
            "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0,
            "vol": 1.0, "amount": 2.0,
        },
    ]

    frame = collector.normalize_minutes(rows, "513100.SH")

    assert frame.height == 1
    assert frame["volume"][0] == 12_300.0


def test_audit_requires_every_frozen_symbol_and_complete_sessions() -> None:
    trade_date = date(2026, 8, 28)
    rows = []
    daily = []
    for symbol in collector.SYMBOLS:
        daily.append({"symbol": symbol, "date": trade_date})
        for timestamp in _session_times(trade_date):
            rows.append(
                {
                    "symbol": symbol, "datetime": timestamp, "open": 2.0,
                    "high": 2.1, "low": 1.9, "close": 2.0,
                    "volume": 1000.0, "amount": 2000.0,
                }
            )

    result = collector.audit(pl.DataFrame(rows), pl.DataFrame(daily), minimum_common_days=1)

    assert result["status"] == "DATA_QUALIFIED"
    assert result["counts"]["common_complete_days"] == 1
    assert result["returns_evaluated"] is False


def test_audit_fails_when_one_symbol_day_is_incomplete() -> None:
    trade_date = date(2026, 8, 28)
    rows = []
    daily = []
    for symbol in collector.SYMBOLS:
        daily.append({"symbol": symbol, "date": trade_date})
        timestamps = _session_times(trade_date)
        if symbol == collector.SYMBOLS[-1]:
            timestamps = timestamps[:-1]
        for timestamp in timestamps:
            rows.append(
                {
                    "symbol": symbol, "datetime": timestamp, "open": 2.0,
                    "high": 2.1, "low": 1.9, "close": 2.0,
                    "volume": 1000.0, "amount": 2000.0,
                }
            )

    result = collector.audit(pl.DataFrame(rows), pl.DataFrame(daily), minimum_common_days=1)

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["all_symbol_days_are_241_bars"] is False
