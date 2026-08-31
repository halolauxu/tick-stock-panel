from __future__ import annotations

import importlib.util
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_gold_etf_minute_data.py"
    )
    spec = importlib.util.spec_from_file_location("p0_gold_etf_minute_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _session_times(trade_date: date) -> list[datetime]:
    morning = [
        datetime.combine(trade_date, time(9, 30)) + timedelta(minutes=offset)
        for offset in range(121)
    ]
    afternoon = [
        datetime.combine(trade_date, time(13, 1)) + timedelta(minutes=offset)
        for offset in range(120)
    ]
    return morning + afternoon


def test_normalize_minutes_keeps_etf_volume_in_shares_and_regular_session() -> None:
    rows = [
        {
            "ts_code": "518880.SH",
            "trade_time": "2026-08-28 09:30:00",
            "open": 9.0,
            "high": 9.1,
            "low": 8.9,
            "close": 9.05,
            "vol": 12_300.0,
            "amount": 111_315.0,
        },
        {
            "ts_code": "518880.SH",
            "trade_time": "2026-08-28 12:00:00",
            "open": 9.0,
            "high": 9.0,
            "low": 9.0,
            "close": 9.0,
            "vol": 1.0,
            "amount": 9.0,
        },
    ]

    frame = collector.normalize_minutes(rows, "518880.SH")

    assert frame.height == 1
    assert frame["volume"][0] == 12_300.0
    assert frame["datetime"][0] == datetime(2026, 8, 28, 9, 30)


def test_audit_requires_four_complete_241_bar_sessions_and_daily_keys() -> None:
    rows = []
    daily_rows = []
    for trade_date in (date(2026, 8, 27), date(2026, 8, 28)):
        for symbol in collector.SYMBOLS:
            daily_rows.append({"symbol": symbol, "date": trade_date})
            for timestamp in _session_times(trade_date):
                rows.append(
                    {
                        "symbol": symbol,
                        "datetime": timestamp,
                        "open": 9.0,
                        "high": 9.1,
                        "low": 8.9,
                        "close": 9.0,
                        "volume": 1000.0,
                        "amount": 9000.0,
                    }
                )
    result = collector.audit(
        pl.DataFrame(rows), pl.DataFrame(daily_rows), minimum_common_days=2
    )

    assert result["status"] == "DATA_QUALIFIED"
    assert result["counts"]["common_complete_days"] == 2
    assert result["returns_evaluated"] is False
    assert result["strategy_metrics_computed"] is False


def test_audit_fails_closed_on_incomplete_symbol_day() -> None:
    trade_date = date(2026, 8, 28)
    rows = []
    daily_rows = []
    for symbol in collector.SYMBOLS:
        daily_rows.append({"symbol": symbol, "date": trade_date})
        timestamps = _session_times(trade_date)
        if symbol == collector.SYMBOLS[-1]:
            timestamps = timestamps[:-1]
        for timestamp in timestamps:
            rows.append(
                {
                    "symbol": symbol,
                    "datetime": timestamp,
                    "open": 9.0,
                    "high": 9.1,
                    "low": 8.9,
                    "close": 9.0,
                    "volume": 1000.0,
                    "amount": 9000.0,
                }
            )

    result = collector.audit(
        pl.DataFrame(rows), pl.DataFrame(daily_rows), minimum_common_days=1
    )

    assert result["status"] == "DATA_GAP"
    assert result["checks"]["all_symbol_days_are_241_bars"] is False
