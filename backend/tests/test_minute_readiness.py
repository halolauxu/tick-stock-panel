from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from app.services import kline_sync, preferences


def _full_minute_day(symbol: str, day: date) -> pl.DataFrame:
    morning = [datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30) + timedelta(minutes=i) for i in range(121)]
    afternoon = [datetime.combine(day, datetime.min.time()).replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
    stamps = morning + afternoon
    return pl.DataFrame({
        "symbol": [symbol] * len(stamps),
        "datetime": stamps,
        "open": [10.0] * len(stamps),
        "high": [10.1] * len(stamps),
        "low": [9.9] * len(stamps),
        "close": [10.0] * len(stamps),
        "volume": [1.0] * len(stamps),
        "amount": [1000.0] * len(stamps),
    })


def test_probe_configured_minute_day_ready(monkeypatch):
    day = date(2026, 8, 27)

    class Provider:
        def get_minute(self, symbols, **kwargs):
            assert symbols == ["600000.SH", "000001.SZ", "000725.SZ"]
            return _full_minute_day("600000.SH", day)

    monkeypatch.setattr(preferences, "get_minute_data_provider", lambda: "tushare")
    monkeypatch.setattr(
        kline_sync,
        "_resolve_minute_provider",
        lambda _name: (Provider(), False, None),
    )

    result = kline_sync.probe_configured_minute_day(
        ["000001.SZ", "000725.SZ", "600000.SH"], day
    )

    assert result["applicable"] is True
    assert result["ready"] is True
    assert result["rows"] == 241
    assert result["full_symbols"] == 1


def test_probe_configured_minute_day_empty_is_not_ready(monkeypatch):
    day = date(2026, 8, 27)

    class Provider:
        def get_minute(self, symbols, **kwargs):
            return pl.DataFrame()

    monkeypatch.setattr(preferences, "get_minute_data_provider", lambda: "tushare")
    monkeypatch.setattr(
        kline_sync,
        "_resolve_minute_provider",
        lambda _name: (Provider(), False, None),
    )

    result = kline_sync.probe_configured_minute_day(["600000.SH"], day)

    assert result["applicable"] is True
    assert result["ready"] is False
    assert result["rows"] == 0
    assert "尚未返回" in str(result["reason"])


def test_probe_configured_minute_day_does_not_change_tickflow(monkeypatch):
    monkeypatch.setattr(preferences, "get_minute_data_provider", lambda: "tickflow")
    monkeypatch.setattr(
        kline_sync,
        "_resolve_minute_provider",
        lambda _name: (None, True, None),
    )

    result = kline_sync.probe_configured_minute_day(["600000.SH"], date(2026, 8, 27))

    assert result["applicable"] is False
    assert result["ready"] is True
