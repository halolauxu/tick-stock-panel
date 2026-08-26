from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl

from app.api.data import _safe_aggregate_minute
from app.services import kline_sync
from app.services.kline_sync import (
    _write_minute_partition,
    find_minute_repair_start,
    minute_coverage_summary,
)


def _minute_day(trade_date: date, bars: int, symbols: int = 2) -> pl.DataFrame:
    rows = []
    for symbol_index in range(symbols):
        symbol = f"00000{symbol_index + 1}.SZ"
        for offset in range(bars):
            rows.append({
                "symbol": symbol,
                "datetime": datetime.combine(trade_date, datetime.min.time()) + timedelta(minutes=offset),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "amount": 1.0,
            })
    return pl.DataFrame(rows)


def _daily_day(data_dir, trade_date: date, symbols: int = 2) -> None:
    out = data_dir / "kline_daily" / f"date={trade_date}" / "part.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "symbol": [f"00000{i + 1}.SZ" for i in range(symbols)],
        "date": [trade_date] * symbols,
    }).write_parquet(out)


def test_minute_coverage_distinguishes_landed_and_complete_days(tmp_path):
    leading = date(2026, 8, 12)
    complete = date(2026, 8, 13)
    trailing = date(2026, 8, 14)
    minute_dir = tmp_path / "kline_minute"
    for trade_date in (leading, complete, trailing):
        _daily_day(tmp_path, trade_date)

    _write_minute_partition(_minute_day(leading, 90), minute_dir)
    _write_minute_partition(_minute_day(complete, 241), minute_dir)
    _write_minute_partition(_minute_day(trailing, 90), minute_dir)

    summary = minute_coverage_summary(tmp_path)

    assert summary is not None
    assert summary["trading_days"] == 3
    assert summary["complete_days"] == 1
    assert summary["incomplete_days"] == 2
    assert summary["latest_complete_date"] == "2026-08-13"
    assert summary["rows"] == (90 + 241 + 90) * 2
    # Leading boundary is deliberately ignored; only the internal/trailing gap is repaired.
    assert find_minute_repair_start(tmp_path) == trailing

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    status = _safe_aggregate_minute(repo)
    assert status is not None
    assert status["complete_days"] == 1
    assert "dates" not in status


def test_streaming_custom_provider_honors_time_segments(monkeypatch):
    class StreamingProvider:
        def __init__(self) -> None:
            self.windows: list[tuple[datetime | None, datetime | None]] = []

        def stream_minute(self, symbols, start_time, end_time, **kwargs):
            self.windows.append((start_time, end_time))
            kwargs["on_batch"](_minute_day(date(2026, 8, 13), 1, symbols=1))
            kwargs["on_chunk_done"](len(symbols), len(symbols))

        def get_minute(self, *args, **kwargs):
            raise AssertionError("persistent streaming path must not accumulate get_minute")

    provider = StreamingProvider()
    monkeypatch.setattr(kline_sync.preferences, "get_minute_data_provider", lambda: "streaming")
    monkeypatch.setattr("app.data_providers.custom.provider_has_dataset", lambda *_: True)
    monkeypatch.setattr("app.data_providers.custom.get_provider", lambda *_: provider)
    batches: list[pl.DataFrame] = []
    segments: list[tuple[int, int, str]] = []

    result = kline_sync.sync_minute_batch(
        ["000001.SZ", "000002.SZ"],
        start_time=datetime(2026, 6, 1),
        end_time=datetime(2026, 7, 16),
        segment_trading_days=20,
        on_segment=batches.append,
        on_segment_done=lambda current, total, label: segments.append((current, total, label)),
    )

    assert result.is_empty()
    assert len(provider.windows) == 2
    assert len(batches) == 2
    assert segments[-1][:2] == (2, 2)
