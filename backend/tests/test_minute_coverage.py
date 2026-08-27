from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from app.api.data import _safe_aggregate_minute
from app.services import kline_sync
from app.services.kline_sync import (
    _write_minute_partition,
    find_minute_repair_start,
    minute_coverage_summary,
    repair_minute_quality_partitions,
    validate_minute_partitions,
)


def _minute_day(trade_date: date, bars: int, symbols: int = 2) -> pl.DataFrame:
    morning = [
        datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=30)
        + timedelta(minutes=offset)
        for offset in range(121)
    ]
    afternoon = [
        datetime.combine(trade_date, datetime.min.time()).replace(hour=13, minute=1)
        + timedelta(minutes=offset)
        for offset in range(120)
    ]
    post_market = [
        datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=1)
        + timedelta(minutes=offset)
        for offset in range(max(0, bars - 241))
    ]
    timestamps = (morning + afternoon)[:bars] + post_market
    rows = []
    for symbol_index in range(symbols):
        symbol = f"00000{symbol_index + 1}.SZ"
        for timestamp in timestamps:
            rows.append({
                "symbol": symbol,
                "datetime": timestamp,
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


def test_minute_coverage_requires_every_daily_symbol(tmp_path):
    trade_date = date(2026, 8, 13)
    _daily_day(tmp_path, trade_date, symbols=2)
    _write_minute_partition(
        _minute_day(trade_date, 241, symbols=1),
        tmp_path / "kline_minute",
    )

    summary = minute_coverage_summary(tmp_path)
    validation = validate_minute_partitions(tmp_path, trade_date, trade_date)

    assert summary["complete_days"] == 0
    assert summary["incomplete_days"] == 1
    assert validation["valid"] is False
    assert validation["dates"][0]["required_full_symbols"] == 2


def test_repairs_post_market_rows_and_rebuilds_complete_stats(tmp_path):
    trade_date = date(2026, 8, 13)
    _daily_day(tmp_path, trade_date)
    day_dir = tmp_path / "kline_minute" / f"date={trade_date}"
    day_dir.mkdir(parents=True)
    polluted = _minute_day(trade_date, 271)
    polluted.write_parquet(day_dir / "part.parquet")

    before = validate_minute_partitions(tmp_path, trade_date, trade_date)
    assert before["valid"] is False
    assert before["dates"][0]["out_of_regular_session"] == 60
    assert before["dates"][0]["extra_symbols"] == 2

    repaired = repair_minute_quality_partitions(tmp_path, trade_date, trade_date)

    assert repaired == {
        "scanned_days": 1,
        "repaired_days": 1,
        "removed_rows": 60,
        "repaired_dates": ["2026-08-13"],
    }
    after = validate_minute_partitions(tmp_path, trade_date, trade_date)
    assert after["valid"] is True
    assert after["complete_days"] == 1
    stored = pl.read_parquet(day_dir / "part.parquet")
    assert stored.height == 482
    assert stored["datetime"].max() == datetime(2026, 8, 13, 15, 0)
    assert minute_coverage_summary(tmp_path)["complete_days"] == 1


def test_repairs_zero_price_placeholder_symbol(tmp_path):
    trade_date = date(2026, 8, 13)
    _daily_day(tmp_path, trade_date, symbols=2)
    day_dir = tmp_path / "kline_minute" / f"date={trade_date}"
    day_dir.mkdir(parents=True)
    valid = _minute_day(trade_date, 241, symbols=2)
    placeholder = _minute_day(trade_date, 271, symbols=1).with_columns(
        pl.lit("920001.BJ").alias("symbol"),
        *[pl.lit(0.0).alias(column) for column in ("open", "high", "low", "close")],
        pl.lit(0.0).alias("volume"),
        pl.lit(0.0).alias("amount"),
    )
    pl.concat([valid, placeholder]).write_parquet(day_dir / "part.parquet")

    repaired = repair_minute_quality_partitions(tmp_path, trade_date, trade_date)

    assert repaired["removed_rows"] == 271
    stored = pl.read_parquet(day_dir / "part.parquet")
    assert stored.height == 482
    assert "920001.BJ" not in stored["symbol"].unique().to_list()
    assert validate_minute_partitions(tmp_path, trade_date, trade_date)["valid"] is True


def test_persistence_rejects_nonzero_invalid_ohlc(tmp_path):
    trade_date = date(2026, 8, 13)
    invalid = _minute_day(trade_date, 1, symbols=1).with_columns(
        pl.lit(0.5).alias("high"),
    )

    with pytest.raises(ValueError, match="分钟K写入被拒绝"):
        _write_minute_partition(invalid, tmp_path / "kline_minute")

    assert not (tmp_path / "kline_minute").exists()


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
