from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl

from app.api.data import _safe_aggregate_supplemental
from app.services import tushare_supplemental_sync as sync


class _Provider:
    def close(self):
        pass

    def get_auction(self, day: date, session: str) -> pl.DataFrame:
        return pl.DataFrame({
            "symbol": ["600000.SH"],
            "date": [day.isoformat()],
            "session": [session],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.05],
            "volume_shares": [1000.0],
            "amount": [10050.0],
            "vwap": [10.05],
        })

    def get_irm_qa(self, exchange: str, *, pub_start: date, pub_end: date) -> pl.DataFrame:
        return pl.DataFrame({
            "symbol": ["600000.SH" if exchange == "sh" else "000001.SZ"],
            "name": ["样例"],
            "date": [pub_end.isoformat()],
            "trade_date": [pub_start.isoformat()],
            "question": [f"{exchange} question"],
            "answer": ["answer"],
            "pub_time": [f"{pub_end.isoformat()} 18:00:00"],
            "industry": [""],
            "exchange": [exchange.upper()],
        })

    def get_forecast(self, day: date) -> pl.DataFrame:
        return pl.DataFrame({
            "symbol": ["600000.SH"],
            "ann_date": [day],
            "period_end": [date(day.year, 12, 31)],
            "type": ["预增"],
            "p_change_min": [20.0],
            "p_change_max": [30.0],
            "net_profit_min": [100_000_000.0],
            "net_profit_max": [120_000_000.0],
            "last_parent_net": [80_000_000.0],
            "first_ann_date": [day],
            "summary": ["sample"],
            "change_reason": ["sample"],
            "collection_source": ["forecast_by_announcement_day"],
        })


def test_supplemental_sync_is_idempotent_and_partitioned(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "TushareProvider", _Provider)

    first_auction = sync.sync_auction(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    second_auction = sync.sync_auction(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    first_qa = sync.sync_irm_qa(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    second_qa = sync.sync_irm_qa(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    first_forecast = sync.sync_forecast(
        tmp_path, end_date=date(2026, 8, 26), lookback_days=1
    )
    second_forecast = sync.sync_forecast(
        tmp_path, end_date=date(2026, 8, 26), lookback_days=1
    )

    auction_path = tmp_path / "tushare_supplemental" / "auction" / "date=2026-08-26" / "part.parquet"
    qa_path = tmp_path / "tushare_supplemental" / "irm_qa" / "date=2026-08-26" / "part.parquet"
    forecast_path = (
        tmp_path / "event_data" / "forecast" / "year=2026" / "part.parquet"
    )
    forecast_receipt = tmp_path / "event_data" / "forecast" / "sync_status.json"
    assert first_auction == second_auction == 2
    assert first_qa == second_qa == 2
    assert first_forecast == second_forecast == 1
    assert pl.read_parquet(auction_path).height == 2
    assert pl.read_parquet(qa_path).height == 2
    assert pl.read_parquet(forecast_path).height == 1
    assert forecast_receipt.read_text(encoding="utf-8") == (
        '{"schema_version":"tushare-forecast-sync-v1","start_date":"2026-08-26",'
        '"end_date":"2026-08-26","rows_returned":1}'
    )

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    auction_stats = _safe_aggregate_supplemental(repo, "auction")
    qa_stats = _safe_aggregate_supplemental(repo, "irm_qa")
    assert auction_stats == {
        "rows": 2,
        "earliest_date": "2026-08-26",
        "latest_date": "2026-08-26",
        "symbols_covered": 1,
        "trading_days": 1,
    }
    assert qa_stats == {
        "rows": 2,
        "earliest_date": "2026-08-26",
        "latest_date": "2026-08-26",
        "symbols_covered": 2,
        "trading_days": 1,
    }


def test_historical_ranges_are_split_by_day_and_source(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str]] = []
    progress: list[tuple[int, int, str]] = []

    class RecordingProvider(_Provider):
        def get_auction(self, day: date, session: str) -> pl.DataFrame:
            calls.append(("auction", day.isoformat(), session))
            return super().get_auction(day, session)

        def get_irm_qa(self, exchange: str, *, pub_start: date, pub_end: date) -> pl.DataFrame:
            assert pub_start == pub_end
            calls.append(("irm_qa", pub_start.isoformat(), exchange))
            return super().get_irm_qa(exchange, pub_start=pub_start, pub_end=pub_end)

    monkeypatch.setattr(sync, "TushareProvider", RecordingProvider)

    auction_rows = sync.sync_auction_range(
        tmp_path,
        start_date=date(2026, 8, 25),
        end_date=date(2026, 8, 26),
        on_progress=lambda done, total, label: progress.append((done, total, label)),
    )
    qa_rows = sync.sync_irm_qa_range(
        tmp_path,
        start_date=date(2026, 8, 25),
        end_date=date(2026, 8, 26),
        on_progress=lambda done, total, label: progress.append((done, total, label)),
    )

    assert auction_rows == 4
    assert qa_rows == 4
    assert calls == [
        ("auction", "2026-08-25", "open"),
        ("auction", "2026-08-25", "close"),
        ("auction", "2026-08-26", "open"),
        ("auction", "2026-08-26", "close"),
        ("irm_qa", "2026-08-25", "sh"),
        ("irm_qa", "2026-08-25", "sz"),
        ("irm_qa", "2026-08-26", "sh"),
        ("irm_qa", "2026-08-26", "sz"),
    ]
    assert progress[0][:2] == (1, 4)
    assert progress[-1][:2] == (4, 4)
