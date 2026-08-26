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


def test_supplemental_sync_is_idempotent_and_partitioned(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "TushareProvider", _Provider)

    first_auction = sync.sync_auction(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    second_auction = sync.sync_auction(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    first_qa = sync.sync_irm_qa(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)
    second_qa = sync.sync_irm_qa(tmp_path, end_date=date(2026, 8, 26), lookback_days=1)

    auction_path = tmp_path / "tushare_supplemental" / "auction" / "date=2026-08-26" / "part.parquet"
    qa_path = tmp_path / "tushare_supplemental" / "irm_qa" / "date=2026-08-26" / "part.parquet"
    assert first_auction == second_auction == 2
    assert first_qa == second_qa == 2
    assert pl.read_parquet(auction_path).height == 2
    assert pl.read_parquet(qa_path).height == 2

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
