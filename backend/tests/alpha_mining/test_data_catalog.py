from __future__ import annotations

# Requirements: AM-S3-001 through AM-S3-011.
from datetime import date, datetime

import polars as pl

from app.alpha_mining.data_catalog import AlphaResearchDataCatalog


def _write(path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_catalog_applies_historical_universe_name_st_and_share_pit(tmp_path) -> None:
    research = tmp_path / "research"
    _write(research / "historical_stock_universe.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "list_date": [date(2020, 1, 1), date(2020, 1, 1)],
        "delist_date": [None, None],
    }))
    _write(research / "historical_stock_names.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "name": ["正常公司", "*ST问题"],
        "start_date": [date(2020, 1, 1), date(2020, 1, 1)],
        "end_date": [None, None],
    }))
    _write(tmp_path / "financials" / "shares" / "part.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "announce_date": [date(2025, 1, 1), date(2025, 1, 1)],
        "period_end": [date(2024, 12, 31), date(2024, 12, 31)],
        "total_shares": [1000.0, 1000.0],
        "float_shares": [800.0, 800.0],
    }))
    _write(tmp_path / "financials" / "metrics" / "part.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"], "announce_date": [date(2025, 1, 1)], "roe": [10.0],
    }))
    day = date(2026, 1, 5)
    enriched = tmp_path / "kline_daily_enriched" / f"date={day.isoformat()}"
    _write(enriched / "part.parquet", pl.DataFrame({"symbol": ["000001.SZ"], "date": [day]}))
    _write(research / "sw_l1_membership.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"], "l1_code": ["801010"], "in_date": [date(2020, 1, 1)], "out_date": [None],
    }))
    _write(research / "sw_l1_daily_context.parquet", pl.DataFrame({
        "l1_code": ["801010"], "date": [day], "industry_momentum_20d": [0.1], "industry_breadth_5d": [0.6],
    }))
    _write(research / "events" / "events.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"], "published_at": [datetime(2026, 1, 4, 10)],
    }))
    catalog = AlphaResearchDataCatalog(tmp_path)
    snapshot = catalog.snapshot(day, day)
    assert snapshot.datasets["historical_universe"].ready is True
    assert snapshot.datasets["financial_pit"].ready is True
    assert snapshot.datasets["industry_pit"].ready is True
    assert snapshot.datasets["event_history"].ready is True
    assert snapshot.datasets["concept_snapshot"].ready is False
    panel = pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"], "date": [day, day], "close": [10.0, 20.0],
    })
    eligible, audit = catalog.apply_formal_pit_context(panel)
    assert eligible["symbol"].to_list() == ["000001.SZ"]
    assert audit["eligible_symbols"] == 1


def test_non_turnover_research_does_not_require_share_history(tmp_path) -> None:
    research = tmp_path / "research"
    _write(research / "historical_stock_universe.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"],
        "list_date": [date(2020, 1, 1)],
        "delist_date": [None],
    }))
    _write(research / "historical_stock_names.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"],
        "name": ["正常公司"],
        "start_date": [date(2020, 1, 1)],
        "end_date": [None],
    }))
    day = date(2026, 1, 5)
    enriched = tmp_path / "kline_daily_enriched" / f"date={day.isoformat()}"
    _write(enriched / "part.parquet", pl.DataFrame({"symbol": ["000001.SZ"], "date": [day]}))
    catalog = AlphaResearchDataCatalog(tmp_path)
    snapshot = catalog.snapshot(day, day)
    assert snapshot.datasets["historical_universe"].ready is True
    assert snapshot.datasets["share_history_pit"].ready is False
    panel = pl.DataFrame({"symbol": ["000001.SZ"], "date": [day], "close": [10.0]})
    eligible, audit = catalog.apply_formal_pit_context(
        panel,
        require_share_history=False,
    )
    assert eligible.height == 1
    assert audit["share_history_required"] is False


def test_financial_context_is_invisible_before_announcement(tmp_path) -> None:
    _write(tmp_path / "financials" / "metrics" / "part.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "announce_date": [date(2026, 1, 10), date(2026, 4, 10)],
        "roe": [8.0, 10.0],
        "revenue_yoy": [5.0, 9.0],
        "net_income_yoy": [4.0, 12.0],
    }))
    panel = pl.DataFrame({
        "symbol": ["000001.SZ"] * 3,
        "date": [date(2026, 1, 9), date(2026, 1, 10), date(2026, 4, 10)],
        "close": [10.0, 10.0, 10.0],
    })
    result, audit = AlphaResearchDataCatalog(tmp_path).attach_financial_context(panel)
    assert result["roe_latest"].to_list() == [None, 8.0, 10.0]
    assert result["financial_revision_revenue"].to_list() == [None, None, 4.0]
    assert result["financial_revision_profit"].to_list() == [None, None, 8.0]
    assert audit["pit_verified"] is True


def test_historical_universe_is_deterministic_at_listing_name_and_share_boundaries(tmp_path) -> None:
    research = tmp_path / "research"
    _write(research / "historical_stock_universe.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"],
        "list_date": [date(2026, 1, 2)],
        "delist_date": [date(2026, 1, 5)],
    }))
    _write(research / "historical_stock_names.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "name": ["正常公司", "ST公司"],
        "start_date": [date(2026, 1, 2), date(2026, 1, 5)],
        "end_date": [date(2026, 1, 4), None],
    }))
    _write(tmp_path / "financials" / "shares" / "part.parquet", pl.DataFrame({
        "symbol": ["000001.SZ"],
        "announce_date": [date(2026, 1, 3)],
        "period_end": [date(2025, 12, 31)],
        "total_shares": [1_000.0],
        "float_shares": [800.0],
    }))
    panel = pl.DataFrame({
        "symbol": ["000001.SZ"] * 5,
        "date": [date(2026, 1, day) for day in range(1, 6)],
        "close": [10.0] * 5,
    })
    catalog = AlphaResearchDataCatalog(tmp_path)
    first, first_audit = catalog.apply_formal_pit_context(panel)
    second, second_audit = catalog.apply_formal_pit_context(panel)
    assert first.equals(second)
    assert first_audit == second_audit
    assert first["date"].to_list() == [date(2026, 1, 3), date(2026, 1, 4)]
