from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from app.services.serenity_event_replay import (
    DEFAULT_BENCHMARK,
    EventReplayStore,
    _normalize_local_rows,
    build_event_report,
    classify_event_title,
    collect_research_prices,
    initialize_event_replay,
    materialize_event_dates,
    persist_model_review,
    resolve_local_price_window,
    settle_discovery_outcomes,
)
from app.services.serenity_pilot import PilotStore


def _seed_universe(store: PilotStore, size: int = 100) -> None:
    rows = []
    for index in range(size):
        code = f"{index + 1:06d}"
        symbol = f"{code}.SZ"
        chain_index = index % 3
        rows.append(
            [
                symbol,
                code,
                f"样本{index + 1}",
                f"chain_{chain_index}",
                f"产业链{chain_index}",
                "candidate",
                index + 1,
                "mid",
                1,
                "[]",
                1_000_000_000.0,
                100_000_000.0,
                date(2026, 8, 26),
            ]
        )
    store.connection.executemany(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _initialized_store(tmp_path: Path) -> EventReplayStore:
    source_root = tmp_path / "source"
    source = PilotStore(source_root)
    try:
        _seed_universe(source)
    finally:
        source.close()
    root = tmp_path / "event"
    initialize_event_replay(
        root,
        source_root,
        start_date=date(2025, 8, 26),
        end_date=date(2026, 8, 27),
    )
    return EventReplayStore(root)


def _insert_price(
    store: EventReplayStore,
    symbol: str,
    day: date,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    pre_close: float,
    volume: float = 1_000.0,
    factor: float = 1.0,
    asset_type: str = "stock",
) -> None:
    store.connection.execute(
        "INSERT INTO research_daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            symbol,
            day,
            open_,
            high,
            low,
            close,
            pre_close,
            volume,
            100_000.0,
            factor,
            asset_type,
            datetime(2026, 8, 28, 12, 0),
        ],
    )


def _insert_event(store: EventReplayStore, published_at: datetime) -> None:
    store.connection.execute(
        "INSERT INTO announcements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "event-1",
            "000001.SZ",
            published_at,
            "关于获得重大合同的公告",
            "https://example.test/event-1.pdf",
            10.0,
            "measured",
            None,
            datetime(2026, 8, 28, 12, 0),
        ],
    )
    store.connection.execute(
        "INSERT INTO event_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "event-1",
            "event-1",
            "000001.SZ",
            published_at,
            "关于获得重大合同的公告",
            "ORDER_CONTRACT",
            '["ORDER_CONTRACT"]',
            "LONG_CANDIDATE",
            "metadata-hash",
            None,
            None,
            "DISCOVERY_ONLY_REVIEW_REQUIRED",
            datetime(2026, 8, 28, 12, 0),
        ],
    )
    store.connection.execute(
        "INSERT INTO document_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "event-1",
            "sha",
            1,
            100,
            100,
            100,
            0,
            0,
            0,
            0,
            0.0,
            None,
            1,
            "ok",
            datetime(2026, 8, 28, 12, 0),
        ],
    )


def test_title_classifier_is_discovery_only_and_risk_aware() -> None:
    assert classify_event_title("2025年年度报告") is None
    positive = classify_event_title("关于收到重大项目中标通知书的公告")
    assert positive == {
        "primary_event_type": "ORDER_CONTRACT",
        "event_types": ["ORDER_CONTRACT"],
        "polarity": "LONG_CANDIDATE",
        "status": "DISCOVERY_ONLY_REVIEW_REQUIRED",
    }
    mixed = classify_event_title("关于重大合同终止及风险提示的公告")
    assert mixed is not None
    assert mixed["polarity"] == "MIXED_REVIEW"
    assert "RISK_INVALIDATION" in mixed["event_types"]


def test_initialization_freezes_100_company_2_to_10_day_contract(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path)
    try:
        manifest = store.get_meta("event_manifest")
        assert manifest["universe_size"] == 100
        assert manifest["strategy_contract"]["horizons"] == [2, 3, 5, 10]
        assert manifest["strategy_contract"]["entry"] == "next_global_trading_day_open"
        assert manifest["strategy_contract"]["title_classification_is_trade_signal"] is False
        assert manifest["qualification"]["status"] == "RETROSPECTIVE_EVENT_DISCOVERY_NOT_ALPHA"
    finally:
        store.close()


def test_daily_normalization_rejects_invalid_ohlc() -> None:
    rows = [
        {
            "symbol": "000001.SZ",
            "date": date(2026, 8, 25),
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "pre_close": 9.8,
            "volume": 100,
            "amount": 1000,
        },
        {
            "symbol": "000001.SZ",
            "date": date(2026, 8, 26),
            "open": 10,
            "high": 9,
            "low": 8,
            "close": 10.5,
            "pre_close": 10.5,
            "volume": 100,
            "amount": 1000,
        },
    ]
    values = _normalize_local_rows(rows, "stock")
    assert len(values) == 1
    assert values[0][9] == 1.0


def test_local_one_year_prices_are_reused_and_collection_is_idempotent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source = PilotStore(source_root)
    try:
        _seed_universe(source)
    finally:
        source.close()
    data_dir = tmp_path / "data"
    days = [date(2025, 8, 26), date(2026, 2, 26), date(2026, 8, 26)]
    for day_index, day in enumerate(days):
        stock_partition = data_dir / "kline_daily_enriched" / f"date={day}"
        stock_partition.mkdir(parents=True)
        pl.DataFrame(
            [
                {
                    "symbol": f"{index + 1:06d}.SZ",
                    "date": day,
                    "open": 10.0 + day_index,
                    "high": 11.0 + day_index,
                    "low": 9.0 + day_index,
                    "close": 10.5 + day_index,
                    "volume": 1_000.0,
                    "amount": 10_000.0,
                }
                for index in range(100)
            ]
        ).write_parquet(stock_partition / "part.parquet")
        index_partition = data_dir / "kline_index_daily" / f"date={day}"
        index_partition.mkdir(parents=True)
        pl.DataFrame(
            [
                {
                    "symbol": DEFAULT_BENCHMARK,
                    "date": day,
                    "open": 4_000.0 + day_index,
                    "high": 4_010.0 + day_index,
                    "low": 3_990.0 + day_index,
                    "close": 4_005.0 + day_index,
                    "volume": 1_000_000.0,
                    "amount": 100_000_000.0,
                }
            ]
        ).write_parquet(index_partition / "part.parquet")
    start, end = resolve_local_price_window(data_dir)
    root = tmp_path / "event"
    initialize_event_replay(root, source_root, start_date=start, end_date=end)
    store = EventReplayStore(root)
    try:
        first = collect_research_prices(store, data_dir)
        second = collect_research_prices(store, data_dir)
        coverage = store.connection.execute(
            "SELECT count(*), count(*) FILTER (WHERE status='ok') FROM price_collection_status"
        ).fetchone()

        assert (start, end) == (days[0], days[-1])
        assert first["queried"] == 101
        assert first["inserted_rows"] == 303
        assert first["failures"] == 0
        assert first["validation_split"]["training_sessions"] == 2
        assert first["validation_split"]["validation_sessions"] == 1
        assert second["queried"] == 0
        assert second["inserted_rows"] == 0
        assert second["failures"] == 0
        assert second["validation_split"] == first["validation_split"]
        assert coverage == (101, 101)
    finally:
        store.close()


def test_event_settlement_uses_next_open_adjusted_prices_and_2_3_5_10_days(
    tmp_path: Path,
) -> None:
    store = _initialized_store(tmp_path)
    try:
        first_day = date(2026, 8, 3)
        trading_days = [first_day + timedelta(days=index) for index in range(12)]
        closes = [100.0 + index for index in range(12)]
        for index, day in enumerate(trading_days):
            _insert_price(
                store,
                DEFAULT_BENCHMARK,
                day,
                open_=closes[index],
                high=closes[index] + 1,
                low=closes[index] - 1,
                close=closes[index],
                pre_close=closes[max(0, index - 1)],
                asset_type="index",
            )
            raw_open = 10.0 + index
            raw_close = raw_open + 0.5
            factor = 1.0 if index < 4 else 0.5
            _insert_price(
                store,
                "000001.SZ",
                day,
                open_=raw_open,
                high=raw_close + 0.5,
                low=raw_open - 0.5,
                close=raw_close,
                pre_close=raw_open - 0.5,
                factor=factor,
            )
        _insert_event(store, datetime(2026, 8, 3, 20, 0))

        dates = materialize_event_dates(store)
        outcomes = settle_discovery_outcomes(store)
        rows = store.connection.execute(
            """
            SELECT horizon, decision_date, entry_date, exit_date, gross_return, net_return, status
            FROM event_discovery_outcomes ORDER BY horizon
            """
        ).fetchall()

        assert dates == {"updated": 1, "unresolved": 0}
        assert outcomes == {"events": 1, "settled": 4, "pending": 0, "unexecutable": 0}
        assert [row[0] for row in rows] == [2, 3, 5, 10]
        assert rows[0][1:4] == (trading_days[0], trading_days[1], trading_days[2])
        expected_gross = (12.5 / 11.0) - 1
        assert rows[0][4] == pytest.approx(expected_gross)
        assert rows[0][5] == pytest.approx(expected_gross - 0.002)
        assert rows[0][6] == "SETTLED"
        # The split-like factor change is applied instead of producing a false raw-price gain.
        assert rows[2][4] < 0
    finally:
        store.close()


def test_locked_next_open_is_not_assumed_executable(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path)
    try:
        days = [date(2026, 8, 3) + timedelta(days=index) for index in range(12)]
        for index, day in enumerate(days):
            _insert_price(
                store,
                DEFAULT_BENCHMARK,
                day,
                open_=100 + index,
                high=101 + index,
                low=99 + index,
                close=100 + index,
                pre_close=99 + index,
                asset_type="index",
            )
            one_price = index == 1
            _insert_price(
                store,
                "000001.SZ",
                day,
                open_=11 if one_price else 10,
                high=11 if one_price else 10.5,
                low=11 if one_price else 9.5,
                close=11 if one_price else 10,
                pre_close=10,
            )
        _insert_event(store, datetime(2026, 8, 3, 20, 0))
        materialize_event_dates(store)

        result = settle_discovery_outcomes(store)

        assert result["unexecutable"] == 4
        statuses = store.connection.execute(
            "SELECT DISTINCT status FROM event_discovery_outcomes"
        ).fetchall()
        assert statuses == [("UNEXECUTABLE_NEXT_OPEN",)]
    finally:
        store.close()


def test_model_review_is_immutable_and_report_does_not_claim_alpha(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path)
    try:
        first = persist_model_review(
            store,
            event_id="event-1",
            input_hash="input-1",
            model="deepseek-v4-flash",
            raw_output={"score": 42},
            review_status="DATA_INSUFFICIENT",
        )
        cached = persist_model_review(
            store,
            event_id="event-1",
            input_hash="input-1",
            model="deepseek-v4-flash",
            raw_output={"score": 42},
            review_status="DATA_INSUFFICIENT",
        )
        with pytest.raises(RuntimeError, match="immutable"):
            persist_model_review(
                store,
                event_id="event-1",
                input_hash="input-1",
                model="deepseek-v4-flash",
                raw_output={"score": 43},
                review_status="REVIEWED",
            )
        report = build_event_report(store)

        assert first["status"] == "stored"
        assert cached["status"] == "cached"
        assert report["alpha_status"] == "UNVERIFIED_ALPHA"
        assert report["trade_signals"] == 0
        assert json.loads((store.root / "event-report.json").read_text())["model_reviews"] == 1
    finally:
        store.close()
