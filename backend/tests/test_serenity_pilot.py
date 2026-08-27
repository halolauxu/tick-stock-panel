from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import serenity_pilot
from app.services.serenity_pilot import (
    CHAIN_SPECS,
    CninfoClient,
    PilotStore,
    _historical_decision_dates,
    _score_serenity,
    analyze_pdf,
    collect_main_business,
    extract_fact_candidates,
    freeze_daily_decisions,
    select_universe,
)


def _write_partition(root: Path, day: date, rows: list[dict]) -> None:
    partition = root / f"date={day.isoformat()}"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition / "part.parquet")


def test_extract_fact_candidates_requires_evidence_and_number() -> None:
    facts = extract_fact_candidates(
        [
            "公司计划扩建年产能1000吨,项目投资2亿元。",
            "公司持续关注市场发展。",
            "产品已通过客户认证并开始批量供货。",
        ],
        "a-1",
    )

    assert {(fact["category"], fact["page_number"]) for fact in facts} == {
        ("capacity_ramp", 1),
        ("capex_project", 1),
        ("demand_order", 3),
        ("customer_validation", 3),
    }
    assert all(fact["review_status"] == "UNVALIDATED" for fact in facts)


def test_score_serenity_keeps_missing_unknown() -> None:
    score, eligible = _score_serenity({"revenue_yoy": None})
    assert score is None
    assert eligible is False

    score, eligible = _score_serenity(
        {
            "revenue_yoy": 30.0,
            "net_income_yoy": 25.0,
            "roe": 15.0,
            "gross_margin": 35.0,
            "debt_to_asset_ratio": 30.0,
            "pb": 2.0,
            "momentum_60d": 0.15,
            "amount_ratio_5d": 0.4,
        }
    )
    assert score is not None and score >= 12
    assert eligible is True


def test_historical_dates_use_explicit_trading_partitions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    trading_days = [
        date(2026, 8, 14),
        date(2026, 8, 17),
        date(2026, 8, 18),
        date(2026, 8, 19),
        date(2026, 8, 20),
        date(2026, 8, 21),
        date(2026, 8, 24),
        date(2026, 8, 25),
        date(2026, 8, 26),
    ]
    for day in trading_days:
        _write_partition(data_dir / "kline_daily_enriched", day, [{"symbol": "000001.SZ"}])

    selected = _historical_decision_dates(data_dir, date(2026, 8, 26), 7)

    assert selected == trading_days[-7:]


def test_historical_dates_fail_when_partitions_are_insufficient(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_partition(
        data_dir / "kline_daily_enriched",
        date(2026, 8, 26),
        [{"symbol": "000001.SZ"}],
    )

    with pytest.raises(RuntimeError, match="only 1 are available"):
        _historical_decision_dates(data_dir, date(2026, 8, 26), 7)


def test_historical_run_freezes_window_and_leakage_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    decision_dates = [
        date(2026, 8, 18),
        date(2026, 8, 19),
        date(2026, 8, 20),
        date(2026, 8, 21),
        date(2026, 8, 24),
        date(2026, 8, 25),
        date(2026, 8, 26),
    ]
    for day in decision_dates:
        _write_partition(data_dir / "kline_daily_enriched", day, [{"symbol": "000001.SZ"}])

    def fake_universe(_data_dir: Path, as_of: date):
        rows = []
        for index in range(100):
            spec = CHAIN_SPECS[0 if index < 34 else 1 if index < 67 else 2]
            code = f"{index + 1:06d}"
            rows.append(
                {
                    "symbol": f"{code}.SZ",
                    "code": code,
                    "name": f"样本{index + 1}",
                    "chain_id": spec.id,
                    "chain_name": spec.name,
                    "chain_role": spec.role,
                    "sample_rank": index + 1,
                    "market_cap_bucket": "mid",
                    "concept_score": 5,
                    "concepts": spec.required_any[0],
                    "market_cap": 5_000_000_000.0,
                    "amount": 200_000_000.0,
                    "source_as_of": as_of.isoformat(),
                }
            )
        return as_of, rows

    monkeypatch.setattr(serenity_pilot, "select_universe", fake_universe)
    monkeypatch.setattr(
        serenity_pilot,
        "collect_documents",
        lambda *_args: {
            "queried_companies": 100,
            "query_failures": 0,
            "discovered_documents": 0,
            "downloaded_documents": 0,
            "downloaded_bytes": 0,
        },
    )
    monkeypatch.setattr(
        serenity_pilot,
        "collect_main_business",
        lambda *_args: {"queried": 0, "covered_companies": 0, "rows": 0, "failures": 0},
    )
    monkeypatch.setattr(
        serenity_pilot,
        "freeze_daily_decisions",
        lambda _store, _data_dir, value: {"status": "frozen", "date": value.isoformat()},
    )
    monkeypatch.setattr(
        serenity_pilot,
        "settle_outcomes",
        lambda *_args, **_kwargs: {"settled": 0, "pending": 0},
    )

    result = serenity_pilot.run_historical(
        tmp_path / "pilot",
        data_dir,
        end_date=date(2026, 8, 26),
        trading_days=7,
    )

    replay = result["report"]["qualification"]["historical_replay"]
    assert result["decision_dates"] == [value.isoformat() for value in decision_dates]
    assert result["report"]["period"] == {"start": "2026-08-18", "end": "2026-08-26"}
    assert replay["status"] == "RETROSPECTIVE_ENGINEERING_SAMPLE_NOT_CLEAN_ROOM"
    assert replay["concept_membership"] == "UNRESOLVED_CURRENT_SNAPSHOT"
    assert replay["pdf_facts_in_strategy_score"] is False


def test_select_universe_is_exact_and_cap_stratified(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "ext_data" / "ext_gn_ths").mkdir(parents=True)
    (data_dir / "instruments").mkdir(parents=True)
    market_date = date(2026, 8, 26)
    concepts: list[dict] = []
    instruments: list[dict] = []
    daily: list[dict] = []
    chain_concepts = ("光刻机", "共封装光学(CPO)", "固态电池")
    offsets = (0, 1000, 2000)
    for chain_index, (concept, offset) in enumerate(zip(chain_concepts, offsets, strict=True)):
        for index in range(60):
            code = f"{offset + index + 1:06d}"
            symbol = f"{code}.SZ"
            concepts.append(
                {"symbol": symbol, "股票简称": f"样本{chain_index}-{index}", "所属概念": concept}
            )
            instruments.append(
                {
                    "symbol": symbol,
                    "listing_date": "2020-01-01",
                    "total_shares": float(200_000_000 + index * 10_000_000),
                }
            )
            daily.append(
                {
                    "symbol": symbol,
                    "close": 20.0,
                    "amount": 200_000_000.0,
                }
            )
    pl.DataFrame(concepts).write_parquet(data_dir / "ext_data" / "ext_gn_ths" / "part.parquet")
    pl.DataFrame(instruments).write_parquet(data_dir / "instruments" / "instruments.parquet")
    _write_partition(data_dir / "kline_daily_enriched", market_date, daily)

    source_date, universe = select_universe(data_dir, market_date)

    assert source_date == market_date
    assert len(universe) == 100
    assert {spec.id: sum(row["chain_id"] == spec.id for row in universe) for spec in CHAIN_SPECS} == {
        "semiconductor_frontend": 34,
        "ai_compute_infrastructure": 33,
        "lithium_materials_control": 33,
    }
    for spec in CHAIN_SPECS:
        buckets = {
            row["market_cap_bucket"] for row in universe if row["chain_id"] == spec.id
        }
        assert buckets == {"small", "mid", "large"}


def test_analyze_pdf_measures_text_without_ocr(tmp_path: Path) -> None:
    import pymupdf as fitz

    pdf_path = tmp_path / "sample.pdf"
    text_path = tmp_path / "sample.txt"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Customer order 100 units and project investment 20 million.")
    document.save(pdf_path)
    document.close()

    metrics, facts = analyze_pdf(pdf_path, text_path, "pdf-1", max_ocr_pages=0)

    assert metrics["pages"] == 1
    assert metrics["pdf_bytes"] == pdf_path.stat().st_size
    assert metrics["embedded_text_bytes"] > 0
    assert metrics["ocr_pages"] == 0
    assert metrics["parse_status"] == "ok"
    assert text_path.exists()
    assert facts == []


def test_cninfo_null_announcements_is_a_valid_empty_window(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"totalAnnouncement": 0, "announcements": None}

    client = CninfoClient(min_interval_s=0)
    monkeypatch.setattr(client, "org_map", lambda: {"300975": "gssz000300975"})
    monkeypatch.setattr(client._http, "post", lambda *_args, **_kwargs: EmptyResponse())
    try:
        rows = client.announcements("300975", date(2026, 8, 18), date(2026, 8, 26))
    finally:
        client.close()

    assert rows == []


def test_cninfo_paginates_deduplicates_and_uses_china_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PageResponse:
        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"totalAnnouncement": 31, "announcements": self._rows}

    timestamp = int(datetime(2026, 8, 1, 8, 0, tzinfo=CN_TZ).timestamp() * 1000)
    first_page = [
        {
            "secCode": "300975",
            "announcementId": f"id-{index}",
            "announcementTime": timestamp,
            "announcementTitle": f"<em>公告{index}</em>",
            "adjunctUrl": f"finalpage/{index}.PDF",
            "adjunctSize": "12.5",
        }
        for index in range(30)
    ]
    second_page = [dict(first_page[-1])]
    second_page.append(
        {
            "secCode": "300975",
            "announcementId": "id-30",
            "announcementTime": timestamp,
            "announcementTitle": "公告30",
            "adjunctUrl": "finalpage/30.PDF",
            "adjunctSize": "13.5",
        }
    )
    responses = iter([PageResponse(first_page), PageResponse(second_page)])
    calls: list[dict] = []
    client = CninfoClient(min_interval_s=0)
    monkeypatch.setattr(client, "org_map", lambda: {"300975": "gssz000300975"})

    def fake_post(_url: str, *, data: dict) -> PageResponse:
        calls.append(data)
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    try:
        rows = client.announcements("300975", date(2026, 7, 1), date(2026, 8, 26))
    finally:
        client.close()

    assert len(rows) == 31
    assert [call["pageNum"] for call in calls] == ["1", "2"]
    assert rows[0]["announce_time"] == datetime(2026, 8, 1, 8, 0)
    assert rows[0]["title"] == "公告0"


def test_daily_decisions_are_immutable_and_next_day_not_used(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    decision_date = date(2026, 8, 26)
    symbols = [f"{index:06d}.SZ" for index in range(1, 13)]
    for day_index in range(70):
        day = decision_date - timedelta(days=69 - day_index)
        rows = []
        for symbol_index, symbol in enumerate(symbols):
            close = 10 + symbol_index + day_index * (0.02 + symbol_index / 1000)
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "amount": 200_000_000.0 + day_index * 1_000_000,
                }
            )
        _write_partition(data_dir / "kline_daily_enriched", day, rows)
    (data_dir / "financials" / "metrics").mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": symbol,
                "period_end": "2026-06-30",
                "announce_date": "2026-08-20",
                "bps": 5.0,
                "roe": 15.0,
                "gross_margin": 35.0,
                "debt_to_asset_ratio": 30.0,
                "revenue_yoy": 30.0,
                "net_income_yoy": 25.0,
            }
            for symbol in symbols
        ]
    ).write_parquet(data_dir / "financials" / "metrics" / "part.parquet")

    store = PilotStore(tmp_path / "pilot")
    try:
        store.connection.executemany(
            "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                [
                    symbol,
                    symbol[:6],
                    f"样本{index}",
                    "semiconductor_frontend",
                    "半导体前道设备与材料",
                    "candidate",
                    index,
                    "mid",
                    5,
                    "光刻机",
                    5_000_000_000.0,
                    200_000_000.0,
                    decision_date,
                ]
                for index, symbol in enumerate(symbols, start=1)
            ],
        )
        first = freeze_daily_decisions(store, data_dir, decision_date)
        second = freeze_daily_decisions(store, data_dir, decision_date)
        selected = store.connection.execute(
            "SELECT count(*) FROM decisions WHERE model='serenity' AND selected"
        ).fetchone()[0]
    finally:
        store.close()

    assert first["status"] == "frozen"
    assert second["status"] == "already_frozen"
    assert selected == 10


def test_main_business_snapshot_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    class FakeClient:
        calls = 0

        def __init__(self, token: str) -> None:
            assert token == "configured"

        def main_business_records(self, symbol: str, *, kind: str):
            type(self).calls += 1
            return [
                {
                    "ts_code": symbol,
                    "end_date": "20260630",
                    "bz_item": "光刻设备",
                    "bz_sales": 100.0,
                    "bz_profit": 40.0,
                    "bz_cost": 60.0,
                    "curr_type": "CNY",
                    "update_flag": "1",
                }
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.serenity_pilot.get_api_key", lambda: "configured")
    monkeypatch.setattr("app.services.serenity_pilot.TushareClient", FakeClient)
    store = PilotStore(tmp_path / "pilot-main-business")
    try:
        store.connection.execute(
            "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "000001.SZ",
                "000001",
                "样本",
                "semiconductor_frontend",
                "半导体前道设备与材料",
                "candidate",
                1,
                "mid",
                5,
                "光刻机",
                5_000_000_000.0,
                200_000_000.0,
                date(2026, 8, 26),
            ],
        )

        first = collect_main_business(store)
        second = collect_main_business(store)
        rows = store.connection.execute("SELECT count(*) FROM main_business").fetchone()[0]
    finally:
        store.close()

    assert first["queried"] == 1
    assert first["inserted_rows"] == 1
    assert second["queried"] == 0
    assert rows == 1
    assert FakeClient.calls == 1
