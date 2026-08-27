from __future__ import annotations

from datetime import date, datetime, timedelta

from app.services.serenity_event_replay import EventReplayStore
from app.services.serenity_strategy_optimizer import (
    DIMENSION_WEIGHTS,
    OPTIMIZATION_ID,
    _dimension_score,
    _event_score_schema,
    _mask_identity,
    _select_support_pages,
    _support_kind,
    classify_capacity_subtype,
    materialize_event_features,
)


def test_support_selection_rejects_routine_documents_and_ranks_evidence_pages() -> None:
    assert _support_kind("某公司2025年年度报告") == "LATEST_PERIODIC_REPORT"
    assert _support_kind("投资者关系活动记录表") == "LATEST_INVESTOR_ACTIVITY"
    assert _support_kind("关于客户认证并取得订单的公告") == "LATEST_PRIOR_OPERATING_DISCLOSURE"
    assert _support_kind("投资者关系管理制度") is None
    assert _support_kind("2025年年度报告摘要") is None

    pages = {
        1: "目录和普通介绍",
        2: "客户认证周期为18个月,现有产能为300吨。",
        3: "供应商数量为2家,关键设备依赖进口。",
    }
    assert list(_select_support_pages(pages)) == [2, 3]


def test_capacity_subtype_rejects_financing_admin_false_positive() -> None:
    assert (
        classify_capacity_subtype("关于募投项目结项并将节余募集资金永久补充流动资金的公告")
        == "FINANCING_ADMIN"
    )
    assert classify_capacity_subtype("关于年产300吨项目建成投产的公告") == "OPERATING_MILESTONE"
    assert classify_capacity_subtype("关于拟购买土地并投资建设项目的公告") == "IMPLEMENTATION_PROGRESS"


def test_masked_model_packet_does_not_keep_identity_or_absolute_date() -> None:
    masked = _mask_identity(
        "证券代码301568,厦门思泰克智能科技股份有限公司于2026年8月27日投产",
        symbol="301568.SZ",
        code="301568",
        name="厦门思泰克智能科技股份有限公司",
    )
    assert "301568" not in masked
    assert "思泰克" not in masked
    assert "2026年8月27日" not in masked
    assert "[发行人]" in masked
    assert "[历史日期]" in masked


def test_event_schema_and_dimension_bounds_keep_unknown_out_of_exact_score() -> None:
    output = {
        "schema_version": "1.0.0",
        "entity_id": "EVENT-12345678",
        "cutoff": "T0",
        "event_review": {
            "event_gate": "DATA_INSUFFICIENT",
            "event_stage": "OTHER",
            "newness": "UNKNOWN",
            "economic_bridge": "DATA_INSUFFICIENT",
            "repricing_horizon": "UNKNOWN",
            "reason": "原文不足",
            "evidence": [],
        },
        "dimensions": [
            {
                "dimension_id": dimension_id,
                "status": "UNKNOWN",
                "rating": None,
                "reason": "原文未覆盖",
                "evidence": [],
            }
            for dimension_id in DIMENSION_WEIGHTS
        ],
        "penalties": [
            {
                "penalty_id": penalty_id,
                "status": "UNKNOWN",
                "rating": None,
                "reason": "原文未覆盖",
                "evidence": [],
            }
            for penalty_id in (
                "dilution_financing",
                "governance",
                "geopolitics",
                "liquidity",
                "hype_risk",
                "accounting_quality",
                "cyclicality",
                "alternative_design_risk",
            )
        ],
        "kill_switches": [],
    }
    schema = _event_score_schema()
    assert "event_review" in schema["required"]
    assert schema["properties"]["event_review"]["additionalProperties"] is False
    assert set(output) == set(schema["required"])
    score = _dimension_score(output)
    assert score == {
        "known_weight": 0.0,
        "known_points": 0.0,
        "lower_bound": 0.0,
        "upper_bound": 64.0,
        "complete": None,
    }


def test_event_features_are_point_in_time_and_materialized_once(tmp_path) -> None:
    store = EventReplayStore(tmp_path / "event")
    try:
        store.connection.execute(
            "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "000001.SZ",
                "000001",
                "样本公司",
                "chain_0",
                "产业链0",
                "candidate",
                1,
                "mid",
                1,
                "[]",
                1_000_000_000.0,
                100_000_000.0,
                date(2026, 8, 26),
            ],
        )
        first_day = date(2026, 7, 1)
        for index in range(23):
            day = first_day + timedelta(days=index)
            for symbol, base, kind in (
                ("000001.SZ", 10.0, "stock"),
                ("000300.SH", 100.0, "index"),
            ):
                close = base + index
                store.connection.execute(
                    "INSERT INTO research_daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        symbol,
                        day,
                        close,
                        close + 1,
                        close - 1,
                        close,
                        close - 1,
                        1_000.0,
                        100_000.0,
                        1.0,
                        kind,
                        datetime(2026, 8, 28, 12, 0),
                    ],
                )
        store.connection.execute(
            "INSERT INTO announcements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "event-1",
                "000001.SZ",
                datetime(2026, 7, 22, 18, 0),
                "关于募投项目结项并补充流动资金的公告",
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
                datetime(2026, 7, 22, 18, 0),
                "关于募投项目结项并补充流动资金的公告",
                "CAPACITY_MILESTONE",
                '["CAPACITY_MILESTONE"]',
                "LONG_CANDIDATE",
                "metadata-hash",
                first_day + timedelta(days=21),
                first_day + timedelta(days=22),
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
                0,
                "ok",
                datetime(2026, 8, 28, 12, 0),
            ],
        )
        for horizon in (2, 3, 5, 10):
            store.connection.execute(
                "INSERT INTO event_discovery_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "event-1",
                    "000001.SZ",
                    "CAPACITY_MILESTONE",
                    horizon,
                    first_day + timedelta(days=21),
                    first_day + timedelta(days=22),
                    first_day + timedelta(days=22),
                    0.01,
                    0.008,
                    0.001,
                    0.002,
                    -0.01,
                    0.02,
                    "SETTLED",
                    datetime(2026, 8, 28, 12, 0),
                ],
            )

        assert materialize_event_features(store) == {"materialized": 1, "blocked": 0}
        row = store.connection.execute(
            """
            SELECT deterministic_subtype, feature_json
            FROM serenity_event_features WHERE optimization_id=?
            """,
            [OPTIMIZATION_ID],
        ).fetchone()
        assert row[0] == "FINANCING_ADMIN"
        assert '"point_in_time": true' in row[1]
    finally:
        store.close()
