from __future__ import annotations

from datetime import date, datetime

import pytest

from app.services.serenity_event_replay import EventReplayStore
from app.services.serenity_execution_scope import (
    EXECUTION_SCOPE_ID,
    evaluate_execution_scope,
    freeze_execution_scope,
)
from app.services.serenity_strategy_optimizer import OPTIMIZATION_ID
from app.services.serenity_thesis_iteration import initialize_thesis_tables


def _insert_universe(store: EventReplayStore, symbol: str, name: str, rank: int) -> None:
    store.connection.execute(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            symbol,
            symbol.split(".", 1)[0],
            name,
            "chain_0",
            "产业链0",
            "candidate",
            rank,
            "mid",
            1,
            "[]",
            1_000_000_000.0,
            100_000_000.0,
            date(2026, 8, 28),
        ],
    )


def _stock_basic(symbol: str, name: str, market: str, exchange: str) -> dict:
    return {
        "ts_code": symbol,
        "name": name,
        "market": market,
        "exchange": exchange,
        "list_status": "L",
        "list_date": "20200101",
    }


def test_freeze_execution_scope_separates_research_and_main_board(tmp_path) -> None:
    store = EventReplayStore(tmp_path / "event")
    try:
        _insert_universe(store, "600001.SH", "主板公司", 1)
        _insert_universe(store, "300001.SZ", "创业板公司", 2)

        result = freeze_execution_scope(
            store,
            [
                _stock_basic("600001.SH", "主板公司", "主板", "SSE"),
                _stock_basic("300001.SZ", "创业板公司", "创业板", "SZSE"),
            ],
            source_as_of=date(2026, 8, 28),
        )

        assert result == {
            "scope_id": EXECUTION_SCOPE_ID,
            "research_companies": 2,
            "execution_eligible_companies": 1,
            "board_counts": {"主板": 1, "创业板": 1},
            "capital_authority": False,
            "reused": False,
        }
        rows = store.connection.execute(
            """
            SELECT symbol, board, execution_eligible, eligibility_reason
            FROM serenity_security_execution_scope
            ORDER BY symbol
            """
        ).fetchall()
        assert rows == [
            ("300001.SZ", "创业板", False, "BOARD_NOT_AUTHORIZED"),
            ("600001.SH", "主板", True, "BOARD_AUTHORIZED_RESEARCH_ONLY"),
        ]
    finally:
        store.close()


def test_freeze_execution_scope_fails_closed_when_security_master_is_incomplete(
    tmp_path,
) -> None:
    store = EventReplayStore(tmp_path / "event")
    try:
        _insert_universe(store, "600001.SH", "主板公司", 1)
        _insert_universe(store, "300001.SZ", "创业板公司", 2)

        with pytest.raises(RuntimeError, match="missing 1 universe symbols"):
            freeze_execution_scope(
                store,
                [_stock_basic("600001.SH", "主板公司", "主板", "SSE")],
                source_as_of=date(2026, 8, 28),
            )

        assert (
            store.connection.execute(
                "SELECT count(*) FROM serenity_execution_scopes"
            ).fetchone()[0]
            == 0
        )
    finally:
        store.close()


def test_execution_evaluation_excludes_restricted_board_hard_gate(tmp_path) -> None:
    store = EventReplayStore(tmp_path / "event")
    try:
        _insert_universe(store, "600001.SH", "主板公司", 1)
        _insert_universe(store, "300001.SZ", "创业板公司", 2)
        freeze_execution_scope(
            store,
            [
                _stock_basic("600001.SH", "主板公司", "主板", "SSE"),
                _stock_basic("300001.SZ", "创业板公司", "创业板", "SZSE"),
            ],
            source_as_of=date(2026, 8, 28),
        )
        initialize_thesis_tables(store)

        for event_id, symbol, decision_day in (
            ("event-main", "600001.SH", date(2026, 6, 1)),
            ("event-gem", "300001.SZ", date(2026, 6, 10)),
        ):
            store.connection.execute(
                """
                INSERT INTO serenity_thesis_consensus_scores VALUES
                (?, ?, ?, ?, 'PASS', 45.0, '[]', true, 'test', ?)
                """,
                [OPTIMIZATION_ID, event_id, symbol, decision_day, datetime.now()],
            )
            store.connection.execute(
                """
                INSERT INTO event_discovery_outcomes VALUES
                (?, ?, 'ORDER_CONTRACT', 5, ?, ?, ?, 0.03, 0.028, 0.01,
                 0.015, -0.01, 0.04, 'SETTLED', ?)
                """,
                [
                    event_id,
                    symbol,
                    decision_day,
                    date(2026, 6, 2),
                    date(2026, 6, 9),
                    datetime.now(),
                ],
            )

        report = evaluate_execution_scope(store)

        assert report["research_hard_gate_event_count"] == 2
        assert report["execution_hard_gate_event_count"] == 1
        assert report["excluded_hard_gate_event_count"] == 1
        all_horizon_five = next(
            row
            for row in report["execution_results"]
            if row["fold_id"] == "ALL" and row["horizon"] == 5
        )
        assert all_horizon_five["observation_count"] == 1
        assert all_horizon_five["mean_net_return"] == pytest.approx(0.028)
        assert report["claim_boundary"] == (
            "MAIN_BOARD_RESEARCH_EXECUTION_SCOPE_NO_CAPITAL_AUTHORITY"
        )
    finally:
        store.close()
