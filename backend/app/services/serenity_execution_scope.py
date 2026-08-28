"""Freeze and evaluate the owner's executable board scope for Serenity research.

The full A-share universe remains available for causal supply-chain research.  This
module separately freezes which securities the owner's current account may use in
execution-oriented replay.  It never grants capital authority and never converts a
restricted-board research event into an executable signal.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.market_time import CN_TZ
from app.plugins.tushare.client import TushareClient
from app.plugins.tushare.provider import get_api_key
from app.services.serenity_event_replay import EventReplayStore
from app.services.serenity_pilot import _atomic_json, _pilot_lock
from app.services.serenity_strategy_optimizer import FOLDS, OPTIMIZATION_ID, _trial_summary
from app.services.serenity_thesis_iteration import (
    _non_overlapping,
    initialize_thesis_tables,
)

EXECUTION_SCOPE_ID = "OWNER_MAIN_BOARD_ONLY_20260828"
ALLOWED_BOARDS = ("主板",)
CAPITAL_AUTHORITY = False
STOCK_BASIC_FIELDS = (
    "ts_code",
    "name",
    "market",
    "exchange",
    "list_status",
    "list_date",
)


def initialize_execution_scope_tables(store: EventReplayStore) -> None:
    initialize_thesis_tables(store)
    store.connection.execute(
        """
        CREATE TABLE IF NOT EXISTS serenity_execution_scopes (
            scope_id VARCHAR PRIMARY KEY,
            allowed_boards_json VARCHAR NOT NULL,
            capital_authority BOOLEAN NOT NULL,
            source VARCHAR NOT NULL,
            source_as_of DATE NOT NULL,
            claim_boundary VARCHAR NOT NULL,
            frozen_at TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS serenity_security_execution_scope (
            scope_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            exchange VARCHAR NOT NULL,
            board VARCHAR NOT NULL,
            list_status VARCHAR NOT NULL,
            listing_date DATE,
            execution_eligible BOOLEAN NOT NULL,
            eligibility_reason VARCHAR NOT NULL,
            source_as_of DATE NOT NULL,
            frozen_at TIMESTAMP NOT NULL,
            PRIMARY KEY (scope_id, symbol)
        );
        CREATE TABLE IF NOT EXISTS serenity_execution_scope_trials (
            scope_id VARCHAR NOT NULL,
            optimization_id VARCHAR NOT NULL,
            fold_id VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            observation_count INTEGER NOT NULL,
            mean_net_return DOUBLE,
            median_net_return DOUBLE,
            win_rate DOUBLE,
            alpha_csi300 DOUBLE,
            alpha_chain DOUBLE,
            mean_mae DOUBLE,
            mean_mfe DOUBLE,
            status VARCHAR NOT NULL,
            evaluated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (scope_id, optimization_id, fold_id, horizon)
        );
        """
    )


def _parse_listing_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError as exc:
        raise RuntimeError(f"invalid Tushare listing date: {raw}") from exc


def _scope_summary(store: EventReplayStore, *, reused: bool) -> dict[str, Any]:
    rows = store.connection.execute(
        """
        SELECT board, execution_eligible
        FROM serenity_security_execution_scope
        WHERE scope_id=?
        """,
        [EXECUTION_SCOPE_ID],
    ).fetchall()
    board_counts = Counter(str(row[0]) for row in rows)
    return {
        "scope_id": EXECUTION_SCOPE_ID,
        "research_companies": len(rows),
        "execution_eligible_companies": sum(bool(row[1]) for row in rows),
        "board_counts": dict(sorted(board_counts.items())),
        "capital_authority": CAPITAL_AUTHORITY,
        "reused": reused,
    }


def freeze_execution_scope(
    store: EventReplayStore,
    stock_basic_rows: list[dict[str, Any]],
    *,
    source_as_of: date,
) -> dict[str, Any]:
    """Freeze a complete current security-master slice for the research universe."""
    initialize_execution_scope_tables(store)
    existing = store.connection.execute(
        "SELECT count(*) FROM serenity_execution_scopes WHERE scope_id=?",
        [EXECUTION_SCOPE_ID],
    ).fetchone()[0]
    if existing:
        expected = store.connection.execute("SELECT count(*) FROM universe").fetchone()[0]
        actual = store.connection.execute(
            """
            SELECT count(*) FROM serenity_security_execution_scope
            WHERE scope_id=?
            """,
            [EXECUTION_SCOPE_ID],
        ).fetchone()[0]
        if actual != expected:
            raise RuntimeError(
                f"frozen execution scope is incomplete: expected {expected}, got {actual}"
            )
        return _scope_summary(store, reused=True)

    universe = {str(row["symbol"]): row for row in store.universe()}
    source: dict[str, dict[str, Any]] = {}
    for raw in stock_basic_rows:
        symbol = str(raw.get("ts_code") or "").strip()
        if not symbol or symbol not in universe:
            continue
        if symbol in source:
            raise RuntimeError(f"duplicate security-master symbol: {symbol}")
        source[symbol] = raw
    missing = sorted(set(universe) - set(source))
    if missing:
        raise RuntimeError(f"security master missing {len(missing)} universe symbols")

    frozen_at = datetime.now(CN_TZ).replace(tzinfo=None)
    values: list[list[Any]] = []
    for symbol in sorted(universe):
        raw = source[symbol]
        board = str(raw.get("market") or "").strip()
        exchange = str(raw.get("exchange") or "").strip()
        list_status = str(raw.get("list_status") or "").strip().upper()
        if not board or not exchange or list_status != "L":
            raise RuntimeError(f"security master is not execution-qualified: {symbol}")
        eligible = board in ALLOWED_BOARDS
        values.append(
            [
                EXECUTION_SCOPE_ID,
                symbol,
                str(raw.get("name") or universe[symbol]["name"]),
                exchange,
                board,
                list_status,
                _parse_listing_date(raw.get("list_date")),
                eligible,
                "BOARD_AUTHORIZED_RESEARCH_ONLY" if eligible else "BOARD_NOT_AUTHORIZED",
                source_as_of,
                frozen_at,
            ]
        )

    store.connection.execute("BEGIN TRANSACTION")
    try:
        store.connection.execute(
            """
            INSERT INTO serenity_execution_scopes VALUES
            (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                EXECUTION_SCOPE_ID,
                json.dumps(ALLOWED_BOARDS, ensure_ascii=False),
                CAPITAL_AUTHORITY,
                "TUSHARE_STOCK_BASIC_CURRENT_SNAPSHOT",
                source_as_of,
                "MAIN_BOARD_RESEARCH_EXECUTION_SCOPE_NO_CAPITAL_AUTHORITY",
                frozen_at,
            ],
        )
        store.connection.executemany(
            """
            INSERT INTO serenity_security_execution_scope VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        store.connection.execute("COMMIT")
    except Exception:
        store.connection.execute("ROLLBACK")
        raise
    return _scope_summary(store, reused=False)


def collect_execution_scope(store: EventReplayStore) -> dict[str, Any]:
    """Fetch Tushare's current board labels once and freeze the complete slice."""
    initialize_execution_scope_tables(store)
    existing = store.connection.execute(
        "SELECT 1 FROM serenity_execution_scopes WHERE scope_id=?",
        [EXECUTION_SCOPE_ID],
    ).fetchone()
    if existing:
        return _scope_summary(store, reused=True)
    client = TushareClient(get_api_key())
    try:
        rows = client.query("stock_basic", {"list_status": "L"}, STOCK_BASIC_FIELDS)
    finally:
        client.close()
    return freeze_execution_scope(
        store,
        rows,
        source_as_of=datetime.now(CN_TZ).date(),
    )


def _funnel_row(
    store: EventReplayStore,
    label: str,
    sql: str,
) -> dict[str, Any]:
    rows = store.connection.execute(sql, [EXECUTION_SCOPE_ID]).fetchall()
    boards = Counter(str(row[1]) for row in rows)
    return {
        "stage": label,
        "records": len(rows),
        "companies": len({str(row[0]) for row in rows}),
        "board_counts": dict(sorted(boards.items())),
        "execution_eligible_records": sum(bool(row[2]) for row in rows),
    }


def board_funnel(store: EventReplayStore) -> list[dict[str, Any]]:
    initialize_execution_scope_tables(store)
    queries = (
        (
            "FROZEN_UNIVERSE",
            """
            SELECT u.symbol, s.board, s.execution_eligible
            FROM universe u JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=?
            """,
        ),
        (
            "SETTLED_DISCOVERY_EVENTS",
            """
            SELECT DISTINCT o.symbol, s.board, s.execution_eligible, o.event_id
            FROM event_discovery_outcomes o
            JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=? AND o.status='SETTLED'
            """,
        ),
        (
            "ROUND1_EVENT_GATE_PASS",
            """
            SELECT e.symbol, s.board, s.execution_eligible, e.event_id
            FROM serenity_event_semantic_scores e
            JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=? AND e.stage='SERENITY_EVENT_SCORE'
              AND e.event_gate='PASS' AND e.newness='NEW_INFORMATION'
              AND e.economic_bridge='PASS' AND e.event_stage!='ROUTINE_ADMIN'
            """,
        ),
        (
            "ROUND2_ENRICHED_SCORE",
            """
            SELECT e.symbol, s.board, s.execution_eligible, e.event_id
            FROM serenity_event_semantic_scores e
            JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=? AND e.stage='SERENITY_EVENT_ENRICHED_SCORE'
            """,
        ),
        (
            "ROUND4_EVIDENCE_REPAIR",
            """
            SELECT e.symbol, s.board, s.execution_eligible, e.event_id
            FROM serenity_thesis_scores e
            JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=? AND e.substage='THESIS_EVIDENCE_REPAIR'
            """,
        ),
        (
            "FINAL_HARD_GATE",
            """
            SELECT e.symbol, s.board, s.execution_eligible, e.event_id
            FROM serenity_thesis_consensus_scores e
            JOIN serenity_security_execution_scope s USING(symbol)
            WHERE s.scope_id=? AND e.hard_gate_pass=true
            """,
        ),
    )
    return [_funnel_row(store, label, sql) for label, sql in queries]


def evaluate_execution_scope(store: EventReplayStore) -> dict[str, Any]:
    """Recalculate hard-gate outcomes using only the frozen executable board scope."""
    initialize_execution_scope_tables(store)
    scope = store.connection.execute(
        "SELECT source_as_of, claim_boundary FROM serenity_execution_scopes WHERE scope_id=?",
        [EXECUTION_SCOPE_ID],
    ).fetchone()
    if scope is None:
        raise RuntimeError("execution scope is not frozen")
    expected = store.connection.execute("SELECT count(*) FROM universe").fetchone()[0]
    covered = store.connection.execute(
        "SELECT count(*) FROM serenity_security_execution_scope WHERE scope_id=?",
        [EXECUTION_SCOPE_ID],
    ).fetchone()[0]
    if covered != expected:
        raise RuntimeError(
            f"execution scope coverage mismatch: expected {expected}, got {covered}"
        )

    hard_event_rows = store.connection.execute(
        """
        SELECT c.event_id, c.symbol, u.name, s.board, s.execution_eligible,
               s.eligibility_reason
        FROM serenity_thesis_consensus_scores c
        JOIN serenity_security_execution_scope s ON s.symbol=c.symbol
        JOIN universe u ON u.symbol=c.symbol
        WHERE c.optimization_id=? AND c.hard_gate_pass=true AND s.scope_id=?
        ORDER BY c.decision_date, c.event_id
        """,
        [OPTIMIZATION_ID, EXECUTION_SCOPE_ID],
    ).fetchall()
    rows = store.connection.execute(
        """
        SELECT c.event_id, c.symbol, c.decision_date, o.entry_date, o.exit_date,
               o.horizon, o.net_return, o.benchmark_return, o.chain_return,
               o.mae, o.mfe, s.board, s.execution_eligible, s.eligibility_reason,
               u.name
        FROM serenity_thesis_consensus_scores c
        JOIN event_discovery_outcomes o ON o.event_id=c.event_id
        JOIN serenity_security_execution_scope s ON s.symbol=c.symbol
        JOIN universe u ON u.symbol=c.symbol
        WHERE c.optimization_id=? AND c.hard_gate_pass=true
          AND s.scope_id=? AND o.status='SETTLED'
        ORDER BY c.decision_date, c.event_id, o.horizon
        """,
        [OPTIMIZATION_ID, EXECUTION_SCOPE_ID],
    ).fetchall()
    records = [
        {
            "event_id": str(row[0]),
            "symbol": str(row[1]),
            "decision_date": row[2],
            "entry_date": row[3],
            "exit_date": row[4],
            "horizon": int(row[5]),
            "net_return": float(row[6]),
            "benchmark_return": float(row[7]),
            "chain_return": float(row[8]),
            "mae": float(row[9]),
            "mfe": float(row[10]),
            "board": str(row[11]),
            "execution_eligible": bool(row[12]),
            "eligibility_reason": str(row[13]),
            "name": str(row[14]),
        }
        for row in rows
    ]
    research_event_ids = {str(row[0]) for row in hard_event_rows}
    executable = [row for row in records if row["execution_eligible"]]
    execution_event_ids = {str(row[0]) for row in hard_event_rows if bool(row[4])}
    excluded = sorted(
        {
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[5]),
            )
            for row in hard_event_rows
            if not bool(row[4])
        }
    )

    now = datetime.now(CN_TZ).replace(tzinfo=None)
    results: list[dict[str, Any]] = []
    store.connection.execute(
        "DELETE FROM serenity_execution_scope_trials WHERE scope_id=? AND optimization_id=?",
        [EXECUTION_SCOPE_ID, OPTIMIZATION_ID],
    )
    for fold_id, start, end in (
        *FOLDS,
        ("ALL", date(2025, 8, 26), date(2026, 8, 27)),
    ):
        for horizon in (2, 3, 5, 10):
            selected = _non_overlapping(
                [
                    row
                    for row in executable
                    if row["horizon"] == horizon
                    and start <= row["decision_date"] <= end
                ]
            )
            summary = _trial_summary(selected)
            status = "OBSERVED" if selected else "NO_OBSERVATIONS"
            result = {
                "fold_id": fold_id,
                "horizon": horizon,
                **summary,
                "status": status,
            }
            results.append(result)
            store.connection.execute(
                """
                INSERT INTO serenity_execution_scope_trials VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    EXECUTION_SCOPE_ID,
                    OPTIMIZATION_ID,
                    fold_id,
                    horizon,
                    summary["observation_count"],
                    summary["mean_net_return"],
                    summary["median_net_return"],
                    summary["win_rate"],
                    summary["alpha_csi300"],
                    summary["alpha_chain"],
                    summary["mean_mae"],
                    summary["mean_mfe"],
                    status,
                    now,
                ],
            )

    report = {
        "scope_id": EXECUTION_SCOPE_ID,
        "optimization_id": OPTIMIZATION_ID,
        "source_as_of": scope[0].isoformat(),
        "allowed_boards": list(ALLOWED_BOARDS),
        "capital_authority": CAPITAL_AUTHORITY,
        "claim_boundary": str(scope[1]),
        "research_hard_gate_event_count": len(research_event_ids),
        "execution_hard_gate_event_count": len(execution_event_ids),
        "excluded_hard_gate_event_count": len(research_event_ids - execution_event_ids),
        "excluded_hard_gate_events": [
            {
                "event_id": row[0],
                "symbol": row[1],
                "name": row[2],
                "board": row[3],
                "reason": row[4],
            }
            for row in excluded
        ],
        "board_funnel": board_funnel(store),
        "execution_results": results,
    }
    output = store.root / "optimization" / OPTIMIZATION_ID
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "round05-main-board-execution-report.json", report)
    return report


def status(store: EventReplayStore) -> dict[str, Any]:
    initialize_execution_scope_tables(store)
    scope = store.connection.execute(
        """
        SELECT source_as_of, claim_boundary
        FROM serenity_execution_scopes WHERE scope_id=?
        """,
        [EXECUTION_SCOPE_ID],
    ).fetchone()
    if scope is None:
        return {"scope_id": EXECUTION_SCOPE_ID, "status": "NOT_FROZEN"}
    return {
        **_scope_summary(store, reused=True),
        "status": "FROZEN",
        "source_as_of": scope[0].isoformat(),
        "claim_boundary": str(scope[1]),
        "board_funnel": board_funnel(store),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate", "status"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with _pilot_lock(args.root):
        store = EventReplayStore(args.root)
        try:
            if args.command == "freeze":
                payload = collect_execution_scope(store)
            elif args.command == "evaluate":
                payload = evaluate_execution_scope(store)
            else:
                payload = status(store)
        finally:
            with contextlib.suppress(Exception):
                store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
