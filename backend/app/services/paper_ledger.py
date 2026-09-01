"""Transactional event ledger and read models for paper trading."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.market_time import CN_TZ, cn_now
from app.trading_rules import TradingCostModel

SCHEMA_VERSION = 6
TERMINAL_ORDER_STATUSES = frozenset({
    "FILLED",
    "PARTIALLY_FILLED",
    "REJECTED_LIMIT_UP",
    "REJECTED_LIMIT_DOWN",
    "REJECTED_SUSPENDED",
    "REJECTED_INSUFFICIENT_CASH",
    "REJECTED_UNSUPPORTED_BOARD",
    "UNKNOWN_MARKET_DATA",
    "EXECUTION_FAILED",
    "MISSED_EXECUTION",
    "CANCELLED",
})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _now_text(value: datetime | None = None) -> str:
    current = value or cn_now()
    current = current.replace(tzinfo=CN_TZ) if current.tzinfo is None else current.astimezone(CN_TZ)
    return current.isoformat(timespec="seconds")


def stable_id(*parts: object) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, "paper://" + "/".join(map(str, parts))).hex


class PaperLedgerError(RuntimeError):
    pass


class PaperLedger:
    """SQLite WAL ledger; event rows are append-only and projections update atomically."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "paper_trading"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "paper_ledger.sqlite3"
        self._schema_lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._schema_lock, self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','paused','deleted')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    baseline_date TEXT NOT NULL,
                    signal_start_date TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    initial_capital REAL NOT NULL CHECK(initial_capital >= 0),
                    cash_balance REAL NOT NULL CHECK(cash_balance >= -0.01),
                    last_signal_date TEXT,
                    last_settlement_date TEXT,
                    last_error TEXT,
                    executor_state TEXT NOT NULL DEFAULT 'READY'
                );
                CREATE TABLE IF NOT EXISTS signal_intents (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                    signal_date TEXT NOT NULL,
                    score REAL,
                    reason TEXT NOT NULL,
                    signal_ref TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    frozen_at TEXT NOT NULL,
                    UNIQUE(account_id, symbol, side, signal_date, reason)
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    signal_id TEXT NOT NULL REFERENCES signal_intents(id),
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                    requested_qty INTEGER NOT NULL CHECK(requested_qty >= 0),
                    filled_qty INTEGER NOT NULL DEFAULT 0 CHECK(filled_qty >= 0),
                    target_amount REAL NOT NULL DEFAULT 0,
                    target_weight REAL NOT NULL DEFAULT 0,
                    planned_session TEXT NOT NULL,
                    scheduled_date TEXT,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    execution_quality TEXT,
                    preflight_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    terminal_at TEXT,
                    UNIQUE(account_id, signal_id, planned_session)
                );
                CREATE INDEX IF NOT EXISTS idx_orders_due
                    ON orders(status, scheduled_date, planned_session);
                CREATE TABLE IF NOT EXISTS fills (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                    quantity INTEGER NOT NULL CHECK(quantity > 0),
                    price REAL NOT NULL CHECK(price > 0),
                    gross_amount REAL NOT NULL,
                    fee_amount REAL NOT NULL,
                    cash_delta REAL NOT NULL,
                    executed_at TEXT NOT NULL,
                    quote_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    reference_price REAL,
                    day_pnl REAL NOT NULL DEFAULT 0,
                    UNIQUE(order_id, id)
                );
                CREATE TABLE IF NOT EXISTS positions (
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    quantity INTEGER NOT NULL CHECK(quantity >= 0),
                    available_qty INTEGER NOT NULL CHECK(available_qty >= 0),
                    locked_qty INTEGER NOT NULL CHECK(locked_qty >= 0),
                    average_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    acquired_on TEXT NOT NULL,
                    hold_days INTEGER NOT NULL DEFAULT 0,
                    max_price REAL NOT NULL DEFAULT 0,
                    last_price REAL,
                    market_value REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0,
                    previous_close REAL,
                    day_pnl REAL NOT NULL DEFAULT 0,
                    pnl_date TEXT,
                    today_bought_qty INTEGER NOT NULL DEFAULT 0,
                    today_bought_cost REAL NOT NULL DEFAULT 0,
                    quote_at TEXT,
                    quote_source TEXT,
                    pending_exit_reason TEXT,
                    pending_exit_date TEXT,
                    PRIMARY KEY(account_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS cash_entries (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    event_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    reference_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    UNIQUE(account_id, event_type, reference_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_key TEXT NOT NULL UNIQUE,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    trading_date TEXT,
                    entity_type TEXT,
                    entity_id TEXT,
                    severity TEXT NOT NULL DEFAULT 'info',
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_account_time
                    ON events(account_id, occurred_at DESC);
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    incident_key TEXT NOT NULL UNIQUE,
                    code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','resolved')),
                    opened_at TEXT NOT NULL,
                    resolved_at TEXT,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT
                );
                CREATE TABLE IF NOT EXISTS nav_snapshots (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(id),
                    trading_date TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL,
                    UNIQUE(account_id, trading_date, source)
                );
                CREATE TABLE IF NOT EXISTS legacy_snapshots (
                    account_id TEXT PRIMARY KEY REFERENCES accounts(id),
                    source_path TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS phase_runs (
                    id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    trading_date TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    UNIQUE(phase, trading_date)
                );
                """
            )
            position_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(positions)").fetchall()
            }
            if "pending_exit_date" not in position_columns:
                conn.execute("ALTER TABLE positions ADD COLUMN pending_exit_date TEXT")
            position_additions = {
                "previous_close": "REAL",
                "day_pnl": "REAL NOT NULL DEFAULT 0",
                "pnl_date": "TEXT",
                "today_bought_qty": "INTEGER NOT NULL DEFAULT 0",
                "today_bought_cost": "REAL NOT NULL DEFAULT 0",
            }
            for column, declaration in position_additions.items():
                if column not in position_columns:
                    conn.execute(f"ALTER TABLE positions ADD COLUMN {column} {declaration}")
            fill_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(fills)").fetchall()
            }
            if "reference_price" not in fill_columns:
                conn.execute("ALTER TABLE fills ADD COLUMN reference_price REAL")
            if "day_pnl" not in fill_columns:
                conn.execute("ALTER TABLE fills ADD COLUMN day_pnl REAL NOT NULL DEFAULT 0")
            conn.execute(
                """UPDATE incidents SET severity='critical'
                    WHERE status='open'
                    AND code IN ('MISSED_EXECUTION','UNKNOWN_MARKET_DATA','EXECUTION_FAILED')"""
            )
            conn.execute(
                """UPDATE orders SET execution_quality='NO_RELIABLE_OPEN_DATA'
                    WHERE status='UNKNOWN_MARKET_DATA'
                    AND execution_quality='RECOVERED_LATE'
                    AND NOT EXISTS (SELECT 1 FROM fills WHERE fills.order_id=orders.id)"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        *,
        event_key: str,
        account_id: str,
        event_type: str,
        title: str,
        detail: str = "",
        occurred_at: str,
        trading_date: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO events(
                id,event_key,account_id,event_type,occurred_at,trading_date,
                entity_type,entity_id,severity,title,detail,payload_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                stable_id("event", event_key), event_key, account_id, event_type,
                occurred_at, trading_date, entity_type, entity_id, severity,
                title, detail, _json(payload or {}),
            ),
        )

    def create_account(
        self,
        *,
        name: str,
        baseline_date: date,
        config: dict[str, Any],
        account_id: str | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("模拟账户名称不能为空")
        now = _now_text(created_at)
        identifier = account_id or uuid.uuid4().hex[:12]
        capital = float(config["initial_capital"])
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO accounts(
                    id,name,status,created_at,updated_at,baseline_date,signal_start_date,
                    config_json,initial_capital,cash_balance
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier, clean_name[:80], "active", now, now,
                    baseline_date.isoformat(), baseline_date.isoformat(), _json(config),
                    capital, capital,
                ),
            )
            cash_id = stable_id(identifier, "INITIAL_CAPITAL")
            conn.execute(
                """INSERT INTO cash_entries(
                    id,account_id,event_type,amount,balance_after,reference_id,occurred_at,detail
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (cash_id, identifier, "INITIAL_CAPITAL", capital, capital, identifier, now, "初始资金入账"),
            )
            self._event(
                conn,
                event_key=f"{identifier}:ACCOUNT_CREATED",
                account_id=identifier,
                event_type="ACCOUNT_CREATED",
                occurred_at=now,
                trading_date=baseline_date.isoformat(),
                entity_type="account",
                entity_id=identifier,
                title="模拟账户已创建",
                detail="账户配置已冻结, 后续只处理创建后的真实时钟事件",
            )
        return self.get_account(identifier)

    def account_row(self, account_id: str, *, include_deleted: bool = False) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
        if row is None or (row["status"] == "deleted" and not include_deleted):
            raise KeyError(account_id)
        return row

    def list_account_rows(self, *, active_only: bool = False) -> list[sqlite3.Row]:
        where = "status='active'" if active_only else "status!='deleted'"
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM accounts WHERE {where} ORDER BY created_at DESC"
            ).fetchall()

    def set_status(self, account_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("不支持的模拟账户状态")
        now = _now_text()
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT status FROM accounts WHERE id=? AND status!='deleted'", (account_id,)
            ).fetchone()
            if current is None:
                raise KeyError(account_id)
            conn.execute(
                "UPDATE accounts SET status=?,updated_at=? WHERE id=?", (status, now, account_id)
            )
            self._event(
                conn,
                event_key=f"{account_id}:STATUS:{status}:{now}",
                account_id=account_id,
                event_type="ACCOUNT_STATUS_CHANGED",
                occurred_at=now,
                entity_type="account",
                entity_id=account_id,
                title="账户已恢复" if status == "active" else "账户已暂停",
                detail="暂停只停止产生新信号和新订单, 不改写已有账本",
            )
        return self.get_account(account_id)

    def increase_capital(
        self,
        account_id: str,
        amount: float,
        *,
        reference_id: str,
        detail: str = "模拟资金追加",
    ) -> dict[str, Any]:
        """Increase an account budget with an idempotent cash-ledger entry."""
        contribution = float(amount)
        if contribution <= 0:
            raise ValueError("追加资金必须大于 0")
        reference = str(reference_id).strip()
        if not reference:
            raise ValueError("追加资金必须提供唯一业务参考号")

        now = _now_text()
        already_applied = False
        with self.transaction() as conn:
            account = conn.execute(
                "SELECT * FROM accounts WHERE id=? AND status!='deleted'", (account_id,)
            ).fetchone()
            if account is None:
                raise KeyError(account_id)
            existing = conn.execute(
                """SELECT amount FROM cash_entries
                    WHERE account_id=? AND event_type='CAPITAL_CONTRIBUTION' AND reference_id=?""",
                (account_id, reference),
            ).fetchone()
            if existing is not None:
                if abs(float(existing["amount"]) - contribution) >= 0.01:
                    raise ValueError("同一资金参考号的金额不一致")
                already_applied = True
            if not already_applied:
                new_initial = float(account["initial_capital"]) + contribution
                new_cash = float(account["cash_balance"]) + contribution
                config = _loads(account["config_json"], {})
                config["initial_capital"] = new_initial
                conn.execute(
                    """UPDATE accounts SET initial_capital=?,cash_balance=?,config_json=?,updated_at=?
                        WHERE id=?""",
                    (new_initial, new_cash, _json(config), now, account_id),
                )
                conn.execute(
                    """INSERT INTO cash_entries(
                        id,account_id,event_type,amount,balance_after,reference_id,occurred_at,detail
                    ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        stable_id(account_id, "CAPITAL_CONTRIBUTION", reference),
                        account_id,
                        "CAPITAL_CONTRIBUTION",
                        contribution,
                        new_cash,
                        reference,
                        now,
                        detail,
                    ),
                )
                self._event(
                    conn,
                    event_key=f"{account_id}:CAPITAL_CONTRIBUTION:{reference}",
                    account_id=account_id,
                    event_type="CAPITAL_CONTRIBUTION",
                    occurred_at=now,
                    entity_type="account",
                    entity_id=account_id,
                    title="模拟资金已追加",
                    detail=f"追加 {contribution:.2f} 元, 资金基准调整为 {new_initial:.2f} 元",
                    payload={"amount": contribution, "capital_budget": new_initial},
                )
        return self.get_account(account_id)

    def delete_account(self, account_id: str) -> dict[str, Any]:
        now = _now_text()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT name FROM accounts WHERE id=? AND status!='deleted'", (account_id,)
            ).fetchone()
            if row is None:
                raise KeyError(account_id)
            self._event(
                conn,
                event_key=f"{account_id}:ACCOUNT_DELETED",
                account_id=account_id,
                event_type="ACCOUNT_DELETED",
                occurred_at=now,
                entity_type="account",
                entity_id=account_id,
                severity="warning",
                title="账户已删除",
                detail="账户已从运营界面移除, 审计账本保留",
            )
            conn.execute(
                "UPDATE accounts SET status='deleted',deleted_at=?,updated_at=? WHERE id=?",
                (now, now, account_id),
            )
        return {"id": account_id, "name": row["name"]}

    def record_signal_and_order(
        self,
        *,
        account_id: str,
        strategy_id: str,
        symbol: str,
        name: str,
        side: str,
        signal_date: date,
        score: float | None,
        reason: str,
        signal_ref: str | None,
        requested_qty: int,
        target_amount: float,
        target_weight: float,
        planned_session: str,
        scheduled_date: date | None = None,
        payload: dict[str, Any] | None = None,
        frozen_at: datetime | None = None,
    ) -> tuple[str, str, bool]:
        frozen = _now_text(frozen_at)
        signal_id = stable_id(account_id, symbol, side, signal_date, reason)
        order_id = stable_id(account_id, signal_id, planned_session)
        created = False
        with self.transaction() as conn:
            account = conn.execute(
                "SELECT status FROM accounts WHERE id=? AND status!='deleted'", (account_id,)
            ).fetchone()
            if account is None:
                raise KeyError(account_id)
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO signal_intents(
                    id,account_id,strategy_id,symbol,name,side,signal_date,score,reason,
                    signal_ref,payload_json,frozen_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    signal_id, account_id, strategy_id, symbol, name, side,
                    signal_date.isoformat(), score, reason, signal_ref, _json(payload or {}), frozen,
                ),
            )
            conn.execute(
                """INSERT OR IGNORE INTO orders(
                    id,account_id,signal_id,symbol,name,side,requested_qty,target_amount,
                    target_weight,planned_session,scheduled_date,status,reason,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_id, account_id, signal_id, symbol, name, side, int(requested_qty),
                    float(target_amount), float(target_weight), planned_session,
                    scheduled_date.isoformat() if scheduled_date else None,
                    "PLANNED", "等待下一真实交易日", frozen, frozen,
                ),
            )
            created = conn.total_changes > before
            if created:
                self._event(
                    conn,
                    event_key=f"{signal_id}:SIGNAL_FROZEN",
                    account_id=account_id,
                    event_type="SIGNAL_FROZEN",
                    occurred_at=frozen,
                    trading_date=signal_date.isoformat(),
                    entity_type="signal",
                    entity_id=signal_id,
                    title=f"{name or symbol} {side} 信号已冻结",
                    detail=f"数量 {requested_qty} 股, 计划 {planned_session}",
                    payload={"score": score, "reason": reason},
                )
                self._event(
                    conn,
                    event_key=f"{order_id}:ORDER_PLANNED",
                    account_id=account_id,
                    event_type="ORDER_PLANNED",
                    occurred_at=frozen,
                    trading_date=signal_date.isoformat(),
                    entity_type="order",
                    entity_id=order_id,
                    title=f"{name or symbol} 订单计划已生成",
                    detail=f"{side} {requested_qty} 股, 等待真实执行时钟",
                )
            conn.execute(
                "UPDATE accounts SET last_signal_date=?,updated_at=? WHERE id=?",
                (signal_date.isoformat(), frozen, account_id),
            )
        return signal_id, order_id, created

    def record_skipped_signal(
        self,
        *,
        account_id: str,
        strategy_id: str,
        symbol: str,
        name: str,
        side: str,
        signal_date: date,
        score: float | None,
        reason: str,
        signal_ref: str | None,
        skip_code: str,
        detail: str,
        payload: dict[str, Any] | None = None,
        frozen_at: datetime | None = None,
    ) -> tuple[str, bool]:
        """Freeze a valid signal even when risk sizing cannot create an order."""
        frozen = _now_text(frozen_at)
        signal_id = stable_id(account_id, symbol, side, signal_date, reason)
        event_payload = {
            "score": score, "reason": reason, "skip_code": skip_code, **(payload or {})
        }
        with self.transaction() as conn:
            account = conn.execute(
                "SELECT status FROM accounts WHERE id=? AND status!='deleted'", (account_id,)
            ).fetchone()
            if account is None:
                raise KeyError(account_id)
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO signal_intents(
                    id,account_id,strategy_id,symbol,name,side,signal_date,score,reason,
                    signal_ref,payload_json,frozen_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    signal_id, account_id, strategy_id, symbol, name, side,
                    signal_date.isoformat(), score, reason, signal_ref,
                    _json(event_payload), frozen,
                ),
            )
            created = conn.total_changes > before
            if created:
                self._event(
                    conn,
                    event_key=f"{signal_id}:SIGNAL_FROZEN",
                    account_id=account_id,
                    event_type="SIGNAL_FROZEN",
                    occurred_at=frozen,
                    trading_date=signal_date.isoformat(),
                    entity_type="signal",
                    entity_id=signal_id,
                    title=f"{name or symbol} {side} 信号已冻结",
                    detail="策略信号已保留, 等待风控生成订单",
                    payload={"score": score, "reason": reason},
                )
                self._event(
                    conn,
                    event_key=f"{signal_id}:SIGNAL_SKIPPED:{skip_code}",
                    account_id=account_id,
                    event_type="SIGNAL_SKIPPED",
                    occurred_at=frozen,
                    trading_date=signal_date.isoformat(),
                    entity_type="signal",
                    entity_id=signal_id,
                    severity="warning",
                    title=f"{name or symbol} 信号未形成订单",
                    detail=detail,
                    payload=event_payload,
                )
        return signal_id, created

    def mark_signal_day(self, account_id: str, signal_date: date) -> None:
        """Record a sealed signal day even when it produced no order."""
        now = _now_text()
        with self.transaction() as conn:
            result = conn.execute(
                """UPDATE accounts SET last_signal_date=?,updated_at=?
                    WHERE id=? AND status!='deleted'""",
                (signal_date.isoformat(), now, account_id),
            )
            if result.rowcount != 1:
                raise KeyError(account_id)

    def assign_due_date(self, order_id: str, trading_date: date, preflight: dict[str, Any]) -> None:
        now = _now_text()
        with self.transaction() as conn:
            order = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            if order is None:
                raise KeyError(order_id)
            if order["status"] in TERMINAL_ORDER_STATUSES:
                return
            conn.execute(
                """UPDATE orders SET scheduled_date=?,status='PREFLIGHT_OK',preflight_json=?,
                    updated_at=? WHERE id=?""",
                (trading_date.isoformat(), _json(preflight), now, order_id),
            )
            self._event(
                conn,
                event_key=f"{order_id}:PREFLIGHT:{trading_date.isoformat()}",
                account_id=order["account_id"],
                event_type="PREFLIGHT_PASSED",
                occurred_at=now,
                trading_date=trading_date.isoformat(),
                entity_type="order",
                entity_id=order_id,
                title=f"{order['name'] or order['symbol']} 盘前校验通过",
                detail="交易日、资金和参考行情校验完成",
                payload=preflight,
            )

    def requeue_invalid_session_order(
        self,
        order_id: str,
        invalid_date: date,
    ) -> bool:
        """Compensate a non-session scheduling error without deleting audit history."""
        now = _now_text()
        reason = "非交易日误调度已纠正; 订单等待下一个真实交易日"
        recoverable = {
            "PLANNED",
            "PREFLIGHT_OK",
            "MISSED_EXECUTION",
            "UNKNOWN_MARKET_DATA",
        }
        with self.transaction() as conn:
            order = conn.execute(
                """SELECT o.*,a.status AS account_status
                    FROM orders o JOIN accounts a ON a.id=o.account_id
                    WHERE o.id=?""",
                (order_id,),
            ).fetchone()
            if order is None:
                raise KeyError(order_id)
            if (
                order["account_status"] == "deleted"
                or order["status"] not in recoverable
                or order["scheduled_date"] != invalid_date.isoformat()
                or int(order["filled_qty"]) != 0
            ):
                return False
            if conn.execute(
                "SELECT 1 FROM fills WHERE order_id=? LIMIT 1", (order_id,)
            ).fetchone() is not None:
                return False
            conn.execute(
                """UPDATE orders SET scheduled_date=NULL,status='PLANNED',reason=?,
                    execution_quality=NULL,preflight_json='{}',terminal_at=NULL,updated_at=?
                    WHERE id=?""",
                (reason, now, order_id),
            )
            conn.execute(
                """UPDATE incidents SET status='resolved',resolved_at=?
                    WHERE entity_type='order' AND entity_id=? AND status='open'""",
                (now, order_id),
            )
            self._event(
                conn,
                event_key=(
                    f"{order_id}:INVALID_SESSION_REQUEUED:{invalid_date.isoformat()}"
                ),
                account_id=order["account_id"],
                event_type="INVALID_SESSION_REQUEUED",
                occurred_at=now,
                trading_date=invalid_date.isoformat(),
                entity_type="order",
                entity_id=order_id,
                severity="warning",
                title=f"{order['name'] or order['symbol']} 非交易日误调度已纠正",
                detail=reason,
                payload={
                    "invalid_date": invalid_date.isoformat(),
                    "previous_status": order["status"],
                },
            )
        return True

    def mark_historically_missed(
        self,
        order_id: str,
        trading_date: date,
        evidence: dict[str, Any],
    ) -> bool:
        """Freeze a past intended session without inventing a successful preflight."""
        now = _now_text()
        reason = "服务恢复后确认订单原定交易日已过去; 进入历史开盘证据恢复"
        with self.transaction() as conn:
            order = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            if order is None:
                raise KeyError(order_id)
            if order["status"] in TERMINAL_ORDER_STATUSES:
                return False
            scheduled = order["scheduled_date"]
            if scheduled is not None and scheduled != trading_date.isoformat():
                raise PaperLedgerError(
                    f"订单已冻结到其他交易日: {scheduled} != {trading_date.isoformat()}"
                )
            conn.execute(
                """UPDATE orders SET scheduled_date=?,status='MISSED_EXECUTION',reason=?,
                    execution_quality='MISSED_EXECUTION',preflight_json=?,terminal_at=?,
                    updated_at=? WHERE id=?""",
                (
                    trading_date.isoformat(),
                    reason,
                    _json(evidence),
                    now,
                    now,
                    order_id,
                ),
            )
            self._event(
                conn,
                event_key=f"{order_id}:HISTORICAL_MISSED:{trading_date.isoformat()}",
                account_id=order["account_id"],
                event_type="MISSED_EXECUTION",
                occurred_at=now,
                trading_date=trading_date.isoformat(),
                entity_type="order",
                entity_id=order_id,
                severity="critical",
                title=f"{order['name'] or order['symbol']} 错过原定开盘",
                detail=reason,
                payload=evidence,
            )
            self._upsert_incident_tx(
                conn,
                account_id=order["account_id"],
                incident_key=f"order:{order_id}:MISSED_EXECUTION",
                code="MISSED_EXECUTION",
                severity="critical",
                title=f"{order['name'] or order['symbol']} 执行异常",
                detail=reason,
                entity_type="order",
                entity_id=order_id,
                now=now,
            )
        return True

    def terminal_order(
        self,
        order_id: str,
        *,
        status: str,
        reason: str,
        quality: str | None = None,
        occurred_at: datetime | None = None,
        scheduled_date: date | None = None,
        severity: str = "warning",
    ) -> None:
        if status not in TERMINAL_ORDER_STATUSES:
            raise ValueError(f"非终态订单状态: {status}")
        now = _now_text(occurred_at)
        with self.transaction() as conn:
            order = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            if order is None:
                raise KeyError(order_id)
            recovering_unknown = (
                order["status"] == "UNKNOWN_MARKET_DATA"
                and quality == "RECOVERED_LATE"
                and status != "UNKNOWN_MARKET_DATA"
            )
            if (
                order["status"] in TERMINAL_ORDER_STATUSES
                and order["status"] != "MISSED_EXECUTION"
                and not recovering_unknown
            ):
                return
            conn.execute(
                """UPDATE orders SET status=?,reason=?,execution_quality=?,terminal_at=?,updated_at=?,
                    scheduled_date=coalesce(scheduled_date,?)
                    WHERE id=?""",
                (
                    status,
                    reason,
                    quality,
                    now,
                    now,
                    scheduled_date.isoformat() if scheduled_date else None,
                    order_id,
                ),
            )
            for prior_status in ("MISSED_EXECUTION", "UNKNOWN_MARKET_DATA"):
                if order["status"] == prior_status and status != prior_status:
                    conn.execute(
                        """UPDATE incidents SET status='resolved',resolved_at=?
                            WHERE incident_key=? AND status='open'""",
                        (now, f"order:{order_id}:{prior_status}"),
                    )
            self._event(
                conn,
                event_key=f"{order_id}:TERMINAL:{status}",
                account_id=order["account_id"],
                event_type=status,
                occurred_at=now,
                trading_date=(
                    scheduled_date.isoformat()
                    if scheduled_date
                    else order["scheduled_date"]
                ),
                entity_type="order",
                entity_id=order_id,
                severity=severity,
                title=f"{order['name'] or order['symbol']} · {status}",
                detail=reason,
                payload={"quality": quality},
            )
            if status in {"MISSED_EXECUTION", "UNKNOWN_MARKET_DATA", "EXECUTION_FAILED"}:
                self._upsert_incident_tx(
                    conn,
                    account_id=order["account_id"],
                    incident_key=f"order:{order_id}:{status}",
                    code=status,
                    severity="critical",
                    title=f"{order['name'] or order['symbol']} 执行异常",
                    detail=reason,
                    entity_type="order",
                    entity_id=order_id,
                    now=now,
                )

    def execute_fill(
        self,
        order_id: str,
        *,
        price: float,
        quantity: int,
        quote_at: datetime,
        source: str,
        quality: str = "ON_TIME",
        previous_close: float | None = None,
    ) -> str:
        if price <= 0 or quantity <= 0:
            raise ValueError("成交价和数量必须为正数")
        now = _now_text()
        quote_text = _now_text(quote_at)
        fill_id = stable_id(order_id, quality, quote_text, quantity, f"{price:.6f}")
        with self.transaction() as conn:
            order = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            if order is None:
                raise KeyError(order_id)
            existing = conn.execute("SELECT id FROM fills WHERE order_id=?", (order_id,)).fetchone()
            if existing is not None:
                return str(existing["id"])
            recoverable_terminal = (
                quality == "RECOVERED_LATE"
                and order["status"] in {"MISSED_EXECUTION", "UNKNOWN_MARKET_DATA"}
            )
            if order["status"] in TERMINAL_ORDER_STATUSES and not recoverable_terminal:
                raise PaperLedgerError(f"订单已终结: {order['status']}")
            account = conn.execute("SELECT * FROM accounts WHERE id=?", (order["account_id"],)).fetchone()
            config = _loads(account["config_json"], {})
            model = TradingCostModel(
                commission_pct=float(config.get("commission_pct", 0.0002)),
                stamp_tax_pct=float(config.get("stamp_tax_pct", 0.001)),
                slippage_bps=float(config.get("slippage_bps", 5.0)),
            )
            side = str(order["side"])
            gross = float(price) * int(quantity)
            quote_date = quote_text[:10]
            execution_date = now[:10]
            day_pnl = 0.0
            if side == "BUY":
                cash_total = model.buy_cash_required(price, quantity)
                if cash_total > float(account["cash_balance"]) + 1e-6:
                    raise PaperLedgerError("可用资金不足")
                cash_delta = -cash_total
                fee_amount = cash_total - gross
                day_pnl = gross - cash_total
            else:
                position = conn.execute(
                    "SELECT * FROM positions WHERE account_id=? AND symbol=?",
                    (order["account_id"], order["symbol"]),
                ).fetchone()
                if position is None or int(position["available_qty"]) < quantity:
                    raise PaperLedgerError("可卖数量不足或受 T+1 锁定")
                cash_total = model.sell_cash_received(price, quantity)
                cash_delta = cash_total
                fee_amount = gross - cash_total
                reference_price = (
                    float(previous_close)
                    if previous_close is not None and previous_close > 0
                    else float(position["average_price"])
                )
                day_pnl = cash_total - reference_price * quantity
            balance_after = float(account["cash_balance"]) + cash_delta
            if balance_after < -0.01:
                raise PaperLedgerError("资金账将变为负数")
            conn.execute(
                """INSERT INTO fills(
                    id,account_id,order_id,symbol,side,quantity,price,gross_amount,
                    fee_amount,cash_delta,executed_at,quote_at,source,quality,
                    reference_price,day_pnl
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fill_id, order["account_id"], order_id, order["symbol"], side,
                    quantity, price, gross, fee_amount, cash_delta, now, quote_text, source, quality,
                    previous_close, day_pnl,
                ),
            )
            conn.execute(
                """INSERT INTO cash_entries(
                    id,account_id,event_type,amount,balance_after,reference_id,occurred_at,detail
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    stable_id(fill_id, "CASH"), order["account_id"], f"{side}_FILL",
                    cash_delta, balance_after, fill_id, now,
                    f"{order['symbol']} {side} {quantity} @ {price:.4f}",
                ),
            )
            conn.execute(
                "UPDATE accounts SET cash_balance=?,updated_at=? WHERE id=?",
                (balance_after, now, order["account_id"]),
            )
            if side == "BUY":
                current = conn.execute(
                    "SELECT * FROM positions WHERE account_id=? AND symbol=?",
                    (order["account_id"], order["symbol"]),
                ).fetchone()
                cost = -cash_delta
                if current is None:
                    unlocked_quantity = quantity if quote_date < execution_date else 0
                    locked_quantity = quantity - unlocked_quantity
                    conn.execute(
                        """INSERT INTO positions(
                            account_id,symbol,name,quantity,available_qty,locked_qty,
                            average_price,cost_basis,acquired_on,max_price,last_price,
                            market_value,unrealized_pnl,previous_close,day_pnl,pnl_date,
                            today_bought_qty,today_bought_cost,quote_at,quote_source
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            order["account_id"], order["symbol"], order["name"], quantity,
                            unlocked_quantity, locked_quantity, cost / quantity, cost,
                            quote_text[:10], price, price,
                            gross, gross - cost, previous_close, gross - cost, quote_date,
                            quantity, cost, quote_text, source,
                        ),
                    )
                else:
                    new_qty = int(current["quantity"]) + quantity
                    new_cost = float(current["cost_basis"]) + cost
                    same_pnl_day = current["pnl_date"] == quote_date
                    bought_qty = int(current["today_bought_qty"] or 0) if same_pnl_day else 0
                    bought_cost = float(current["today_bought_cost"] or 0) if same_pnl_day else 0
                    old_qty = int(current["quantity"]) - bought_qty
                    reference = (
                        float(previous_close)
                        if previous_close is not None and previous_close > 0
                        else float(current["previous_close"] or current["average_price"])
                    )
                    current_day_pnl = (
                        old_qty * (price - reference)
                        + bought_qty * price - bought_cost
                    )
                    unlocked_quantity = quantity if quote_date < execution_date else 0
                    locked_quantity = quantity - unlocked_quantity
                    conn.execute(
                        """UPDATE positions SET quantity=?,available_qty=available_qty+?,
                            locked_qty=locked_qty+?,
                            average_price=?,cost_basis=?,last_price=?,market_value=?,
                            unrealized_pnl=?,previous_close=?,day_pnl=?,pnl_date=?,
                            today_bought_qty=?,today_bought_cost=?,quote_at=?,quote_source=?,
                            max_price=max(max_price,?)
                            WHERE account_id=? AND symbol=?""",
                        (
                            new_qty, unlocked_quantity, locked_quantity,
                            new_cost / new_qty, new_cost, price,
                            new_qty * price, new_qty * price - new_cost, reference,
                            current_day_pnl + gross - cost, quote_date,
                            bought_qty + quantity, bought_cost + cost, quote_text, source, price,
                            order["account_id"], order["symbol"],
                        ),
                    )
            else:
                current = conn.execute(
                    "SELECT * FROM positions WHERE account_id=? AND symbol=?",
                    (order["account_id"], order["symbol"]),
                ).fetchone()
                remaining = int(current["quantity"]) - quantity
                remaining_cost = float(current["cost_basis"]) * remaining / int(current["quantity"])
                if remaining <= 0:
                    conn.execute(
                        "DELETE FROM positions WHERE account_id=? AND symbol=?",
                        (order["account_id"], order["symbol"]),
                    )
                else:
                    bought_qty = (
                        int(current["today_bought_qty"] or 0)
                        if current["pnl_date"] == quote_date else 0
                    )
                    bought_cost = (
                        float(current["today_bought_cost"] or 0)
                        if current["pnl_date"] == quote_date else 0
                    )
                    reference = (
                        float(previous_close)
                        if previous_close is not None and previous_close > 0
                        else float(current["previous_close"] or current["average_price"])
                    )
                    older_remaining = max(remaining - bought_qty, 0)
                    remaining_day_pnl = (
                        older_remaining * (price - reference)
                        + bought_qty * price - bought_cost
                    )
                    conn.execute(
                        """UPDATE positions SET quantity=?,available_qty=available_qty-?,
                            cost_basis=?,market_value=?,unrealized_pnl=?,last_price=?,
                            previous_close=?,day_pnl=?,pnl_date=?,quote_at=?,quote_source=?,
                            pending_exit_reason=NULL,pending_exit_date=NULL
                            WHERE account_id=? AND symbol=?""",
                        (
                            remaining, quantity, remaining_cost, remaining * price,
                            remaining * price - remaining_cost, price, reference,
                            remaining_day_pnl, quote_date, quote_text, source,
                            order["account_id"], order["symbol"],
                        ),
                    )
            order_status = "FILLED" if quantity >= int(order["requested_qty"]) else "PARTIALLY_FILLED"
            conn.execute(
                """UPDATE orders SET filled_qty=?,status=?,reason=?,execution_quality=?,
                    terminal_at=?,updated_at=? WHERE id=?""",
                (
                    quantity, order_status, "成交已写入资金与持仓账", quality,
                    now, now, order_id,
                ),
            )
            for prior_status in ("MISSED_EXECUTION", "UNKNOWN_MARKET_DATA"):
                if order["status"] == prior_status:
                    conn.execute(
                        """UPDATE incidents SET status='resolved',resolved_at=?
                            WHERE incident_key=? AND status='open'""",
                        (now, f"order:{order_id}:{prior_status}"),
                    )
            self._event(
                conn,
                event_key=f"{order_id}:FILL:{fill_id}",
                account_id=order["account_id"],
                event_type=order_status,
                occurred_at=now,
                trading_date=quote_text[:10],
                entity_type="fill",
                entity_id=fill_id,
                title=f"{order['name'] or order['symbol']} {side} 成交",
                detail=f"{quantity} 股 x {price:.4f}, 费用 {fee_amount:.2f}",
                severity="warning" if quality == "RECOVERED_LATE" else "info",
                payload={"quality": quality, "source": source, "quote_at": quote_text},
            )
        return fill_id

    def update_marks(
        self,
        quotes: dict[str, dict[str, Any]],
        *,
        source: str,
    ) -> int:
        updated = 0
        with self.transaction() as conn:
            rows = conn.execute(
                """SELECT p.* FROM positions p JOIN accounts a ON a.id=p.account_id
                    WHERE p.quantity>0 AND a.status!='deleted'"""
            ).fetchall()
            for row in rows:
                quote = quotes.get(str(row["symbol"]))
                if not quote:
                    continue
                price = float(quote.get("last_price") or quote.get("close") or 0)
                if price <= 0:
                    continue
                quote_at = str(quote.get("quote_at") or _now_text())
                quote_date = quote_at[:10]
                previous_close = quote.get("prev_close")
                reference = (
                    float(previous_close)
                    if previous_close is not None and float(previous_close) > 0
                    else float(row["previous_close"] or row["average_price"])
                )
                same_pnl_day = row["pnl_date"] == quote_date
                bought_qty = int(row["today_bought_qty"] or 0) if same_pnl_day else 0
                bought_cost = float(row["today_bought_cost"] or 0) if same_pnl_day else 0
                older_qty = int(row["quantity"]) - bought_qty
                market_value = int(row["quantity"]) * price
                day_pnl = older_qty * (price - reference) + bought_qty * price - bought_cost
                materially_changed = (
                    abs(float(row["last_price"] or 0) - price) > 1e-9
                    or abs(float(row["market_value"] or 0) - market_value) > 1e-6
                    or abs(
                        float(row["unrealized_pnl"] or 0)
                        - (market_value - float(row["cost_basis"]))
                    ) > 1e-6
                    or abs(float(row["day_pnl"] or 0) - day_pnl) > 1e-6
                )
                conn.execute(
                    """UPDATE positions SET last_price=?,market_value=?,unrealized_pnl=?,
                        previous_close=?,day_pnl=?,pnl_date=?,today_bought_qty=?,
                        today_bought_cost=?,quote_at=?,quote_source=?,max_price=max(max_price,?)
                        WHERE account_id=? AND symbol=?""",
                    (
                        price, market_value, market_value - float(row["cost_basis"]),
                        reference, day_pnl, quote_date, bought_qty, bought_cost,
                        quote_at, source, price, row["account_id"], row["symbol"],
                    ),
                )
                updated += int(materially_changed)
        return updated

    def unlock_positions(self, trading_date: date) -> int:
        with self.transaction() as conn:
            result = conn.execute(
                """UPDATE positions SET available_qty=quantity,locked_qty=0
                    WHERE quantity>0 AND acquired_on<? AND locked_qty>0""",
                (trading_date.isoformat(),),
            )
            return result.rowcount

    def planned_orders(self, *, account_id: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT o.*,s.signal_date,s.reason AS signal_reason,s.score
            FROM orders o JOIN signal_intents s ON s.id=o.signal_id
            WHERE o.status IN ('PLANNED','PREFLIGHT_OK','MISSED_EXECUTION')"""
        params: tuple[Any, ...] = ()
        if account_id:
            sql += " AND account_id=?"
            params = (account_id,)
        sql += " ORDER BY created_at,id"
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def recovery_orders(self, *, account_id: str | None = None) -> list[sqlite3.Row]:
        """Orders whose execution-window result awaits later reliable evidence."""
        sql = """SELECT o.*,s.signal_date,s.reason AS signal_reason,s.score
            FROM orders o
            JOIN signal_intents s ON s.id=o.signal_id
            JOIN accounts a ON a.id=o.account_id
            WHERE o.status IN ('MISSED_EXECUTION','UNKNOWN_MARKET_DATA')
            AND o.scheduled_date IS NOT NULL
            AND a.status!='deleted'"""
        params: tuple[Any, ...] = ()
        if account_id:
            sql += " AND o.account_id=?"
            params = (account_id,)
        sql += " ORDER BY o.scheduled_date,o.created_at,o.id"
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def accounts_needing_settlement_restatement(self, trading_date: date) -> set[str]:
        """Accounts whose same-day settlement predates a recovered-late fill."""
        day = trading_date.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT f.account_id
                    FROM fills f
                    JOIN accounts a ON a.id=f.account_id AND a.status!='deleted'
                    WHERE f.quality='RECOVERED_LATE'
                    AND substr(f.quote_at,1,10)=?
                    AND EXISTS (
                        SELECT 1 FROM nav_snapshots n
                        WHERE n.account_id=f.account_id AND n.trading_date=?
                    )
                    AND f.executed_at > (
                        SELECT max(n.captured_at) FROM nav_snapshots n
                        WHERE n.account_id=f.account_id AND n.trading_date=?
                    )""",
                (day, day, day),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def settled_accounts_holding_symbols(
        self,
        trading_date: date,
        symbols: set[str],
    ) -> set[str]:
        """Return settled accounts whose close valuation uses any selected symbol."""
        if not symbols:
            return set()
        placeholders = ",".join("?" for _ in symbols)
        day = trading_date.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT DISTINCT p.account_id
                    FROM positions p
                    JOIN accounts a ON a.id=p.account_id AND a.status!='deleted'
                    JOIN nav_snapshots n ON n.account_id=p.account_id
                        AND n.trading_date=?
                    WHERE p.quantity>0 AND p.symbol IN ({placeholders})""",
                (day, *sorted(symbols)),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def tracked_symbols(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT p.symbol FROM positions p JOIN accounts a ON a.id=p.account_id
                    WHERE p.quantity>0 AND a.status!='deleted'
                    UNION SELECT o.symbol FROM orders o JOIN accounts a ON a.id=o.account_id
                    WHERE a.status!='deleted'
                    AND o.status IN (
                        'PLANNED','PREFLIGHT_OK','MISSED_EXECUTION','UNKNOWN_MARKET_DATA'
                    )"""
            ).fetchall()
        return {str(row[0]) for row in rows if row[0]}

    def position_rows(self, *, account_id: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT p.* FROM positions p JOIN accounts a ON a.id=p.account_id
            WHERE p.quantity>0 AND a.status!='deleted'"""
        params: tuple[Any, ...] = ()
        if account_id:
            sql += " AND p.account_id=?"
            params = (account_id,)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY p.account_id,p.symbol", params).fetchall()

    def mark_position_exit_triggered(
        self,
        account_id: str,
        symbol: str,
        *,
        reason: str,
        trading_date: date,
        locked: bool,
    ) -> None:
        now = _now_text()
        with self.transaction() as conn:
            position = conn.execute(
                "SELECT * FROM positions WHERE account_id=? AND symbol=?",
                (account_id, symbol),
            ).fetchone()
            if position is None:
                return
            conn.execute(
                """UPDATE positions SET pending_exit_reason=?,pending_exit_date=?
                    WHERE account_id=? AND symbol=?""",
                (reason, trading_date.isoformat(), account_id, symbol),
            )
            self._event(
                conn,
                event_key=f"{account_id}:{symbol}:EXIT_TRIGGER:{trading_date}:{reason}",
                account_id=account_id,
                event_type="EXIT_TRIGGERED_T1_LOCKED" if locked else "EXIT_TRIGGERED",
                occurred_at=now,
                trading_date=trading_date.isoformat(),
                entity_type="position",
                entity_id=f"{account_id}:{symbol}",
                severity="warning" if locked else "info",
                title=f"{position['name'] or symbol} 已触发退出",
                detail=(
                    f"{reason} 已触发, 但今日买入受 T+1 锁定, 等待可卖"
                    if locked else f"{reason} 已触发, 等待下一有效行情执行"
                ),
            )

    def record_account_event(
        self,
        account_id: str,
        *,
        event_key: str,
        event_type: str,
        title: str,
        detail: str,
        trading_date: date | None = None,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction() as conn:
            if conn.execute("SELECT id FROM accounts WHERE id=?", (account_id,)).fetchone() is None:
                raise KeyError(account_id)
            self._event(
                conn,
                event_key=event_key,
                account_id=account_id,
                event_type=event_type,
                occurred_at=_now_text(),
                trading_date=trading_date.isoformat() if trading_date else None,
                entity_type="account",
                entity_id=account_id,
                severity=severity,
                title=title,
                detail=detail,
                payload=payload,
            )

    @staticmethod
    def _upsert_incident_tx(
        conn: sqlite3.Connection,
        *,
        account_id: str,
        incident_key: str,
        code: str,
        severity: str,
        title: str,
        detail: str,
        entity_type: str | None,
        entity_id: str | None,
        now: str,
    ) -> None:
        conn.execute(
            """INSERT INTO incidents(
                id,account_id,incident_key,code,severity,status,opened_at,title,detail,
                entity_type,entity_id
            ) VALUES(?,?,?,?,?,'open',?,?,?,?,?)
            ON CONFLICT(incident_key) DO UPDATE SET
                severity=excluded.severity,status='open',resolved_at=NULL,
                title=excluded.title,detail=excluded.detail""",
            (
                stable_id("incident", incident_key), account_id, incident_key, code,
                severity, now, title, detail, entity_type, entity_id,
            ),
        )

    def open_incident(
        self,
        *,
        account_id: str,
        incident_key: str,
        code: str,
        severity: str,
        title: str,
        detail: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> None:
        now = _now_text()
        with self.transaction() as conn:
            self._upsert_incident_tx(
                conn,
                account_id=account_id,
                incident_key=incident_key,
                code=code,
                severity=severity,
                title=title,
                detail=detail,
                entity_type=entity_type,
                entity_id=entity_id,
                now=now,
            )

    def resolve_incident(self, incident_key: str) -> bool:
        now = _now_text()
        with self.transaction() as conn:
            result = conn.execute(
                """UPDATE incidents SET status='resolved',resolved_at=?
                    WHERE incident_key=? AND status='open'""",
                (now, incident_key),
            )
            return result.rowcount == 1

    def reconcile(self, account_id: str, *, open_incident: bool = True) -> dict[str, Any]:
        with self._connect() as conn:
            account = conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
            if account is None:
                raise KeyError(account_id)
            cash_sum = float(conn.execute(
                "SELECT coalesce(sum(amount),0) FROM cash_entries WHERE account_id=?", (account_id,)
            ).fetchone()[0])
            buys = conn.execute(
                """SELECT symbol,coalesce(sum(CASE WHEN side='BUY' THEN quantity ELSE -quantity END),0)
                    FROM fills WHERE account_id=? GROUP BY symbol""",
                (account_id,),
            ).fetchall()
            expected = {str(row[0]): int(row[1]) for row in buys if int(row[1]) != 0}
            actual_rows = conn.execute(
                "SELECT symbol,quantity FROM positions WHERE account_id=? AND quantity>0", (account_id,)
            ).fetchall()
            actual = {str(row[0]): int(row[1]) for row in actual_rows}
        cash_ok = abs(cash_sum - float(account["cash_balance"])) < 0.01
        position_ok = expected == actual
        result = {
            "ok": cash_ok and position_ok,
            "cash_ok": cash_ok,
            "position_ok": position_ok,
            "ledger_cash": round(cash_sum, 2),
            "account_cash": round(float(account["cash_balance"]), 2),
            "expected_positions": expected,
            "actual_positions": actual,
            "checked_at": _now_text(),
        }
        if open_incident and not result["ok"]:
            self.open_incident(
                account_id=account_id,
                incident_key=f"account:{account_id}:LEDGER_IMBALANCE",
                code="LEDGER_IMBALANCE",
                severity="critical",
                title="资金账、持仓账与成交账不平",
                detail=_json(result),
                entity_type="account",
                entity_id=account_id,
            )
        return result

    def settle_account(
        self,
        account_id: str,
        trading_date: date,
        *,
        source: str,
        restatement: bool = False,
    ) -> dict[str, Any]:
        now = _now_text()
        with self.transaction() as conn:
            account = conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
            if account is None:
                raise KeyError(account_id)
            positions = conn.execute(
                "SELECT * FROM positions WHERE account_id=? AND quantity>0", (account_id,)
            ).fetchall()
            market_value = sum(float(row["market_value"]) for row in positions)
            unrealized = sum(float(row["unrealized_pnl"]) for row in positions)
            equity = float(account["cash_balance"]) + market_value
            snapshot_id = stable_id(account_id, trading_date, source)
            already_settled = conn.execute(
                "SELECT 1 FROM nav_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone() is not None
            conn.execute(
                """INSERT OR REPLACE INTO nav_snapshots(
                    id,account_id,trading_date,captured_at,cash,market_value,equity,
                    unrealized_pnl,source
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot_id, account_id, trading_date.isoformat(), now,
                    account["cash_balance"], market_value, equity, unrealized, source,
                ),
            )
            if not already_settled:
                conn.execute(
                    """UPDATE positions SET hold_days=hold_days+1
                        WHERE account_id=? AND acquired_on<?""",
                    (account_id, trading_date.isoformat()),
                )
            conn.execute(
                "UPDATE accounts SET last_settlement_date=?,updated_at=? WHERE id=?",
                (trading_date.isoformat(), now, account_id),
            )
            self._event(
                conn,
                event_key=(
                    f"{account_id}:SETTLEMENT_RESTATED:{trading_date.isoformat()}:{now}:"
                    f"{equity:.6f}"
                    if restatement
                    else f"{account_id}:SETTLED:{trading_date.isoformat()}"
                ),
                account_id=account_id,
                event_type=("SETTLEMENT_RESTATED" if restatement else "ACCOUNT_SETTLED"),
                occurred_at=now,
                trading_date=trading_date.isoformat(),
                entity_type="account",
                entity_id=account_id,
                title=("收盘证据更新后重述结算" if restatement else "收盘结算完成"),
                detail=f"权益 {equity:.2f}, 现金 {float(account['cash_balance']):.2f}",
            )
        return self.get_account(account_id)

    def import_legacy_snapshot(self, account_id: str, source_path: Path, payload: dict[str, Any]) -> bool:
        with self.transaction() as conn:
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO legacy_snapshots(account_id,source_path,imported_at,payload_json)
                    VALUES(?,?,?,?)""",
                (account_id, source_path.name, _now_text(), _json(payload)),
            )
            return conn.total_changes > before

    def _account_projection(self, row: sqlite3.Row) -> dict[str, Any]:
        account_id = str(row["id"])
        with self._connect() as conn:
            signals = [dict(item) for item in conn.execute(
                """SELECT s.*,
                       (SELECT o.id FROM orders o WHERE o.signal_id=s.id LIMIT 1) AS order_id,
                       EXISTS(
                           SELECT 1 FROM events e
                           WHERE e.entity_type='signal' AND e.entity_id=s.id
                             AND e.event_type='SIGNAL_SKIPPED'
                       ) AS skipped
                    FROM signal_intents s
                    WHERE s.account_id=?
                    ORDER BY s.frozen_at DESC,s.id DESC LIMIT 2000""",
                (account_id,),
            ).fetchall()]
            positions = [dict(item) for item in conn.execute(
                "SELECT * FROM positions WHERE account_id=? AND quantity>0 ORDER BY symbol", (account_id,)
            ).fetchall()]
            orders = [dict(item) for item in conn.execute(
                """SELECT o.*,s.signal_date,s.score,s.reason AS signal_reason,s.signal_ref
                    FROM orders o JOIN signal_intents s ON s.id=o.signal_id
                    WHERE o.account_id=? ORDER BY o.created_at DESC,o.id DESC LIMIT 200""", (account_id,)
            ).fetchall()]
            fills = [dict(item) for item in conn.execute(
                "SELECT * FROM fills WHERE account_id=? ORDER BY executed_at DESC,id DESC LIMIT 200", (account_id,)
            ).fetchall()]
            cash_entries = [dict(item) for item in conn.execute(
                "SELECT * FROM cash_entries WHERE account_id=? ORDER BY occurred_at DESC,id DESC LIMIT 200", (account_id,)
            ).fetchall()]
            timeline = [dict(item) for item in conn.execute(
                "SELECT * FROM events WHERE account_id=? ORDER BY occurred_at DESC,rowid DESC LIMIT 200",
                (account_id,),
            ).fetchall()]
            incidents = [dict(item) for item in conn.execute(
                "SELECT * FROM incidents WHERE account_id=? ORDER BY status,opened_at DESC", (account_id,)
            ).fetchall()]
            nav = [dict(item) for item in conn.execute(
                "SELECT * FROM nav_snapshots WHERE account_id=? ORDER BY trading_date", (account_id,)
            ).fetchall()]
        for order in orders:
            order["preflight"] = _loads(order.pop("preflight_json", "{}"), {})
        for signal in signals:
            signal["payload"] = _loads(signal.pop("payload_json", "{}"), {})
            signal["skipped"] = bool(signal["skipped"])
        for event in timeline:
            event["payload"] = _loads(event.pop("payload_json", "{}"), {})
        config = _loads(row["config_json"], {})
        market_value = sum(float(item["market_value"]) for item in positions)
        unrealized = sum(float(item["unrealized_pnl"]) for item in positions)
        today = cn_now().date().isoformat()
        positions_have_today_marks = all(item.get("pnl_date") == today for item in positions)
        position_day_pnl = sum(
            float(item.get("day_pnl") or 0) for item in positions if item.get("pnl_date") == today
        )
        realized_day_pnl = sum(
            float(item.get("day_pnl") or 0)
            for item in fills
            if item["side"] == "SELL" and str(item["quote_at"]).startswith(today)
        )
        today_pnl = position_day_pnl + realized_day_pnl if positions_have_today_marks else None
        cash = float(row["cash_balance"])
        equity = cash + market_value
        initial = float(row["initial_capital"])
        pending = [item for item in orders if item["status"] not in TERMINAL_ORDER_STATUSES]
        open_incidents = [item for item in incidents if item["status"] == "open"]
        execution_state = {
            "code": "error" if any(i["severity"] == "critical" for i in open_incidents) else (
                "waiting_open" if pending else "holding" if positions else "scanning"
            ),
            "label": "存在阻断异常" if any(i["severity"] == "critical" for i in open_incidents) else (
                "等待真实执行时钟" if pending else "持仓监控中" if positions else "等待下一封板信号"
            ),
            "detail": open_incidents[0]["detail"] if open_incidents else "页面状态直接来自事件账本",
            "next_action": "先处理红色异常" if open_incidents else "按交易时钟自动推进",
        }
        result = {
            "run_id": f"ledger-{account_id}",
            "config": config,
            "stats": {
                "initial_capital": initial,
                "final_equity": equity,
                "total_return": equity / initial - 1 if initial > 0 else 0,
                "n_trades": len(fills),
            },
            "equity_curve": [
                {
                    "date": item["trading_date"], "value": item["equity"],
                    "cash": item["cash"], "positions": None,
                    "exposure": item["market_value"] / item["equity"] if item["equity"] else 0,
                }
                for item in nav
            ],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "trades": fills,
            "open_positions": positions,
            "pending_orders": pending,
            "per_symbol_stats": [],
            "strategy_info": {"id": config.get("strategy_id"), "name": config.get("strategy_name")},
            "elapsed_ms": 0,
            "error": row["last_error"],
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_version": 2,
            "execution_policy": "event_driven",
            "id": account_id,
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "baseline_date": row["baseline_date"],
            "signal_start_date": row["signal_start_date"],
            "start_date": row["signal_start_date"],
            "last_processed_date": row["last_signal_date"],
            "last_run_at": row["updated_at"],
            "last_error": row["last_error"],
            "config": config,
            "summary": {
                "cash": round(cash, 2),
                "market_value": round(market_value, 2),
                "equity": round(equity, 2),
                "unrealized_pnl": round(unrealized, 2),
                "today_pnl": round(today_pnl, 2) if today_pnl is not None else None,
                "today_pnl_available": today_pnl is not None,
                "total_return": equity / initial - 1 if initial > 0 else 0,
                "exposure": market_value / equity if equity > 0 else 0,
                "position_count": len(positions),
                "pending_order_count": len(pending),
                "open_incident_count": len(open_incidents),
                "critical_incident_count": sum(
                    1 for item in open_incidents if item["severity"] == "critical"
                ),
            },
            "positions": positions,
            "signals": signals,
            "orders": orders,
            "fills": fills,
            "cash_entries": cash_entries,
            "timeline": timeline,
            "incidents": incidents,
            "nav": nav,
            "execution_state": execution_state,
            "result": result,
        }

    def get_account(self, account_id: str) -> dict[str, Any]:
        return self._account_projection(self.account_row(account_id))

    def list_accounts(self) -> list[dict[str, Any]]:
        return [self._account_projection(row) for row in self.list_account_rows()]
