"""Event-driven paper trading orchestration.

The service advances only from observed clock/data events. It never rebuilds an
account by replaying a historical backtest and never labels recovered work as
an on-time execution.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import polars as pl

from app.backtest.regime_alignment import build_regime_filter_mask
from app.market_time import CN_TZ, cn_now
from app.price_limits import is_risk_warning_name, price_limit_pct
from app.services.paper_ledger import (
    TERMINAL_ORDER_STATUSES,
    PaperLedger,
    PaperLedgerError,
)
from app.services.screener import ScreenerService
from app.trading_rules import (
    TradingCostModel,
    is_one_price_locked,
    is_same_day_t_plus_one_locked,
    round_lot_quantity,
)

logger = logging.getLogger(__name__)

OPEN_START = dt_time(9, 30)
OPEN_DEADLINE = dt_time(9, 31)
SETTLEMENT_TIME = dt_time(15, 5)
SIGNAL_SEAL_TIME = dt_time(15, 30)
QUOTE_STALE_SECONDS = 90


class PaperTradingStoreError(PaperLedgerError):
    """Compatibility error exposed by the API boundary."""


def _as_cn(value: datetime | None = None) -> datetime:
    current = value or cn_now()
    if current.tzinfo is None:
        return current.replace(tzinfo=CN_TZ)
    return current.astimezone(CN_TZ)


def _quote_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_cn(value)
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=CN_TZ)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str) and value:
        try:
            return _as_cn(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def _valid_price(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _scalar_limit_price(previous: float, limit_pct: float, *, up: bool) -> float:
    sign = 1 if up else -1
    numerator = round((1 + sign * limit_pct) * 100)
    cents = math.floor(previous * 100 + 0.5)
    return ((cents * numerator + 50) // 100) / 100


def _row_quote(record: dict[str, Any], *, fallback_source: str) -> dict[str, Any] | None:
    symbol = str(record.get("symbol") or "")
    quote_at = _quote_datetime(record.get("timestamp") or record.get("quote_ts") or record.get("quote_at"))
    if not symbol or quote_at is None:
        return None
    last_price = record.get("last_price", record.get("close"))
    return {
        **record,
        "symbol": symbol,
        "last_price": float(last_price) if _valid_price(last_price) else None,
        "quote_at": quote_at.isoformat(timespec="seconds"),
        "_quote_dt": quote_at,
        "source": str(record.get("source") or fallback_source),
    }


def _quote_map(records: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        quote = _row_quote(record, fallback_source=source)
        if quote is not None:
            result[quote["symbol"]] = quote
    return result


def _account_cost_model(config: dict[str, Any]) -> TradingCostModel:
    return TradingCostModel(
        commission_pct=float(config.get("commission_pct", 0.0002)),
        stamp_tax_pct=float(config.get("stamp_tax_pct", 0.001)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
    )


class PaperTradingStore:
    """Compatibility facade backed by the transactional ledger."""

    def __init__(self, data_dir: Path) -> None:
        self.ledger = PaperLedger(data_dir)

    def get(self, account_id: str) -> dict[str, Any]:
        return self.ledger.get_account(account_id)

    def list(self) -> list[dict[str, Any]]:
        return self.ledger.list_accounts()

    def create(
        self,
        *,
        name: str,
        start_date: date,
        config: dict[str, Any],
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        return self.ledger.create_account(
            name=name,
            baseline_date=start_date,
            config=config,
            created_at=created_at,
        )

    def set_status(self, account_id: str, status: str) -> dict[str, Any]:
        return self.ledger.set_status(account_id, status)

    def delete(self, account_id: str) -> dict[str, Any]:
        receipt = self.ledger.delete_account(account_id)
        return {**receipt, "status": "deleted"}


class PaperTradingService:
    """Coordinates the signal, execution, quote and settlement clocks."""

    def __init__(self, app_state) -> None:
        self.app_state = app_state
        self.repo = app_state.repo
        self.ledger = PaperLedger(self.repo.store.data_dir)
        self._lock = threading.RLock()
        self._migrate_legacy_json()

    def _migrate_legacy_json(self) -> None:
        legacy_root = self.ledger.root / "accounts"
        if not legacy_root.exists():
            return
        for path in sorted(legacy_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                account_id = str(payload["id"])
                try:
                    self.ledger.account_row(account_id, include_deleted=True)
                    continue
                except KeyError:
                    pass
                config = dict(payload.get("config") or {})
                result = dict(payload.get("result") or {})
                strategy_info = dict(result.get("strategy_info") or {})
                config.setdefault("exit_mode", "eod")
                config.setdefault("enforce_t_plus_one", True)
                config.setdefault(
                    "strategy_name",
                    strategy_info.get("name") or config.get("strategy_id") or "legacy",
                )
                baseline = date.fromisoformat(str(
                    payload.get("baseline_date")
                    or payload.get("signal_start_date")
                    or payload.get("start_date")
                ))
                try:
                    created_at = datetime.fromisoformat(str(payload.get("created_at") or ""))
                except ValueError:
                    created_at = datetime.combine(baseline, SIGNAL_SEAL_TIME, tzinfo=CN_TZ)
                self.ledger.create_account(
                    name=str(payload.get("name") or account_id),
                    baseline_date=baseline,
                    config=config,
                    account_id=account_id,
                    created_at=created_at,
                )
                self.ledger.import_legacy_snapshot(account_id, path, payload)
                self.ledger.record_account_event(
                    account_id,
                    event_key=f"{account_id}:LEGACY_MIGRATED",
                    event_type="LEGACY_REPLAY_MIGRATED",
                    trading_date=baseline,
                    severity="warning",
                    title="旧盘后回放账户已迁移",
                    detail="旧快照只保留为审计附件; 资金、信号和订单已进入事件账本, 不导入伪历史成交",
                )
                self._migrate_legacy_orders(account_id, payload)
            except Exception:
                logger.exception("legacy paper account migration failed: %s", path.name)

    def _signal_close(self, asset_type: str, symbol: str, signal_date: date) -> float | None:
        frame = self.repo.get_daily_asset(
            asset_type,
            symbol,
            signal_date,
            signal_date,
            ["raw_close", "close"],
        )
        if frame.is_empty():
            return None
        row = frame.row(-1, named=True)
        value = row.get("raw_close") or row.get("close")
        return float(value) if _valid_price(value) else None

    def _migrate_legacy_orders(self, account_id: str, payload: dict[str, Any]) -> None:
        config = dict(payload.get("config") or {})
        result = dict(payload.get("result") or {})
        orders = list(result.get("pending_orders") or [])
        if not orders:
            return
        capital = float(config.get("initial_capital", 0))
        max_positions = max(int(config.get("max_positions", 10)), 1)
        max_exposure = float(config.get("max_exposure_pct", 1.0))
        target = capital * max_exposure / max_positions
        model = _account_cost_model(config)
        for item in orders:
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            signal_date = date.fromisoformat(str(item.get("signal_date")))
            close = self._signal_close(str(config.get("asset_type", "stock")), symbol, signal_date)
            quantity = round_lot_quantity(target, close or 0, model)
            self.ledger.record_signal_and_order(
                account_id=account_id,
                strategy_id=str(config.get("strategy_id") or "legacy"),
                symbol=symbol,
                name=str(item.get("name") or symbol),
                side="BUY",
                signal_date=signal_date,
                score=float(item.get("score") or 0),
                reason="legacy_frozen_signal",
                signal_ref=item.get("entry_signal_id"),
                requested_qty=quantity,
                target_amount=target,
                target_weight=max_exposure / max_positions,
                planned_session="NEXT_OPEN",
                payload={"legacy_status": item.get("status")},
            )

    def subscription_symbols(self) -> set[str]:
        return self.ledger.tracked_symbols()

    def _quotes_from_cache(self) -> dict[str, dict[str, Any]]:
        service = getattr(self.app_state, "quote_service", None)
        if service is None:
            return {}
        frame, frame_date = service.get_enriched_today()
        if frame is None or frame.is_empty() or frame_date is None:
            return {}
        records = frame.to_dicts()
        result = _quote_map(records, source="realtime_cache")
        if result:
            return result
        status = service.status()
        fetched_ms = status.get("last_fetch_ms")
        if not fetched_ms:
            return {}
        quote_at = datetime.fromtimestamp(float(fetched_ms) / 1000, tz=CN_TZ)
        for record in records:
            symbol = str(record.get("symbol") or "")
            if not symbol:
                continue
            result[symbol] = {
                **record,
                "last_price": record.get("close"),
                "quote_at": quote_at.isoformat(timespec="seconds"),
                "_quote_dt": quote_at,
                "source": "realtime_cache",
            }
        return result

    @staticmethod
    def _market_is_observed(trading_date: date, quotes: dict[str, dict[str, Any]]) -> bool:
        return any(
            quote.get("_quote_dt") is not None
            and quote["_quote_dt"].date() == trading_date
            and _valid_price(quote.get("last_price"))
            for quote in quotes.values()
        )

    def preflight_all(
        self,
        *,
        now: datetime | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        current = _as_cn(now)
        trading_date = current.date()
        quotes = quotes if quotes is not None else self._quotes_from_cache()
        summary = {"checked": 0, "deferred": 0, "assigned": 0}
        with self._lock:
            self.ledger.unlock_positions(trading_date)
            if not self._market_is_observed(trading_date, quotes):
                for account in self.ledger.list_accounts():
                    if account["status"] != "active":
                        continue
                    self.ledger.open_incident(
                        account_id=account["id"],
                        incident_key=f"account:{account['id']}:TRADING_DAY_UNCONFIRMED:{trading_date}",
                        code="TRADING_DAY_UNCONFIRMED",
                        severity="warning",
                        title="尚未确认今日开市",
                        detail="没有带当日时间戳的有效行情, 订单保持计划态, 不猜测节假日或制造执行结果",
                        entity_type="account",
                        entity_id=account["id"],
                    )
                summary["deferred"] = len(self.ledger.planned_orders())
                return summary
            for account in self.ledger.list_accounts():
                self.ledger.resolve_incident(
                    f"account:{account['id']}:TRADING_DAY_UNCONFIRMED:{trading_date}"
                )
            self._release_t1_waiting_exits(trading_date)
            accounts = {row["id"]: row for row in self.ledger.list_account_rows(active_only=True)}
            for order in self.ledger.planned_orders():
                if order["account_id"] not in accounts:
                    continue
                if date.fromisoformat(str(order["signal_date"])) >= trading_date:
                    continue
                if order["scheduled_date"] and order["scheduled_date"] != trading_date.isoformat():
                    continue
                summary["checked"] += 1
                self.ledger.assign_due_date(
                    order["id"],
                    trading_date,
                    {
                        "market_observed": True,
                        "checked_at": current.isoformat(timespec="seconds"),
                        "cash": float(accounts[order["account_id"]]["cash_balance"]),
                    },
                )
                summary["assigned"] += 1
        return summary

    def _release_t1_waiting_exits(self, trading_date: date) -> None:
        account_rows = {row["id"]: row for row in self.ledger.list_account_rows(active_only=True)}
        for position in self.ledger.position_rows():
            reason = position["pending_exit_reason"]
            if not reason or int(position["available_qty"]) <= 0:
                continue
            account = account_rows.get(position["account_id"])
            if account is None:
                continue
            config = json.loads(account["config_json"])
            self.ledger.record_signal_and_order(
                account_id=position["account_id"],
                strategy_id=str(config.get("strategy_id")),
                symbol=position["symbol"],
                name=position["name"],
                side="SELL",
                signal_date=date.fromisoformat(
                    str(position["pending_exit_date"] or position["acquired_on"])
                ),
                score=None,
                reason=str(reason),
                signal_ref=None,
                requested_qty=int(position["available_qty"]),
                target_amount=float(position["market_value"]),
                target_weight=0,
                planned_session="NEXT_OPEN",
            )

    def _reference_close(self, order: dict[str, Any], quote: dict[str, Any]) -> float | None:
        value = quote.get("prev_close")
        if _valid_price(value):
            return float(value)
        signal_date = date.fromisoformat(str(order["signal_date"]))
        account = self.ledger.get_account(order["account_id"])
        return self._signal_close(account["config"].get("asset_type", "stock"), order["symbol"], signal_date)

    def _blocked_status(
        self,
        order: dict[str, Any],
        quote: dict[str, Any],
        trading_date: date,
    ) -> tuple[str | None, str]:
        values = [quote.get(key) for key in ("open", "high", "low", "last_price")]
        volume = quote.get("volume")
        if not all(_valid_price(value) for value in values) or volume in (None, 0):
            return "REJECTED_SUSPENDED", "开盘行情无有效 OHLC 或成交量为零"
        previous = self._reference_close(order, quote)
        if previous is None:
            return "UNKNOWN_MARKET_DATA", "缺少昨收, 无法验证涨跌停价格"
        name = str(order["name"] or "")
        pct = price_limit_pct(
            order["symbol"], trading_date, is_risk_warning=is_risk_warning_name(name)
        )
        limit = _scalar_limit_price(previous, pct, up=order["side"] == "BUY")
        if is_one_price_locked(
            open_price=float(quote["open"]),
            high_price=float(quote["high"]),
            low_price=float(quote["low"]),
            close_price=float(quote["last_price"]),
            limit_price=limit,
        ):
            if order["side"] == "BUY":
                return "REJECTED_LIMIT_UP", f"一字涨停 {limit:.2f}, 无法买入"
            return "REJECTED_LIMIT_DOWN", f"一字跌停 {limit:.2f}, 无法卖出"
        return None, ""

    def execute_open_orders(
        self,
        *,
        now: datetime | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
        finalize_missing: bool = False,
        quality: str = "ON_TIME",
    ) -> dict[str, int]:
        current = _as_cn(now)
        trading_date = current.date()
        quotes = quotes if quotes is not None else self._quotes_from_cache()
        summary = {"filled": 0, "partial": 0, "rejected": 0, "unknown": 0, "waiting": 0}
        with self._lock:
            self.preflight_all(now=current, quotes=quotes)
            for row in self.ledger.planned_orders():
                order = dict(row)
                if order["status"] != "PREFLIGHT_OK" or order["scheduled_date"] != trading_date.isoformat():
                    continue
                quote = quotes.get(order["symbol"])
                if quote is None or quote.get("_quote_dt") is None or quote["_quote_dt"].date() != trading_date:
                    if finalize_missing:
                        self.ledger.terminal_order(
                            order["id"],
                            status="UNKNOWN_MARKET_DATA",
                            reason="09:31 前未取得该标的带当日时间戳的有效行情",
                            quality=quality,
                        )
                        summary["unknown"] += 1
                    else:
                        summary["waiting"] += 1
                    continue
                valid_open_bar = (
                    all(_valid_price(quote.get(key)) for key in ("open", "high", "low", "last_price"))
                    and quote.get("volume") not in (None, 0)
                )
                if not valid_open_bar and not finalize_missing:
                    summary["waiting"] += 1
                    continue
                blocked, reason = self._blocked_status(order, quote, trading_date)
                if blocked:
                    if blocked == "UNKNOWN_MARKET_DATA" and not finalize_missing:
                        summary["waiting"] += 1
                        continue
                    self.ledger.terminal_order(order["id"], status=blocked, reason=reason, quality=quality)
                    summary["unknown" if blocked == "UNKNOWN_MARKET_DATA" else "rejected"] += 1
                    continue
                price = float(quote.get("open") or quote["last_price"])
                requested = int(order["requested_qty"])
                account = self.ledger.get_account(order["account_id"])
                if order["side"] == "BUY":
                    model = _account_cost_model(account["config"])
                    affordable = round_lot_quantity(account["summary"]["cash"], price, model)
                    if requested <= 0:
                        requested = round_lot_quantity(float(order["target_amount"]), price, model)
                    quantity = min(requested, affordable)
                    if quantity <= 0:
                        self.ledger.terminal_order(
                            order["id"],
                            status="REJECTED_INSUFFICIENT_CASH",
                            reason="可用资金不足以买入一手",
                            quality=quality,
                        )
                        summary["rejected"] += 1
                        continue
                else:
                    positions = {p["symbol"]: p for p in account["positions"]}
                    position = positions.get(order["symbol"])
                    quantity = min(requested, int(position["available_qty"]) if position else 0)
                    if quantity <= 0:
                        self.ledger.terminal_order(
                            order["id"],
                            status="EXECUTION_FAILED",
                            reason="可卖数量为零或仍受 T+1 锁定",
                            quality=quality,
                        )
                        summary["rejected"] += 1
                        continue
                try:
                    self.ledger.execute_fill(
                        order["id"],
                        price=price,
                        quantity=quantity,
                        quote_at=quote["_quote_dt"],
                        source=str(quote.get("source") or "realtime"),
                        quality=quality,
                        previous_close=self._reference_close(order, quote),
                    )
                    if quantity < requested:
                        summary["partial"] += 1
                    else:
                        summary["filled"] += 1
                except Exception as exc:
                    logger.exception("paper order execution failed: %s", order["id"])
                    self.ledger.terminal_order(
                        order["id"], status="EXECUTION_FAILED", reason=str(exc), quality=quality
                    )
                    summary["rejected"] += 1
        return summary

    def finalize_open_window(self, *, now: datetime | None = None) -> dict[str, int]:
        return self.execute_open_orders(now=now, finalize_missing=True)

    def on_quote_records(self, records: list[dict[str, Any]], *, source: str) -> dict[str, int]:
        quotes = _quote_map(records, source=source)
        if not quotes:
            return {"marked": 0, "executed": 0, "risk_triggers": 0}
        current = max(quote["_quote_dt"] for quote in quotes.values())
        tracked = self.subscription_symbols()
        selected = {symbol: quote for symbol, quote in quotes.items() if symbol in tracked}
        marked = self.ledger.update_marks(selected, source=source) if selected else 0
        executed = 0
        if OPEN_START <= current.time() <= OPEN_DEADLINE:
            result = self.execute_open_orders(now=current, quotes=quotes, finalize_missing=False)
            executed = result["filled"] + result["partial"]
        risk_triggers = self._process_intraday_risk(current, quotes)
        return {"marked": marked, "executed": executed, "risk_triggers": risk_triggers}

    def _process_intraday_risk(
        self,
        current: datetime,
        quotes: dict[str, dict[str, Any]],
    ) -> int:
        self._execute_next_quote_orders(current, quotes)
        triggered = 0
        accounts = {item["id"]: item for item in self.ledger.list_accounts() if item["status"] == "active"}
        for position in self.ledger.position_rows():
            account = accounts.get(position["account_id"])
            quote = quotes.get(position["symbol"])
            if account is None or quote is None or account["config"].get("exit_mode", "eod") != "intraday":
                continue
            if position["pending_exit_reason"]:
                continue
            price = quote.get("last_price")
            if not _valid_price(price):
                continue
            entry = float(position["average_price"])
            overrides = account["config"].get("overrides") or {}
            stop_loss = overrides.get("stop_loss")
            take_profit = overrides.get("take_profit")
            reason = None
            if stop_loss is not None and float(price) <= entry * (1 - abs(float(stop_loss))):
                reason = "stop_loss"
            elif take_profit is not None and float(price) >= entry * (1 + abs(float(take_profit))):
                reason = "take_profit"
            if reason is None:
                continue
            locked = is_same_day_t_plus_one_locked(position["acquired_on"], current.date())
            self.ledger.mark_position_exit_triggered(
                position["account_id"], position["symbol"], reason=reason,
                trading_date=current.date(), locked=locked,
            )
            if not locked:
                self.ledger.record_signal_and_order(
                    account_id=position["account_id"],
                    strategy_id=account["config"]["strategy_id"],
                    symbol=position["symbol"],
                    name=position["name"],
                    side="SELL",
                    signal_date=current.date(),
                    score=None,
                    reason=reason,
                    signal_ref=None,
                    requested_qty=int(position["available_qty"]),
                    target_amount=float(position["market_value"]),
                    target_weight=0,
                    planned_session="NEXT_QUOTE",
                    payload={"trigger_quote_at": quote["quote_at"]},
                )
            triggered += 1
        return triggered

    def _execute_next_quote_orders(
        self,
        current: datetime,
        quotes: dict[str, dict[str, Any]],
    ) -> None:
        for row in self.ledger.planned_orders():
            order = dict(row)
            if order["planned_session"] != "NEXT_QUOTE" or order["status"] != "PLANNED":
                continue
            if date.fromisoformat(order["signal_date"]) != current.date():
                continue
            quote = quotes.get(order["symbol"])
            if quote is None:
                continue
            self.ledger.assign_due_date(order["id"], current.date(), {"next_quote": True})
            try:
                self.ledger.execute_fill(
                    order["id"],
                    price=float(quote["last_price"]),
                    quantity=int(order["requested_qty"]),
                    quote_at=quote["_quote_dt"],
                    source=str(quote.get("source") or "realtime"),
                    quality="ON_TIME",
                    previous_close=self._reference_close(order, quote),
                )
            except Exception as exc:
                self.ledger.terminal_order(order["id"], status="EXECUTION_FAILED", reason=str(exc))

    def _regime_allows(self, signal_date: date, config: dict[str, Any]) -> bool:
        regime_filter = config.get("regime_filter")
        if not regime_filter or (
            not regime_filter.get("states") and regime_filter.get("min_score") is None
        ):
            return True
        from app.services import regime_builder

        frame = regime_builder.load_regime_history(self.repo.store.data_dir)
        if frame.is_empty():
            raise ValueError("市场环境数据为空, 不能生成模拟交易订单")
        earlier = frame.filter(pl.col("date") < signal_date).sort("date")
        if earlier.is_empty():
            raise ValueError("缺少信号日前一交易日的市场环境")
        previous = earlier.row(-1, named=True)
        labels = (str(previous["date"]), signal_date.isoformat())
        mask = build_regime_filter_mask(
            labels,
            regime_filter,
            {previous["date"]: {"state": previous.get("state"), "score": previous.get("score")}},
            required_start=signal_date,
            required_end=signal_date,
        )
        return bool(mask is not None and mask[-1])

    @staticmethod
    def _risk_reason(position: dict[str, Any], row: dict[str, Any], account: dict[str, Any]) -> str | None:
        entry = float(position["average_price"])
        overrides = account["config"].get("overrides") or {}
        stop_loss = overrides.get("stop_loss")
        take_profit = overrides.get("take_profit")
        low = row.get("raw_low", row.get("low"))
        high = row.get("raw_high", row.get("high"))
        if stop_loss is not None and _valid_price(low) and float(low) <= entry * (1 - abs(float(stop_loss))):
            return "stop_loss"
        if take_profit is not None and _valid_price(high) and float(high) >= entry * (1 + abs(float(take_profit))):
            return "take_profit"
        max_hold = overrides.get("max_hold_days")
        if max_hold is not None and int(position["hold_days"]) >= int(max_hold):
            return "max_hold"
        return None

    def seal_account_signals(self, account_id: str, signal_date: date) -> dict[str, int]:
        account = self.ledger.get_account(account_id)
        if account["status"] != "active":
            return {"signals": 0, "orders": 0}
        if account.get("last_processed_date") == signal_date.isoformat():
            return {"signals": 0, "orders": 0}
        config = account["config"]
        if not self._regime_allows(signal_date, config):
            self.ledger.record_account_event(
                account_id,
                event_key=f"{account_id}:REGIME_BLOCKED:{signal_date}",
                event_type="REGIME_BLOCKED",
                trading_date=signal_date,
                title="市场环境过滤未通过",
                detail="该封板日不生成买入订单",
            )
            self.ledger.mark_signal_day(account_id, signal_date)
            return {"signals": 0, "orders": 0}
        engine = self.app_state.strategy_engine
        screener = ScreenerService(self.repo, asset_type=config.get("asset_type", "stock"))
        context = screener.build_strategy_context(
            engine,
            signal_date,
            [config["strategy_id"]],
            params_map={config["strategy_id"]: config.get("params") or {}},
            overrides_map={config["strategy_id"]: config.get("overrides") or {}},
        )
        result = engine.run(
            config["strategy_id"], context, pool=config.get("symbols") or None,
            params=config.get("params") or None, overrides=config.get("overrides") or None,
        )
        current_rows = {
            str(row["symbol"]): row
            for row in (context.current.to_dicts() if context.current is not None else [])
        }
        exit_hits = {str(item["symbol"]): item for item in result.exit_signal_hits}
        created = 0
        positions = account["positions"]
        for position in positions:
            reason = "strategy_exit" if position["symbol"] in exit_hits else None
            if reason is None and config.get("exit_mode", "eod") == "eod":
                row = current_rows.get(position["symbol"])
                if row:
                    reason = self._risk_reason(position, row, account)
            if reason:
                self.ledger.mark_position_exit_triggered(
                    account_id, position["symbol"], reason=reason,
                    trading_date=signal_date,
                    locked=is_same_day_t_plus_one_locked(
                        position["acquired_on"], signal_date
                    ),
                )
                _, _, was_created = self.ledger.record_signal_and_order(
                    account_id=account_id,
                    strategy_id=config["strategy_id"],
                    symbol=position["symbol"],
                    name=position["name"],
                    side="SELL",
                    signal_date=signal_date,
                    score=None,
                    reason=reason,
                    signal_ref=(exit_hits.get(position["symbol"], {}).get("signals") or [None])[0],
                    requested_qty=int(position["quantity"]),
                    target_amount=float(position["market_value"]),
                    target_weight=0,
                    planned_session="NEXT_OPEN",
                )
                created += int(was_created)
        held = {item["symbol"] for item in positions}
        open_buy_orders = {
            item["symbol"] for item in account["orders"]
            if item["side"] == "BUY" and item["status"] not in TERMINAL_ORDER_STATUSES
        }
        slots = max(int(config.get("max_positions", 10)) - len(held) - len(open_buy_orders), 0)
        candidates = [
            row for row in result.rows
            if str(row.get("symbol")) not in held | open_buy_orders
        ][:slots]
        if not candidates:
            self.ledger.record_account_event(
                account_id,
                event_key=f"{account_id}:SIGNAL_SEAL_EMPTY:{signal_date}",
                event_type="SIGNAL_SEAL_COMPLETED",
                trading_date=signal_date,
                title="封板信号已完成",
                detail="当日没有新的可执行买入候选",
            )
            self.ledger.mark_signal_day(account_id, signal_date)
            return {"signals": created, "orders": created}
        equity = float(account["summary"]["equity"])
        cash = float(account["summary"]["cash"])
        max_exposure = float(config.get("max_exposure_pct", 1.0))
        capacity = max(equity * max_exposure - float(account["summary"]["market_value"]), 0)
        budget = min(cash, capacity)
        scores = [max(float(row.get("score") or 0), 0) for row in candidates]
        if config.get("position_sizing") == "score_weight" and sum(scores) > 0:
            weights = [score / sum(scores) for score in scores]
        else:
            weights = [1 / len(candidates)] * len(candidates)
        model = _account_cost_model(config)
        for row, weight in zip(candidates, weights, strict=True):
            price = row.get("raw_close") or row.get("close")
            allocation = budget * weight
            quantity = round_lot_quantity(allocation, float(price or 0), model)
            if quantity <= 0:
                continue
            symbol = str(row["symbol"])
            hit = next((item for item in result.entry_signal_hits if item["symbol"] == symbol), {})
            signal_ref = (hit.get("signals") or [None])[0]
            _, _, was_created = self.ledger.record_signal_and_order(
                account_id=account_id,
                strategy_id=config["strategy_id"],
                symbol=symbol,
                name=str(row.get("name") or symbol),
                side="BUY",
                signal_date=signal_date,
                score=float(row.get("score") or 0),
                reason="strategy_entry",
                signal_ref=signal_ref,
                requested_qty=quantity,
                target_amount=allocation,
                target_weight=max_exposure * weight,
                planned_session="NEXT_OPEN",
                payload={"reference_close": float(price)},
            )
            created += int(was_created)
        self.ledger.mark_signal_day(account_id, signal_date)
        return {"signals": created, "orders": created}

    def seal_daily_signals(self, signal_date: date | None = None) -> dict[str, int]:
        summary = {"processed": 0, "failed": 0, "orders": 0}
        with self._lock:
            for row in self.ledger.list_account_rows(active_only=True):
                config = json.loads(row["config_json"])
                latest = signal_date or self.repo.latest_enriched_date(config.get("asset_type", "stock"))
                if latest is None:
                    summary["failed"] += 1
                    continue
                try:
                    result = self.seal_account_signals(row["id"], latest)
                    summary["processed"] += 1
                    summary["orders"] += result["orders"]
                except Exception as exc:
                    summary["failed"] += 1
                    self.ledger.open_incident(
                        account_id=row["id"],
                        incident_key=f"account:{row['id']}:SIGNAL_FAILURE:{latest}",
                        code="SIGNAL_FAILURE",
                        severity="critical",
                        title="封板信号生成失败",
                        detail=str(exc),
                        entity_type="account",
                        entity_id=row["id"],
                    )
                    logger.exception("paper signal seal failed: %s", row["id"])
        return summary

    def settle_all(self, *, trading_date: date | None = None) -> dict[str, int]:
        current_date = trading_date or _as_cn().date()
        summary = {"settled": 0, "failed": 0, "imbalanced": 0}
        with self._lock:
            for row in self.ledger.list_account_rows():
                try:
                    self.ledger.settle_account(row["id"], current_date, source="15:05_close")
                    check = self.ledger.reconcile(row["id"])
                    summary["settled"] += 1
                    summary["imbalanced"] += int(not check["ok"])
                except Exception:
                    summary["failed"] += 1
                    logger.exception("paper settlement failed: %s", row["id"])
        return summary

    def recover_missed_open(self, *, now: datetime | None = None) -> dict[str, int]:
        current = _as_cn(now)
        result = {"missed": 0, "recovered": 0, "unknown": 0}
        if current.time() < OPEN_DEADLINE:
            return result
        with self._lock:
            quotes = self._quotes_from_cache()
            symbols = sorted(self.subscription_symbols())
            if not quotes and symbols:
                quote_service = getattr(self.app_state, "quote_service", None)
                if quote_service is not None:
                    try:
                        refresh = getattr(
                            quote_service, "refresh_paper_symbols", quote_service.refresh
                        )
                        fetched = refresh()
                        records = fetched.get("records", []) if isinstance(fetched, dict) else []
                        quotes = _quote_map(records, source="realtime_recovery")
                    except Exception:
                        logger.exception("paper recovery quote refresh failed")
                    if not quotes:
                        quotes = self._quotes_from_cache()
            minutes = self.repo.get_minute_batch(symbols, current.date()) if symbols else pl.DataFrame()
            minute_by_symbol: dict[str, dict[str, Any]] = {}
            if not minutes.is_empty() and "datetime" in minutes.columns:
                minute_time = pl.col("datetime")
                minute_dtype = minutes.schema.get("datetime")
                if isinstance(minute_dtype, pl.Datetime) and minute_dtype.time_zone:
                    minute_time = minute_time.dt.convert_time_zone("Asia/Shanghai")
                valid = minutes.filter(
                    (minute_time.dt.time() >= OPEN_START)
                    & (minute_time.dt.time() <= OPEN_DEADLINE)
                ).sort(["symbol", "datetime"])
                for symbol, frame in valid.group_by("symbol", maintain_order=True):
                    key = symbol[0] if isinstance(symbol, tuple) else symbol
                    if not frame.is_empty():
                        minute_by_symbol[str(key)] = frame.row(0, named=True)
            market_observed = self._market_is_observed(current.date(), quotes) or bool(minute_by_symbol)
            if not market_observed:
                return result
            recovery_quotes = quotes or {
                symbol: {
                    "symbol": symbol,
                    "last_price": row.get("close"),
                    "open": row.get("open"), "high": row.get("high"), "low": row.get("low"),
                    "volume": row.get("volume"),
                    "quote_at": _as_cn(row["datetime"]).isoformat(timespec="seconds"),
                    "_quote_dt": _as_cn(row["datetime"]), "source": "minute_k",
                }
                for symbol, row in minute_by_symbol.items()
            }
            self.preflight_all(now=current, quotes=recovery_quotes)
            for row in self.ledger.planned_orders():
                order = dict(row)
                if order["status"] != "PREFLIGHT_OK" or order["scheduled_date"] != current.date().isoformat():
                    continue
                self.ledger.terminal_order(
                    order["id"],
                    status="MISSED_EXECUTION",
                    reason="服务未在 09:30 执行该订单, 已按真实发生时间记录错过开盘",
                    quality="MISSED_EXECUTION",
                    severity="critical",
                )
                result["missed"] += 1
                minute = minute_by_symbol.get(order["symbol"])
                if minute is None or not all(_valid_price(minute.get(key)) for key in ("open", "high", "low", "close")):
                    self.ledger.terminal_order(
                        order["id"], status="UNKNOWN_MARKET_DATA",
                        reason="缺少 09:30-09:31 的完整可靠行情, 不制造成交",
                        quality="NO_RELIABLE_OPEN_DATA",
                    )
                    result["unknown"] += 1
                    continue
                quote = {
                    **minute,
                    "last_price": minute["close"],
                    "quote_at": _as_cn(minute["datetime"]).isoformat(timespec="seconds"),
                    "_quote_dt": _as_cn(minute["datetime"]),
                    "source": "minute_k_recovery",
                }
                blocked, reason = self._blocked_status(order, quote, current.date())
                if blocked:
                    self.ledger.terminal_order(order["id"], status=blocked, reason=reason, quality="RECOVERED_LATE")
                    continue
                account = self.ledger.get_account(order["account_id"])
                price = float(minute["open"])
                quantity = int(order["requested_qty"])
                if order["side"] == "BUY":
                    affordable = round_lot_quantity(
                        account["summary"]["cash"], price, _account_cost_model(account["config"])
                    )
                    quantity = min(quantity or affordable, affordable)
                if quantity <= 0:
                    self.ledger.terminal_order(
                        order["id"], status="REJECTED_INSUFFICIENT_CASH",
                        reason="恢复撮合时可用资金不足", quality="RECOVERED_LATE",
                    )
                    continue
                try:
                    self.ledger.execute_fill(
                        order["id"], price=price, quantity=quantity,
                        quote_at=quote["_quote_dt"], source="minute_k_recovery",
                        quality="RECOVERED_LATE",
                        previous_close=self._reference_close(order, quote),
                    )
                    result["recovered"] += 1
                except Exception as exc:
                    self.ledger.terminal_order(
                        order["id"], status="EXECUTION_FAILED", reason=str(exc),
                        quality="RECOVERED_LATE",
                    )
        return result

    def sync_account(self, account_id: str) -> dict[str, Any]:
        self.ledger.account_row(account_id)
        self.recover_missed_open()
        account = self.ledger.get_account(account_id)
        latest = self.repo.latest_enriched_date(account["config"].get("asset_type", "stock"))
        now = _as_cn()
        if latest and (latest < now.date() or now.time() >= SIGNAL_SEAL_TIME):
            self.seal_account_signals(account_id, latest)
        self.ledger.reconcile(account_id)
        return self.account(account_id)

    def account(self, account_id: str) -> dict[str, Any]:
        account = self.ledger.get_account(account_id)
        account["system"] = self.system_status()
        account["reconciliation"] = self.ledger.reconcile(account_id, open_incident=False)
        return account

    def accounts(self) -> list[dict[str, Any]]:
        system = self.system_status()
        items = self.ledger.list_accounts()
        for account in items:
            account["system"] = system
            account["reconciliation"] = self.ledger.reconcile(account["id"], open_incident=False)
        return items

    def system_status(self) -> dict[str, Any]:
        now = _as_cn()
        quote_service = getattr(self.app_state, "quote_service", None)
        quote_status = quote_service.status() if quote_service is not None else {}
        t = now.time()
        if now.weekday() >= 5 or t < dt_time(8, 30) or t >= dt_time(16, 0):
            phase = "CLOSED"
        elif t < dt_time(9, 25):
            phase = "PRE_MARKET"
        elif t < OPEN_START:
            phase = "PREFLIGHT"
        elif t <= dt_time(11, 30):
            phase = "TRADING"
        elif t < dt_time(13, 0):
            phase = "LUNCH_BREAK"
        elif t <= dt_time(15, 0):
            phase = "TRADING"
        elif t < SETTLEMENT_TIME:
            phase = "CLOSE_PENDING"
        elif t < SIGNAL_SEAL_TIME:
            phase = "SETTLEMENT"
        else:
            phase = "SIGNAL_SEAL"
        accounts = self.ledger.list_accounts()
        incidents = sum(account["summary"]["open_incident_count"] for account in accounts)
        critical_incidents = sum(
            account["summary"]["critical_incident_count"] for account in accounts
        )
        tracked_symbols = len(self.subscription_symbols())
        quote_age = quote_status.get("quote_age_ms")
        stale = bool(
            tracked_symbols
            and phase == "TRADING"
            and (quote_age is None or float(quote_age) > QUOTE_STALE_SECONDS * 1000)
        )
        health = (
            "ERROR" if critical_incidents
            else "DEGRADED" if incidents or (tracked_symbols and stale and phase == "TRADING")
            else "HEALTHY"
        )
        return {
            "beijing_time": now.isoformat(timespec="seconds"),
            "market_phase": phase,
            "quote_age_ms": quote_age,
            "quote_stale": stale,
            "quote_source_mode": quote_status.get("mode", "none"),
            "quote_enabled": bool(quote_status.get("enabled")),
            "executor_health": health,
            "tracked_symbol_count": tracked_symbols,
            "open_incident_count": incidents,
            "critical_incident_count": critical_incidents,
            "ledger_path": self.ledger.path.name,
        }


def get_service(app_state) -> PaperTradingService:
    service = getattr(app_state, "paper_trading_service", None)
    if service is None:
        service = PaperTradingService(app_state)
        app_state.paper_trading_service = service
    return service


def run_account(app_state, account_id: str, *, force: bool = False) -> dict[str, Any]:
    del force
    return get_service(app_state).sync_account(account_id)


def run_active_accounts(app_state) -> dict[str, int]:
    """Compatibility name: the after-close phase now freezes one day of signals."""
    return get_service(app_state).seal_daily_signals()
