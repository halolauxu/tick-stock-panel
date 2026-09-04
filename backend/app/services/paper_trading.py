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
from app.market_time import CN_TZ, cn_now, is_possible_cn_equity_session
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
PAPER_STOCK_SUPPORTED_BOARDS = ("沪主板", "深主板")


def is_supported_paper_stock_symbol(symbol: str) -> bool:
    """Paper stock accounts are restricted to the Shanghai/Shenzhen main boards."""
    normalized = str(symbol or "").strip().upper()
    code, _, suffix = normalized.partition(".")
    if suffix == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if suffix == "SZ":
        return code.startswith(("000", "001", "002", "003"))
    if suffix:
        return False
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


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
        self._recovery_fetch_last: dict[tuple[date, str], datetime] = {}
        self._recovery_quote_fetch_last: datetime | None = None
        self._close_quote_refresh_last: datetime | None = None
        self._migrate_legacy_json()
        repaired = self._repair_invalid_session_orders()
        if repaired:
            logger.warning("requeued %d paper orders scheduled on non-session dates", repaired)

    def _repair_invalid_session_orders(self) -> int:
        """Requeue safe, unfilled orders that were scheduled on a weekend."""
        candidates = {
            str(row["id"]): dict(row)
            for row in [*self.ledger.planned_orders(), *self.ledger.recovery_orders()]
        }
        repaired = 0
        for order in candidates.values():
            scheduled = order.get("scheduled_date")
            if not scheduled:
                continue
            trading_date = date.fromisoformat(str(scheduled))
            if is_possible_cn_equity_session(trading_date):
                continue
            repaired += int(
                self.ledger.requeue_invalid_session_order(order["id"], trading_date)
            )
        return repaired

    def claim_close_quote_refresh(
        self,
        *,
        now: datetime | None = None,
        min_interval_seconds: int = 300,
    ) -> bool:
        """Throttle the independent post-close paper quote finalizer."""
        current = _as_cn(now)
        with self._lock:
            previous = self._close_quote_refresh_last
            if (
                previous is not None
                and previous.date() == current.date()
                and (current - previous).total_seconds() < min_interval_seconds
            ):
                return False
            self._close_quote_refresh_last = current
            return True

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
            # Repository cache reconciliation always needs ``date`` even when
            # the consumer only reads prices from the returned row.
            ["date", "raw_close", "close"],
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

    @staticmethod
    def quotes_from_records(
        records: list[dict[str, Any]], *, source: str = "realtime"
    ) -> dict[str, dict[str, Any]]:
        """Normalize one provider response for direct use by the execution clock."""
        return _quote_map(records, source=source)

    def _quotes_from_cache(self) -> dict[str, dict[str, Any]]:
        service = getattr(self.app_state, "quote_service", None)
        if service is None:
            return {}
        frame, frame_date = service.get_enriched_today()
        if frame is None or frame.is_empty() or frame_date is None:
            return {}
        records = frame.to_dicts()
        # Never replace a missing market timestamp with the wall-clock fetch
        # time.  On weekends and exchange holidays that turns a stale previous
        # session close into apparently current execution evidence.
        return _quote_map(records, source="realtime_cache")

    @staticmethod
    def _market_is_observed(trading_date: date, quotes: dict[str, dict[str, Any]]) -> bool:
        return is_possible_cn_equity_session(trading_date) and any(
            quote.get("_quote_dt") is not None
            and quote["_quote_dt"].date() == trading_date
            and _valid_price(quote.get("last_price"))
            for quote in quotes.values()
        )

    @staticmethod
    def _quote_ready_for_preflight(
        quote: dict[str, Any] | None, trading_date: date
    ) -> bool:
        return bool(
            is_possible_cn_equity_session(trading_date)
            and quote
            and quote.get("_quote_dt") is not None
            and quote["_quote_dt"].date() == trading_date
            and _valid_price(quote.get("last_price"))
            and _valid_price(quote.get("prev_close"))
        )

    def probe_quote_chain(
        self,
        *,
        now: datetime | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """09:20 canary: prove the next-open execution data path before preflight."""
        current = _as_cn(now)
        trading_date = current.date()
        quotes = quotes if quotes is not None else self._quotes_from_cache()
        tracked = self.subscription_symbols()
        required = set(tracked)
        if tracked:
            required.add("000001.SH")
        missing = sorted(
            symbol
            for symbol in required
            if not self._quote_ready_for_preflight(quotes.get(symbol), trading_date)
        )
        ready = not missing
        accounts = [
            item for item in self.ledger.list_accounts() if item["status"] == "active"
        ]
        for account in accounts:
            incident_key = f"account:{account['id']}:QUOTE_CHAIN_NOT_READY:{trading_date}"
            if ready:
                self.ledger.resolve_incident(incident_key)
            else:
                self.ledger.open_incident(
                    account_id=account["id"],
                    incident_key=incident_key,
                    code="QUOTE_CHAIN_NOT_READY",
                    severity="critical",
                    title="开盘行情链路未就绪",
                    detail=f"09:20 探针未取得当日有效行情: {', '.join(missing)}",
                    entity_type="account",
                    entity_id=account["id"],
                )
            self.ledger.record_account_event(
                account["id"],
                event_key=(
                    f"{account['id']}:QUOTE_CHAIN:"
                    f"{'READY' if ready else 'NOT_READY'}:{trading_date}"
                ),
                event_type="QUOTE_CHAIN_READY" if ready else "QUOTE_CHAIN_NOT_READY",
                trading_date=trading_date,
                severity="info" if ready else "critical",
                title="开盘行情链路已就绪" if ready else "开盘行情链路未就绪",
                detail=(
                    f"已验证 {len(required)} 个执行/市场探针标的"
                    if ready
                    else f"缺少当日有效行情: {', '.join(missing)}"
                ),
                payload={"required": sorted(required), "missing": missing},
            )
        return {
            "ready": ready,
            "required": len(required),
            "available": len(required) - len(missing),
            "missing": missing,
        }

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
            if not is_possible_cn_equity_session(trading_date):
                summary["deferred"] = len(self.ledger.planned_orders())
                return summary
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
            self.ledger.unlock_positions(trading_date)
            for account in self.ledger.list_accounts():
                self.ledger.resolve_incident(
                    f"account:{account['id']}:TRADING_DAY_UNCONFIRMED:{trading_date}"
                )
            tracked_ready = all(
                self._quote_ready_for_preflight(quotes.get(symbol), trading_date)
                for symbol in self.subscription_symbols()
            )
            if tracked_ready:
                for account in self.ledger.list_accounts():
                    self.ledger.resolve_incident(
                        f"account:{account['id']}:QUOTE_CHAIN_NOT_READY:{trading_date}"
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
                account_config = json.loads(accounts[order["account_id"]]["config_json"])
                if (
                    order["side"] == "BUY"
                    and account_config.get("asset_type", "stock") == "stock"
                    and not is_supported_paper_stock_symbol(str(order["symbol"]))
                ):
                    self.ledger.terminal_order(
                        order["id"],
                        status="REJECTED_UNSUPPORTED_BOARD",
                        reason="账户仅允许买入沪深主板,已拒绝创业板、科创板或其他板块订单",
                        quality="POLICY_REJECTED",
                        scheduled_date=trading_date,
                    )
                    continue
                quote = quotes.get(str(order["symbol"]))
                quote_incident = (
                    f"order:{order['id']}:ORDER_QUOTE_NOT_READY:{trading_date}"
                )
                if not self._quote_ready_for_preflight(quote, trading_date):
                    self.ledger.open_incident(
                        account_id=order["account_id"],
                        incident_key=quote_incident,
                        code="ORDER_QUOTE_NOT_READY",
                        severity="warning",
                        title=f"{order['name'] or order['symbol']} 行情未就绪",
                        detail="盘前未取得该标的当日价格和昨收, 订单保持计划态",
                        entity_type="order",
                        entity_id=order["id"],
                    )
                    summary["deferred"] += 1
                    continue
                self.ledger.resolve_incident(quote_incident)
                self.ledger.assign_due_date(
                    order["id"],
                    trading_date,
                    {
                        "market_observed": True,
                        "checked_at": current.isoformat(timespec="seconds"),
                        "cash": float(accounts[order["account_id"]]["cash_balance"]),
                        "quote_at": quote["quote_at"],
                        "quote_source": quote.get("source"),
                        "previous_close": quote.get("prev_close"),
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
        if not is_possible_cn_equity_session(trading_date):
            summary["waiting"] = len(self.ledger.planned_orders())
            return summary
        with self._lock:
            self.preflight_all(now=current, quotes=quotes)
            market_observed = self._market_is_observed(trading_date, quotes)
            planned_orders = sorted(
                self.ledger.planned_orders(),
                key=lambda row: (
                    0 if row["side"] == "SELL" else 1,
                    row["created_at"],
                    row["id"],
                ),
            )
            for row in planned_orders:
                order = dict(row)
                eligible_today = (
                    date.fromisoformat(str(order["signal_date"])) < trading_date
                    and (
                        not order["scheduled_date"]
                        or order["scheduled_date"] == trading_date.isoformat()
                    )
                )
                if (
                    finalize_missing
                    and market_observed
                    and order["status"] == "PLANNED"
                    and eligible_today
                ):
                    self.ledger.resolve_incident(
                        f"order:{order['id']}:ORDER_QUOTE_NOT_READY:{trading_date}"
                    )
                    self.ledger.terminal_order(
                        order["id"],
                        status="UNKNOWN_MARKET_DATA",
                        reason="09:31 前未取得该标的当日价格和昨收, 盘前校验未通过",
                        quality="NO_RELIABLE_OPEN_DATA",
                        scheduled_date=trading_date,
                    )
                    summary["unknown"] += 1
                    continue
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

    def finalize_open_window(
        self,
        *,
        now: datetime | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        return self.execute_open_orders(
            now=now,
            quotes=quotes,
            finalize_missing=True,
        )

    def on_quote_records(self, records: list[dict[str, Any]], *, source: str) -> dict[str, int]:
        quotes = self.quotes_from_records(records, source=source)
        if not quotes:
            return {"marked": 0, "executed": 0, "risk_triggers": 0}
        current = max(quote["_quote_dt"] for quote in quotes.values())
        tracked = self.subscription_symbols()
        selected = {symbol: quote for symbol, quote in quotes.items() if symbol in tracked}
        marked = self.ledger.update_marks(selected, source=source) if selected else 0
        observed = _as_cn()
        if not is_possible_cn_equity_session(current.date()) or (
            source == "realtime" and current.date() != observed.date()
        ):
            return {"marked": marked, "executed": 0, "risk_triggers": 0}
        if marked and observed.time() >= SETTLEMENT_TIME:
            close_symbols = {
                symbol for symbol, quote in selected.items()
                if quote["_quote_dt"].date() == observed.date()
            }
            for account_id in sorted(
                self.ledger.settled_accounts_holding_symbols(
                    observed.date(),
                    close_symbols,
                )
            ):
                self.ledger.settle_account(
                    account_id,
                    observed.date(),
                    source="15:05_close",
                    restatement=True,
                )
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
        from app.services import risk_admitted_forecast_paper

        if risk_admitted_forecast_paper.is_managed_account(config):
            return risk_admitted_forecast_paper.seal_account(
                self, account_id, signal_date
            )
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
        signals_created = 0
        orders_created = 0
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
                signals_created += int(was_created)
                orders_created += int(was_created)
        held = {item["symbol"] for item in positions}
        open_buy_orders = {
            item["symbol"] for item in account["orders"]
            if item["side"] == "BUY" and item["status"] not in TERMINAL_ORDER_STATUSES
        }
        slots = max(int(config.get("max_positions", 10)) - len(held) - len(open_buy_orders), 0)
        strategy_rows = list(result.rows)
        unsupported_rows = (
            [
                row for row in strategy_rows
                if not is_supported_paper_stock_symbol(str(row.get("symbol")))
            ]
            if config.get("asset_type", "stock") == "stock"
            else []
        )
        unsupported_symbols = {str(row.get("symbol")) for row in unsupported_rows}
        supported_rows = [
            row for row in strategy_rows
            if str(row.get("symbol")) not in unsupported_symbols
        ]
        held_rows = [row for row in supported_rows if str(row.get("symbol")) in held]
        pending_rows = [
            row for row in supported_rows
            if str(row.get("symbol")) in open_buy_orders
            and str(row.get("symbol")) not in held
        ]
        executable_rows = [
            row for row in supported_rows
            if str(row.get("symbol")) not in held | open_buy_orders
        ]
        candidates = executable_rows[:slots]
        funnel = {
            "strategy_hits": len(strategy_rows),
            "unsupported_board": len(unsupported_rows),
            "already_held": len(held_rows),
            "already_pending": len(pending_rows),
            "available_slots": slots,
            "slot_limited": max(len(executable_rows) - slots, 0),
            "final_candidates": len(candidates),
        }
        funnel_detail = (
            f"策略已运行: 命中 {funnel['strategy_hits']} 只; "
            f"非沪深主板 {funnel['unsupported_board']} 只; "
            f"已持仓 {funnel['already_held']} 只; "
            f"已有待执行订单 {funnel['already_pending']} 只; "
            f"槽位截断 {funnel['slot_limited']} 只; "
            f"最终形成 {funnel['final_candidates']} 只新买候选"
        )
        if not candidates:
            self.ledger.record_account_event(
                account_id,
                event_key=f"{account_id}:SIGNAL_SEAL_EMPTY:{signal_date}",
                event_type="SIGNAL_SEAL_COMPLETED",
                trading_date=signal_date,
                title="选股已运行 · 未生成新订单",
                detail=funnel_detail,
                payload=funnel,
            )
            self.ledger.mark_signal_day(account_id, signal_date)
            return {"signals": signals_created, "orders": orders_created}
        equity = float(account["summary"]["equity"])
        cash = float(account["summary"]["cash"])
        max_exposure = float(config.get("max_exposure_pct", 1.0))
        capacity = max(equity * max_exposure - float(account["summary"]["market_value"]), 0)
        budget = min(cash, capacity)
        scores = [max(float(row.get("score") or 0), 0) for row in candidates]
        if config.get("position_sizing") == "score_weight" and sum(scores) > 0:
            weights = [score / sum(scores) for score in scores]
            allocations = [budget * weight for weight in weights]
            target_weights = [max_exposure * weight for weight in weights]
        else:
            per_position_target = equity * max_exposure / max(
                int(config.get("max_positions", 10)), 1
            )
            remaining_budget = budget
            allocations = []
            for _row in candidates:
                allocation = min(per_position_target, remaining_budget)
                allocations.append(allocation)
                remaining_budget = max(remaining_budget - allocation, 0)
            target_weights = [
                allocation / equity if equity > 0 else 0 for allocation in allocations
            ]
        model = _account_cost_model(config)
        buy_orders_before = orders_created
        for row, allocation, target_weight in zip(
            candidates, allocations, target_weights, strict=True
        ):
            price = row.get("raw_close") or row.get("close")
            quantity = round_lot_quantity(allocation, float(price or 0), model)
            symbol = str(row["symbol"])
            hit = next((item for item in result.entry_signal_hits if item["symbol"] == symbol), {})
            signal_ref = (hit.get("signals") or [None])[0]
            if quantity <= 0:
                valid_price = _valid_price(price)
                required_cash = (
                    float(price) * 100 * (1 + model.buy_cost_pct()) if valid_price else None
                )
                skip_code = (
                    "INSUFFICIENT_BUYING_POWER" if valid_price else "INVALID_REFERENCE_PRICE"
                )
                detail = (
                    f"可分配 {allocation:.2f} 元, 按参考价 {float(price):.2f} 元不足买入一手"
                    if valid_price
                    else "缺少有效参考价, 无法计算可执行数量"
                )
                _, signal_created = self.ledger.record_skipped_signal(
                    account_id=account_id,
                    strategy_id=config["strategy_id"],
                    symbol=symbol,
                    name=str(row.get("name") or symbol),
                    side="BUY",
                    signal_date=signal_date,
                    score=float(row.get("score") or 0),
                    reason="strategy_entry",
                    signal_ref=signal_ref,
                    skip_code=skip_code,
                    detail=detail,
                    payload={
                        "reference_close": float(price) if valid_price else None,
                        "allocated_cash": allocation,
                        "required_one_lot_cash": required_cash,
                        "available_cash": cash,
                        "remaining_capacity": capacity,
                    },
                )
                signals_created += int(signal_created)
                continue
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
                target_weight=target_weight,
                planned_session="NEXT_OPEN",
                payload={"reference_close": float(price)},
            )
            signals_created += int(was_created)
            orders_created += int(was_created)
        buy_orders_created = orders_created - buy_orders_before
        self.ledger.record_account_event(
            account_id,
            event_key=f"{account_id}:SIGNAL_SEAL_COMPLETED:{signal_date}",
            event_type="SIGNAL_SEAL_COMPLETED",
            trading_date=signal_date,
            title=(
                "选股已运行 · 次日订单已生成"
                if buy_orders_created
                else "选股已运行 · 信号未形成订单"
            ),
            detail=funnel_detail,
            payload={**funnel, "orders_created": buy_orders_created},
        )
        self.ledger.mark_signal_day(account_id, signal_date)
        return {"signals": signals_created, "orders": orders_created}

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

    @staticmethod
    def _opening_minutes(
        frame: pl.DataFrame,
        trading_date: date,
    ) -> dict[str, dict[str, Any]]:
        if frame.is_empty() or "datetime" not in frame.columns or "symbol" not in frame.columns:
            return {}
        minute_time = pl.col("datetime")
        minute_dtype = frame.schema.get("datetime")
        if isinstance(minute_dtype, pl.Datetime) and minute_dtype.time_zone:
            minute_time = minute_time.dt.convert_time_zone("Asia/Shanghai")
        valid = frame.filter(
            (minute_time.dt.time() >= OPEN_START)
            & (minute_time.dt.time() <= OPEN_DEADLINE)
        ).sort(["symbol", "datetime"])
        result: dict[str, dict[str, Any]] = {}
        for symbol, symbol_frame in valid.group_by("symbol", maintain_order=True):
            key = symbol[0] if isinstance(symbol, tuple) else symbol
            for row in symbol_frame.iter_rows(named=True):
                if _as_cn(row["datetime"]).date() == trading_date:
                    result[str(key)] = row
                    break
        return result

    def _remote_opening_minute(
        self,
        order: dict[str, Any],
        trading_date: date,
        current: datetime,
    ) -> dict[str, Any] | None:
        """Fetch only one missing execution symbol, with a five-minute retry backoff."""
        capset = getattr(self.app_state, "capabilities", None)
        if capset is None:
            return None
        try:
            from app.tickflow.capabilities import Cap

            if not capset.has(Cap.KLINE_MINUTE_BATCH):
                return None
        except Exception:
            return None
        key = (trading_date, str(order["symbol"]))
        last_attempt = self._recovery_fetch_last.get(key)
        if last_attempt is not None and (current - last_attempt).total_seconds() < 300:
            return None
        self._recovery_fetch_last[key] = current
        try:
            from app.services.kline_sync import fetch_minute_single

            account = self.ledger.get_account(order["account_id"])
            frame = fetch_minute_single(
                str(order["symbol"]),
                trading_date,
                asset_type=str(account["config"].get("asset_type", "stock")),
            )
            minute = self._opening_minutes(frame, trading_date).get(str(order["symbol"]))
            if minute is not None:
                minute["_recovery_source"] = "minute_k_targeted_recovery"
            return minute
        except Exception:
            logger.exception(
                "targeted paper recovery minute fetch failed: %s %s",
                order["symbol"],
                trading_date,
            )
            return None

    def _first_observed_session_after(
        self,
        signal_date: date,
        before_date: date,
    ) -> date | None:
        """Use persisted market partitions to recover the intended session after downtime."""
        root = Path(self.repo.store.data_dir) / "kline_daily"
        observed: list[date] = []
        if not root.exists():
            return None
        for partition in root.glob("date=*"):
            if not any(partition.glob("*.parquet")):
                continue
            try:
                candidate = date.fromisoformat(partition.name.removeprefix("date="))
            except ValueError:
                continue
            if signal_date < candidate < before_date:
                observed.append(candidate)
        return min(observed) if observed else None

    def _completed_day_opening_evidence(
        self,
        order: dict[str, Any],
        trading_date: date,
        current: datetime,
        quotes: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Fall back to completed-day OHLCV, matching the backtest open_t+1 contract."""
        if trading_date == current.date():
            quote = quotes.get(str(order["symbol"]))
            quote_at = quote.get("_quote_dt") if quote else None
            if (
                current.time() < dt_time(15, 0)
                or quote_at is None
                or quote_at.date() != trading_date
                or quote_at.time() < dt_time(15, 0)
            ):
                return None
            values = {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("last_price"),
                "volume": quote.get("volume"),
            }
            if not all(_valid_price(values[key]) for key in ("open", "high", "low", "close")):
                return None
            if values["volume"] in (None, 0):
                return None
            return {
                "symbol": str(order["symbol"]),
                "datetime": quote_at,
                **values,
                "prev_close": quote.get("prev_close"),
                "_recovery_source": "realtime_close_snapshot_open_recovery",
            }

        account = self.ledger.get_account(order["account_id"])
        frame = self.repo.get_daily_asset(
            str(account["config"].get("asset_type", "stock")),
            str(order["symbol"]),
            trading_date,
            trading_date,
            [
                "date", "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume",
                "open", "high", "low", "close", "volume", "prev_close", "quote_ts",
            ],
        )
        if frame.is_empty():
            return None
        row = frame.row(-1, named=True)
        row_date = row.get("date")
        if row_date is not None and str(row_date)[:10] != trading_date.isoformat():
            return None
        live_marker = _quote_datetime(row.get("quote_ts"))
        if live_marker is not None and (
            live_marker.date() != trading_date or live_marker.time() < dt_time(15, 0)
        ):
            return None

        def daily_value(key: str) -> Any:
            raw = row.get(f"raw_{key}")
            return raw if raw is not None else row.get(key)

        values = {
            "open": daily_value("open"),
            "high": daily_value("high"),
            "low": daily_value("low"),
            "close": daily_value("close"),
            "volume": daily_value("volume"),
        }
        if not all(_valid_price(values[key]) for key in ("open", "high", "low", "close")):
            return None
        if values["volume"] in (None, 0):
            return None
        return {
            "symbol": str(order["symbol"]),
            "datetime": datetime.combine(trading_date, dt_time(15, 0), tzinfo=CN_TZ),
            **values,
            "prev_close": row.get("prev_close"),
            "_recovery_source": "daily_close_snapshot_open_recovery",
        }

    def _promote_historical_misses(self, current_date: date) -> int:
        """Freeze the original due session instead of shifting orders to deployment day."""
        promoted = 0
        active_accounts = {
            row["id"] for row in self.ledger.list_account_rows(active_only=True)
        }
        for row in self.ledger.planned_orders():
            order = dict(row)
            if (
                order["account_id"] not in active_accounts
                or order["status"] not in {"PLANNED", "PREFLIGHT_OK"}
            ):
                continue
            scheduled_text = order.get("scheduled_date")
            if scheduled_text:
                intended = date.fromisoformat(str(scheduled_text))
                evidence_source = "frozen_schedule"
            else:
                intended = self._first_observed_session_after(
                    date.fromisoformat(str(order["signal_date"])),
                    current_date,
                )
                evidence_source = "local_daily_partition"
            if intended is None or intended >= current_date:
                continue
            if self.ledger.mark_historically_missed(
                order["id"],
                intended,
                {
                    "source": evidence_source,
                    "observed_after_restart": True,
                    "checked_on": current_date.isoformat(),
                },
            ):
                promoted += 1
        return promoted

    def recover_missed_open(self, *, now: datetime | None = None) -> dict[str, int]:
        """Reconcile missed windows whenever reliable opening evidence becomes available."""
        current = _as_cn(now)
        result = {
            "missed": 0,
            "recovered": 0,
            "resolved": 0,
            "unknown": 0,
            "waiting_evidence": 0,
        }
        self._repair_invalid_session_orders()
        current_session_candidate = is_possible_cn_equity_session(current.date())
        with self._lock:
            result["missed"] += self._promote_historical_misses(current.date())
        existing_recovery = self.ledger.recovery_orders()
        if current.time() < OPEN_DEADLINE and not existing_recovery:
            return result
        with self._lock:
            recovered_accounts: set[str] = set()
            quotes = self._quotes_from_cache() if current_session_candidate else {}
            symbols = sorted(self.subscription_symbols())
            current_recovery_symbols = {
                str(row["symbol"])
                for row in self.ledger.recovery_orders()
                if current_session_candidate
                and row["scheduled_date"] == current.date().isoformat()
            }
            fresh_recovery_quotes: dict[str, dict[str, Any]] = {}
            missing_current_quotes = [
                symbol for symbol in symbols
                if symbol not in quotes
                or quotes[symbol].get("_quote_dt") is None
                or quotes[symbol]["_quote_dt"].date() != current.date()
            ] if current_session_candidate else []
            quote_retry_due = (
                self._recovery_quote_fetch_last is None
                or (current - self._recovery_quote_fetch_last).total_seconds() >= 300
            )
            requires_completed_day_evidence = (
                current.time() >= dt_time(15, 0) and bool(current_recovery_symbols)
            )
            if (missing_current_quotes or requires_completed_day_evidence) and quote_retry_due:
                quote_service = getattr(self.app_state, "quote_service", None)
                if quote_service is not None:
                    self._recovery_quote_fetch_last = current
                    try:
                        refresh = getattr(quote_service, "refresh_paper_symbols", None)
                        if refresh is None:
                            refresh = quote_service.refresh
                        fetched = refresh()
                        records = fetched.get("records", []) if isinstance(fetched, dict) else []
                        fresh_recovery_quotes = _quote_map(
                            records,
                            source="realtime_recovery",
                        )
                        quotes.update(fresh_recovery_quotes)
                    except Exception:
                        logger.exception("paper recovery quote refresh failed")

            current_minutes = (
                self._opening_minutes(
                    self.repo.get_minute_batch(symbols, current.date()),
                    current.date(),
                )
                if current_session_candidate and symbols else {}
            )
            minute_quotes = {
                symbol: {
                    "symbol": symbol,
                    "last_price": row.get("close"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "volume": row.get("volume"),
                    "quote_at": _as_cn(row["datetime"]).isoformat(timespec="seconds"),
                    "_quote_dt": _as_cn(row["datetime"]),
                    "source": "minute_k",
                }
                for symbol, row in current_minutes.items()
            }
            if current_session_candidate and current.time() >= OPEN_DEADLINE:
                for row in self.ledger.planned_orders():
                    order = dict(row)
                    if (
                        order["status"] not in {"PLANNED", "PREFLIGHT_OK"}
                        or date.fromisoformat(str(order["signal_date"])) >= current.date()
                        or (
                            order.get("scheduled_date") is not None
                            and order["scheduled_date"] != current.date().isoformat()
                        )
                        or str(order["symbol"]) in current_minutes
                    ):
                        continue
                    minute = self._remote_opening_minute(order, current.date(), current)
                    if minute is not None:
                        current_minutes[str(order["symbol"])] = minute
                        minute_quotes[str(order["symbol"])] = {
                            "symbol": str(order["symbol"]),
                            "last_price": minute.get("close"),
                            "open": minute.get("open"),
                            "high": minute.get("high"),
                            "low": minute.get("low"),
                            "volume": minute.get("volume"),
                            "quote_at": _as_cn(minute["datetime"]).isoformat(
                                timespec="seconds"
                            ),
                            "_quote_dt": _as_cn(minute["datetime"]),
                            "source": str(
                                minute.get("_recovery_source")
                                or "minute_k_targeted_recovery"
                            ),
                        }
            # 分钟 K 通常不携带昨收；恢复路径仍需用冻结信号日收盘价完成
            # 涨跌停校验，不能因为盘前接口字段缺失而绕过同一套规则。
            for row in self.ledger.planned_orders():
                order = dict(row)
                minute_quote = minute_quotes.get(str(order["symbol"]))
                if minute_quote is None or _valid_price(minute_quote.get("prev_close")):
                    continue
                minute_quote["prev_close"] = self._reference_close(order, minute_quote)
            confirmation_quotes = {**minute_quotes, **quotes}
            if current_session_candidate and current.time() >= OPEN_DEADLINE and (
                self._market_is_observed(current.date(), confirmation_quotes)
                or bool(current_minutes)
            ):
                self.preflight_all(now=current, quotes=confirmation_quotes)
                for row in self.ledger.planned_orders():
                    order = dict(row)
                    eligible_today = (
                        date.fromisoformat(str(order["signal_date"])) < current.date()
                        and (
                            not order["scheduled_date"]
                            or order["scheduled_date"] == current.date().isoformat()
                        )
                    )
                    if not eligible_today:
                        continue
                    if order["status"] == "PLANNED":
                        self.ledger.resolve_incident(
                            f"order:{order['id']}:ORDER_QUOTE_NOT_READY:{current.date()}"
                        )
                        self.ledger.terminal_order(
                            order["id"],
                            status="UNKNOWN_MARKET_DATA",
                            reason="开盘窗口已结束, 但该标的盘前/开盘行情证据不足",
                            quality="NO_RELIABLE_OPEN_DATA",
                            scheduled_date=current.date(),
                            severity="critical",
                        )
                        result["missed"] += 1
                        result["unknown"] += 1
                        continue
                    if (
                        order["status"] != "PREFLIGHT_OK"
                        or order["scheduled_date"] != current.date().isoformat()
                    ):
                        continue
                    self.ledger.terminal_order(
                        order["id"],
                        status="MISSED_EXECUTION",
                        reason="服务未在 09:30 执行该订单, 已按真实发生时间记录错过开盘",
                        quality="MISSED_EXECUTION",
                        severity="critical",
                    )
                    result["missed"] += 1

            recovery_orders = [dict(row) for row in self.ledger.recovery_orders()]
            by_date: dict[date, list[dict[str, Any]]] = {}
            for order in recovery_orders:
                trading_date = date.fromisoformat(str(order["scheduled_date"]))
                by_date.setdefault(trading_date, []).append(order)

            for trading_date, orders in by_date.items():
                date_symbols = sorted({str(order["symbol"]) for order in orders})
                minute_by_symbol = (
                    dict(current_minutes)
                    if trading_date == current.date()
                    else self._opening_minutes(
                        self.repo.get_minute_batch(date_symbols, trading_date),
                        trading_date,
                    )
                )
                for order in orders:
                    minute = minute_by_symbol.get(str(order["symbol"]))
                    if minute is None:
                        minute = self._remote_opening_minute(order, trading_date, current)
                        if minute is not None:
                            minute_by_symbol[str(order["symbol"])] = minute
                    if minute is None:
                        minute = self._completed_day_opening_evidence(
                            order,
                            trading_date,
                            current,
                            fresh_recovery_quotes,
                        )
                        if minute is not None:
                            minute_by_symbol[str(order["symbol"])] = minute
                    reliable = minute is not None and all(
                        _valid_price(minute.get(key))
                        for key in ("open", "high", "low", "close")
                    ) and minute.get("volume") not in (None, 0)
                    if not reliable:
                        if order["status"] != "UNKNOWN_MARKET_DATA":
                            self.ledger.terminal_order(
                                order["id"],
                                status="UNKNOWN_MARKET_DATA",
                                reason=(
                                    "09:31 当时缺少可靠开盘行情; 已进入自动证据恢复队列, "
                                    "不会制造成交"
                                ),
                                quality="NO_RELIABLE_OPEN_DATA",
                            )
                            result["unknown"] += 1
                        result["waiting_evidence"] += 1
                        continue
                    quote = {
                        **minute,
                        "last_price": minute["close"],
                        "quote_at": _as_cn(minute["datetime"]).isoformat(timespec="seconds"),
                        "_quote_dt": _as_cn(minute["datetime"]),
                        "source": str(
                            minute.get("_recovery_source") or "minute_k_recovery"
                        ),
                    }
                    blocked, reason = self._blocked_status(order, quote, trading_date)
                    if blocked:
                        if blocked == "UNKNOWN_MARKET_DATA":
                            result["waiting_evidence"] += 1
                            continue
                        self.ledger.terminal_order(
                            order["id"],
                            status=blocked,
                            reason=reason,
                            quality="RECOVERED_LATE",
                        )
                        result["resolved"] += 1
                        continue
                    account = self.ledger.get_account(order["account_id"])
                    price = float(minute["open"])
                    quantity = int(order["requested_qty"])
                    if order["side"] == "BUY":
                        affordable = round_lot_quantity(
                            account["summary"]["cash"],
                            price,
                            _account_cost_model(account["config"]),
                        )
                        quantity = min(quantity or affordable, affordable)
                    else:
                        positions = {p["symbol"]: p for p in account["positions"]}
                        position = positions.get(order["symbol"])
                        quantity = min(
                            quantity,
                            int(position["available_qty"]) if position else 0,
                        )
                    if quantity <= 0:
                        status = (
                            "REJECTED_INSUFFICIENT_CASH"
                            if order["side"] == "BUY"
                            else "EXECUTION_FAILED"
                        )
                        reason = (
                            "恢复撮合时可用资金不足"
                            if order["side"] == "BUY"
                            else "恢复撮合时可卖数量为零或受 T+1 锁定"
                        )
                        self.ledger.terminal_order(
                            order["id"],
                            status=status,
                            reason=reason,
                            quality="RECOVERED_LATE",
                        )
                        result["resolved"] += 1
                        continue
                    try:
                        self.ledger.execute_fill(
                            order["id"],
                            price=price,
                            quantity=quantity,
                            quote_at=quote["_quote_dt"],
                            source=quote["source"],
                            quality="RECOVERED_LATE",
                            previous_close=self._reference_close(order, quote),
                        )
                        current_quote = (
                            fresh_recovery_quotes.get(str(order["symbol"]))
                            or quotes.get(str(order["symbol"]))
                        )
                        if current_quote is not None:
                            self.ledger.update_marks(
                                {str(order["symbol"]): current_quote},
                                source=str(current_quote.get("source") or "recovery_mark"),
                            )
                        result["recovered"] += 1
                        recovered_accounts.add(str(order["account_id"]))
                    except Exception as exc:
                        self.ledger.terminal_order(
                            order["id"],
                            status="EXECUTION_FAILED",
                            reason=str(exc),
                            quality="RECOVERED_LATE",
                        )
                        result["resolved"] += 1
            if current.time() >= SETTLEMENT_TIME:
                recovered_accounts.update(
                    self.ledger.accounts_needing_settlement_restatement(current.date())
                )
                for account_id in sorted(recovered_accounts):
                    self.ledger.settle_account(
                        account_id,
                        current.date(),
                        source="15:05_close",
                        restatement=True,
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
