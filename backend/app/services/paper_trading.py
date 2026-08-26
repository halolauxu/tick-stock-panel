"""基于策略回测撮合口径的持久化模拟交易账户。"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.backtest.strategy import StrategyBacktestConfig
from app.backtest.worker import make_worker_task, run_worker_task

logger = logging.getLogger(__name__)

_ACCOUNT_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_STORE_LOCK = threading.RLock()
_RUN_LOCKS: dict[str, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


class PaperTradingStoreError(RuntimeError):
    pass


def _normalise_account(account: dict[str, Any]) -> dict[str, Any]:
    """兼容首版账户, 并补齐严格前向运行所需的激活边界。"""
    account = dict(account)
    if int(account.get("schema_version", 1)) >= 2:
        return account
    original_start = date.fromisoformat(str(account["start_date"]))
    try:
        created_day = date.fromisoformat(str(account.get("created_at", ""))[:10])
    except ValueError:
        created_day = original_start + timedelta(days=1)
    signal_start = max(original_start + timedelta(days=1), created_day)
    account.update({
        "schema_version": 2,
        "baseline_date": original_start.isoformat(),
        "signal_start_date": signal_start.isoformat(),
        "execution_policy": "after_close_daily",
    })
    return account


def _signal_start(account: dict[str, Any]) -> date:
    return date.fromisoformat(str(account.get("signal_start_date") or account["start_date"]))


class PaperTradingStore:
    """每账户一个 JSON 文件; 配置冻结, 结果以完整快照原子替换。"""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "paper_trading" / "accounts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, account_id: str) -> Path:
        if not _ACCOUNT_ID_RE.fullmatch(account_id):
            raise KeyError(account_id)
        return self.root / f"{account_id}.json"

    def _read(self, account_id: str) -> dict[str, Any]:
        path = self._path(account_id)
        if not path.exists():
            raise KeyError(account_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise PaperTradingStoreError(f"模拟账户文件已损坏: {path.name}") from exc
        if not isinstance(value, dict) or value.get("id") != account_id:
            raise PaperTradingStoreError(f"模拟账户文件格式无效: {path.name}")
        return _normalise_account(value)

    def get(self, account_id: str) -> dict[str, Any]:
        with _STORE_LOCK:
            return self._read(account_id)

    def list(self) -> list[dict[str, Any]]:
        with _STORE_LOCK:
            accounts = [self._read(path.stem) for path in sorted(self.root.glob("*.json"))]
        accounts.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return accounts

    def create(self, *, name: str, start_date: date, config: dict[str, Any]) -> dict[str, Any]:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("模拟账户名称不能为空")
        try:
            json.dumps(config, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("模拟账户配置无法序列化") from exc
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        signal_start = start_date + timedelta(days=1)
        account = {
            "schema_version": 2,
            "id": uuid.uuid4().hex[:12],
            "name": clean_name[:80],
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "baseline_date": start_date.isoformat(),
            "signal_start_date": signal_start.isoformat(),
            "start_date": signal_start.isoformat(),
            "execution_policy": "after_close_daily",
            "execution_state": {
                "code": "waiting_first_data",
                "label": "等待首个前向交易日",
                "detail": f"只处理 {signal_start.isoformat()} 及之后新完成的数据, 不倒填创建前订单",
                "next_action": "等待盘后日线与指标完整落盘",
            },
            "last_processed_date": None,
            "last_run_at": None,
            "last_error": None,
            "config": config,
            "result": None,
        }
        with _STORE_LOCK:
            self._write(account)
        return account

    def replace(self, account: dict[str, Any]) -> dict[str, Any]:
        account = dict(account)
        account["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        with _STORE_LOCK:
            self._read(str(account["id"]))
            self._write(account)
        return account

    def set_status(self, account_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("不支持的模拟账户状态")
        with _STORE_LOCK:
            account = self._read(account_id)
            account["status"] = status
            account["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self._write(account)
            return account

    def delete(self, account_id: str) -> dict[str, Any]:
        """删除指定账户文件; 与账户运行共用单账户锁, 避免写回已删除账户。"""
        with _run_lock(account_id), _STORE_LOCK:
            account = self._read(account_id)
            try:
                self._path(account_id).unlink()
            except OSError as exc:
                raise PaperTradingStoreError(f"模拟账户删除失败: {account_id}") from exc
            return account

    def _write(self, account: dict[str, Any]) -> None:
        path = self._path(str(account["id"]))
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(account, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _run_lock(account_id: str) -> threading.Lock:
    with _RUN_LOCKS_GUARD:
        return _RUN_LOCKS.setdefault(account_id, threading.Lock())


def _empty_result(account: dict[str, Any], as_of: date) -> dict[str, Any]:
    capital = float(account["config"]["initial_capital"])
    return {
        "run_id": f"paper-{account['id']}-{as_of.isoformat()}",
        "config": {
            **account["config"],
            "start": _signal_start(account).isoformat(),
            "end": as_of.isoformat(),
            "liquidate_on_end": False,
        },
        "stats": {
            "total_return": 0.0,
            "final_equity": capital,
            "initial_capital": capital,
            "n_trades": 0,
            "execution": {},
        },
        "equity_curve": [{
            "date": as_of.isoformat(),
            "value": capital,
            "cash": capital,
            "positions": 0,
            "exposure": 0.0,
        }],
        "drawdown_curve": [{"date": as_of.isoformat(), "value": 0.0}],
        "benchmark_curve": [],
        "trades": [],
        "open_positions": [],
        "pending_orders": [],
        "per_symbol_stats": [],
        "strategy_info": {
            "id": account["config"]["strategy_id"],
            "name": account["config"].get("strategy_name") or account["config"]["strategy_id"],
            "description": "",
            "entry_signals": [],
            "exit_signals": [],
            "stop_loss": None,
            "take_profit": None,
            "trailing_stop": None,
            "trailing_take_profit_activate": None,
            "trailing_take_profit_drawdown": None,
            "score_min": None,
            "score_max": None,
            "max_hold_days": None,
            "source": "",
        },
        "elapsed_ms": 0.0,
        "error": None,
    }


def _execution_state(
    account: dict[str, Any],
    latest: date,
    result: dict[str, Any] | None,
) -> dict[str, str]:
    if account.get("last_error"):
        return {
            "code": "error",
            "label": "运行失败",
            "detail": str(account["last_error"]),
            "next_action": "修复数据或配置后手动同步",
        }
    signal_start = _signal_start(account)
    if latest < signal_start:
        return {
            "code": "waiting_first_data",
            "label": "等待首个前向交易日",
            "detail": f"当前完整数据到 {latest.isoformat()}, 账户只处理 {signal_start.isoformat()} 及之后的新信号",
            "next_action": "等待盘后日线与指标完整落盘",
        }
    result = result or {}
    positions = list(result.get("open_positions") or [])
    orders = list(result.get("pending_orders") or [])
    pending_exits = [position for position in positions if position.get("pending_exit_reason")]
    if pending_exits:
        return {
            "code": "waiting_exit",
            "label": "等待可卖时点",
            "detail": f"{len(pending_exits)} 个持仓已触发退出, 受 T+1、停牌或跌停约束尚未成交",
            "next_action": "下一完整交易日按最早可成交价格继续撮合",
        }
    if orders:
        signal_dates = sorted({str(order.get("signal_date") or "") for order in orders})
        signal_text = signal_dates[-1] if signal_dates else latest.isoformat()
        return {
            "code": "waiting_open",
            "label": "待下一交易日开盘",
            "detail": f"{signal_text} 收盘后确认 {len(orders)} 笔候选, 尚未取得下一交易日完整行情",
            "next_action": "下一完整交易日按开盘价及停牌、涨跌停约束判定成交",
        }
    if positions:
        return {
            "code": "holding",
            "label": "持仓跟踪中",
            "detail": f"当前 {len(positions)} 个持仓, 数据截至 {latest.isoformat()}",
            "next_action": "盘后重放当日风控与卖出信号",
        }
    return {
        "code": "scanning",
        "label": "等待新信号",
        "detail": f"已处理至 {latest.isoformat()}, 当日没有可执行持仓或订单",
        "next_action": "下一次完整数据落盘后继续选股",
    }


def _backtest_config(account: dict[str, Any], as_of: date) -> StrategyBacktestConfig:
    config = account["config"]
    entry_fill = config.get("entry_fill", "open_t+1")
    return StrategyBacktestConfig(
        strategy_id=config["strategy_id"],
        symbols=config.get("symbols") or None,
        start=_signal_start(account),
        end=as_of,
        params=config.get("params"),
        overrides=config.get("overrides"),
        matching=entry_fill,
        entry_fill=entry_fill,
        exit_fill=config.get("exit_fill", "open_t+1"),
        fees_pct=float(config.get("commission_pct", 0.0002)),
        commission_pct=float(config.get("commission_pct", 0.0002)),
        stamp_tax_pct=float(config.get("stamp_tax_pct", 0.001)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
        max_positions=int(config.get("max_positions", 10)),
        max_exposure_pct=float(config.get("max_exposure_pct", 1.0)),
        initial_capital=float(config.get("initial_capital", 1_000_000.0)),
        position_sizing=config.get("position_sizing", "equal"),
        mode="position",
        holding_days=int(config.get("holding_days", 5)),
        asset_type=config.get("asset_type", "stock"),
        minute_fill=bool(config.get("minute_fill", False)),
        regime_filter=config.get("regime_filter"),
        liquidate_on_end=False,
        enforce_t_plus_one=bool(config.get("enforce_t_plus_one", True)),
    )


def run_account(app_state, account_id: str, *, force: bool = False) -> dict[str, Any]:
    """同步到账户资产类型的最新完整 enriched 交易日。"""
    store = PaperTradingStore(app_state.repo.store.data_dir)
    with _run_lock(account_id):
        account = store.get(account_id)
        if account["status"] != "active" and not force:
            return account
        asset_type = account["config"].get("asset_type", "stock")
        latest = app_state.repo.latest_enriched_date(asset_type)
        if latest is None:
            raise ValueError(f"缺少{asset_type}指标数据, 请先完成日K采集与指标计算")
        if latest < _signal_start(account):
            account["result"] = _empty_result(account, latest)
            account["last_processed_date"] = latest.isoformat()
            account["last_run_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            account["last_error"] = None
            account["execution_state"] = _execution_state(account, latest, account["result"])
            return store.replace(account)
        if not force and account.get("last_processed_date") == latest.isoformat():
            return account

        config = _backtest_config(account, latest)
        try:
            result = run_worker_task(
                make_worker_task("backtest", app_state.repo.store.data_dir, config)
            )
        except Exception as exc:
            account["last_error"] = str(exc)
            account["execution_state"] = _execution_state(account, latest, account.get("result"))
            store.replace(account)
            raise

        error = result.get("error")
        if error in {"在指定区间内未产生买入信号", "no data or no signals"}:
            result = _empty_result(account, latest)
        elif error:
            account["last_error"] = str(error)
            account["execution_state"] = _execution_state(account, latest, account.get("result"))
            store.replace(account)
            raise RuntimeError(str(error))

        account["result"] = result
        account["last_processed_date"] = latest.isoformat()
        account["last_run_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        account["last_error"] = None
        account["execution_state"] = _execution_state(account, latest, result)
        return store.replace(account)


def run_active_accounts(app_state) -> dict[str, int]:
    """盘后管道成功后的失败隔离入口: 单账户失败不影响数据任务及其他账户。"""
    store = PaperTradingStore(app_state.repo.store.data_dir)
    summary = {"processed": 0, "skipped": 0, "failed": 0}
    for account in store.list():
        if account.get("status") != "active":
            summary["skipped"] += 1
            continue
        before = account.get("last_processed_date")
        try:
            updated = run_account(app_state, account["id"])
            if updated.get("last_processed_date") == before:
                summary["skipped"] += 1
            else:
                summary["processed"] += 1
        except Exception:
            summary["failed"] += 1
            logger.exception("paper account daily run failed: %s", account["id"])
    return summary
