# ruff: noqa: RUF001
"""Backtest-only adapter for the frozen risk-admitted forecast portfolio.

The portfolio cannot be expressed by ``StrategyDef``: it mixes 5% weekly
micro-cap slots with 20% event slots and admits an event only on its first
tradable open when the prior-day risk state is off.  This adapter therefore
exposes the verified account-level research artifact to the existing backtest
result contract without registering the portfolio in screeners or monitors.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.risk_admitted_forecast_paper import (
    EVENT_LIFETIME,
    EVENT_WEIGHT,
    INITIAL_CAPITAL,
    MAX_EVENT_POSITIONS,
    MICROCAP_WEIGHT,
    RESULT_FILE,
    RESULT_SHA256,
    STRATEGY_ID,
    TOTAL_SLOTS,
)

STRATEGY_NAME = "主板微盘 × 特异性业绩预告"
EXECUTION_BACKEND = "frozen_portfolio"
FROZEN_PERIODS = (
    {
        "id": "validation",
        "label": "2021–2023 独立验证",
        "start": "2021-01-04",
        "end": "2023-12-29",
    },
    {
        "id": "known_stress",
        "label": "2024–2026-08 压力期",
        "start": "2024-01-02",
        "end": "2026-08-28",
    },
)
DEFAULT_PERIOD_ID = "known_stress"
ROLLING_SCHEMA = "risk-admitted-idiosyncratic-forecast-rolling-v1"
ROLLING_START = date(2024, 1, 1)
FROZEN_THROUGH = date(2026, 8, 28)


def strategy_detail(data_dir: Path) -> dict[str, Any]:
    """Return a read-only strategy descriptor used only by Backtest."""
    path = data_dir / "research" / RESULT_FILE
    verified = _verified(path, RESULT_SHA256)
    periods = [dict(row) for row in FROZEN_PERIODS]
    latest = _latest_rolling_end(data_dir)
    if latest is not None and latest > FROZEN_THROUGH:
        periods.append(
            {
                "id": "rolling_observation",
                "label": f"冻结合同延伸至 {latest:%m-%d}",
                "start": FROZEN_PERIODS[-1]["start"],
                "end": latest.isoformat(),
                "evidence_scope": "post_freeze_observation",
            }
        )
    default = periods[-1] if periods[-1]["id"] == "rolling_observation" else next(
        row for row in periods if row["id"] == DEFAULT_PERIOD_ID
    )
    return {
        "id": STRATEGY_ID,
        "name": STRATEGY_NAME,
        "description": (
            "冻结账户组合：沪深主板微盘周调仓为底仓；风险关闭时，使用公司特异性"
            "正向业绩预告替换部分微盘暴露。回测读取哈希校验后的账户级历史证据。"
        ),
        "tags": ["主板", "微盘", "业绩预告", "冻结组合"],
        "source": "builtin",
        "execution_backend": EXECUTION_BACKEND,
        "asset_types": ["stock"],
        "timeframes": ["1d"],
        "version": "1.0.0",
        "basic_filter": {"enabled": True, "boards": ["沪主板", "深主板"]},
        "params": [],
        "params_defaults": {},
        "scoring": {},
        "scoring_directions": {},
        "entry_signals": [],
        "exit_signals": [],
        "minute_exit_trigger_supported_signals": [],
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "trailing_take_profit_activate": None,
        "trailing_take_profit_drawdown": None,
        "max_hold_days": EVENT_LIFETIME,
        "order_by": "frozen_target_weight",
        "descending": True,
        "limit": TOTAL_SLOTS,
        "display_limit": TOTAL_SLOTS,
        "immutable_contract": True,
        "artifact_verified": verified,
        "backtest_defaults": {"start": default["start"], "end": default["end"]},
        "backtest_periods": periods,
        "locked_contract": {
            "initial_capital": INITIAL_CAPITAL,
            "total_slots": TOTAL_SLOTS,
            "microcap_weight": MICROCAP_WEIGHT,
            "event_weight": EVENT_WEIGHT,
            "max_event_positions": MAX_EVENT_POSITIONS,
            "event_lifetime_days": EVENT_LIFETIME,
            "entry_fill": "open_t+1",
            "exit_fill": "open_t+1",
            "commission_pct": 0.0002,
            "stamp_tax": "按成交日历史税率",
            "slippage_bps": 5.0,
        },
    }


def run_artifact_backtest(
    data_dir: Path,
    *,
    start: date,
    end: date,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Any = None,
    expected_sha256: str = RESULT_SHA256,
) -> dict[str, Any]:
    """Adapt one immutable artifact window to ``StrategyBacktestResult`` fields."""
    if start > end:
        raise ValueError("回测开始日期不能晚于结束日期")
    _check_cancel(cancel_event)
    _progress(progress_cb, "正在校验冻结账户回测证据", 0, 3, start)

    path = data_dir / "research" / RESULT_FILE
    if not _verified(path, expected_sha256):
        raise ValueError("冻结账户回测证据缺失或哈希不一致，拒绝返回未经验证的结果")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if end > FROZEN_THROUGH:
        period_id, period, evidence = _load_rolling_period(
            data_dir,
            frozen_payload=payload,
            start=start,
            end=end,
            expected_sha256=expected_sha256,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
    else:
        period_id, period = _select_period(payload, start, end)
        evidence = {
            "scope": "frozen_evidence",
            "period_id": period_id,
            "artifact_sha256": expected_sha256,
            "artifact_verified": True,
            "contract_frozen": payload.get("contract_frozen"),
        }

    _check_cancel(cancel_event)
    _progress(progress_cb, "正在读取账户净值与逐笔成交", 1, 3, start)
    all_daily = sorted(period.get("daily_equity") or [], key=lambda row: str(row["date"]))
    selected_daily = [
        row for row in all_daily if start.isoformat() <= str(row["date"])[:10] <= end.isoformat()
    ]
    if not selected_daily:
        raise ValueError("所选区间内没有冻结账户净值记录")

    previous = [row for row in all_daily if str(row["date"])[:10] < start.isoformat()]
    base_equity = float(previous[-1]["equity"]) if previous else INITIAL_CAPITAL
    equity_curve, drawdown_curve, returns = _curves(selected_daily, base_equity)
    trades = _closed_trades(
        period.get("orders") or [],
        period.get("settlements") or [],
        all_daily,
        start,
        end,
    )
    stats = _stats(
        selected_daily,
        returns,
        trades,
        base_equity,
        period.get("orders") or [],
        start,
        end,
    )
    _check_cancel(cancel_event)
    _progress(progress_cb, "冻结账户回测结果已还原", 3, 3, end)

    return {
        "config": {
            "strategy_id": STRATEGY_ID,
            "symbols": None,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "matching": "open_t+1",
            "entry_fill": "open_t+1",
            "exit_fill": "open_t+1",
            "commission_pct": 0.0002,
            "stamp_tax_pct": None,
            "stamp_tax_rule": "historical_by_trade_date",
            "slippage_bps": 5.0,
            "max_positions": TOTAL_SLOTS,
            "max_exposure_pct": 1.0,
            "initial_capital": INITIAL_CAPITAL,
            "position_sizing": "frozen_target_weight",
            "mode": "position",
            "holding_days": EVENT_LIFETIME,
            "asset_type": "stock",
            "minute_price_fill": False,
            "minute_exit_trigger": False,
            "regime_filter": None,
            "overrides": {"basic_filter": {"enabled": True, "boards": ["沪主板", "深主板"]}},
        },
        "stats": {
            **stats,
            "mode": "position",
            "execution_backend": EXECUTION_BACKEND,
            "frozen_evidence": evidence,
        },
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "benchmark_curve": [],
        "trades": trades,
        "open_positions": [],
        "pending_orders": [],
        "per_symbol_stats": _per_symbol_stats(trades),
        "strategy_info": {
            "id": STRATEGY_ID,
            "name": STRATEGY_NAME,
            "description": "哈希校验后的冻结账户级回测，不是普通选股策略近似。",
            "entry_signals": [],
            "exit_signals": [],
            "stop_loss": None,
            "take_profit": None,
            "trailing_stop": None,
            "trailing_take_profit_activate": None,
            "trailing_take_profit_drawdown": None,
            "score_min": None,
            "score_max": None,
            "max_hold_days": EVENT_LIFETIME,
            "source": "builtin",
            "execution_backend": EXECUTION_BACKEND,
        },
    }


def _select_period(payload: dict[str, Any], start: date, end: date) -> tuple[str, dict[str, Any]]:
    results = payload.get("results") or {}
    for declared in FROZEN_PERIODS:
        row = results.get(declared["id"])
        if not isinstance(row, dict):
            continue
        period = row.get("period") or {}
        period_start = str(period.get("start") or declared["start"])[:10]
        period_end = str(period.get("end") or declared["end"])[:10]
        if period_start <= start.isoformat() and end.isoformat() <= period_end:
            return declared["id"], row
    supported = "；".join(
        f"{row['label']} {row['start']} 至 {row['end']}" for row in FROZEN_PERIODS
    )
    raise ValueError(f"所选区间未包含在同一个冻结证据窗口内。可用窗口：{supported}")


def _load_rolling_period(
    data_dir: Path,
    *,
    frozen_payload: dict[str, Any],
    start: date,
    end: date,
    expected_sha256: str,
    progress_cb: Callable[[dict[str, Any]], None] | None,
    cancel_event: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if start < date.fromisoformat(FROZEN_PERIODS[-1]["start"]):
        raise ValueError("冻结合同延伸回放只能从 2024-01-02 及之后开始")
    latest = _latest_rolling_end(data_dir)
    # 已有的按日缓存仍需允许读取, 避免仅因测试/启动瞬间未加载数据状态而误拒绝。
    path = _rolling_path(data_dir, end)
    if not path.is_file():
        if latest is None or end > latest:
            available = latest.isoformat() if latest is not None else "暂无完整数据"
            raise ValueError(f"结束日期超过最新完整交易日，可回放至 {available}")
        _generate_rolling_artifact(
            data_dir,
            end=end,
            output=path,
            contract_sha256=expected_sha256,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
    try:
        rolling_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("冻结合同延伸回放文件损坏，请重新生成") from exc
    if rolling_payload.get("schema_version") != ROLLING_SCHEMA:
        raise ValueError("冻结合同延伸回放版本不兼容")
    if rolling_payload.get("contract_sha256") != expected_sha256:
        raise ValueError("冻结合同延伸回放使用了不同策略合同，拒绝混合结果")
    period = rolling_payload.get("result")
    if not isinstance(period, dict):
        raise ValueError("冻结合同延伸回放缺少账户结果")
    declared = period.get("period") or {}
    if str(declared.get("end") or "")[:10] < end.isoformat():
        raise ValueError("冻结合同延伸回放尚未覆盖所选结束日期")
    _validate_frozen_prefix(frozen_payload, period)
    observed = [
        str(row.get("date"))[:10]
        for row in period.get("daily_equity") or []
        if str(row.get("date"))[:10] > FROZEN_THROUGH.isoformat()
        and str(row.get("date"))[:10] <= end.isoformat()
    ]
    if not observed:
        raise ValueError("所选区间没有冻结后的完整交易日记录")
    return (
        "rolling_observation",
        period,
        {
            "scope": "frozen_contract_extension",
            "period_id": "rolling_observation",
            "artifact_sha256": _sha256(path),
            "artifact_verified": True,
            "contract_sha256": expected_sha256,
            "contract_frozen": frozen_payload.get("contract_frozen"),
            "frozen_through": FROZEN_THROUGH.isoformat(),
            "observed_from": min(observed),
            "observed_through": max(observed),
            "generated_at": rolling_payload.get("generated_at"),
        },
    )


def _validate_frozen_prefix(
    frozen_payload: dict[str, Any], rolling_period: dict[str, Any]
) -> None:
    frozen = (frozen_payload.get("results") or {}).get(DEFAULT_PERIOD_ID) or {}
    for key in ("daily_equity", "orders", "settlements"):
        frozen_rows = _rows_through(frozen.get(key) or [], FROZEN_THROUGH)
        rolling_rows = _rows_through(rolling_period.get(key) or [], FROZEN_THROUGH)
        if frozen_rows != rolling_rows:
            raise ValueError(f"冻结区间一致性校验失败：{key} 与原始证据不一致")


def _rows_through(rows: list[dict[str, Any]], cutoff: date) -> list[str]:
    selected = [
        row
        for row in rows
        if str(row.get("date") or row.get("trading_date") or "")[:10]
        <= cutoff.isoformat()
    ]
    return sorted(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in selected
    )


def _latest_rolling_end(data_dir: Path) -> date | None:
    from app.services.data_integrity import latest_complete_partition_date

    daily = latest_complete_partition_date(data_dir, "kline_daily")
    receipt_path = data_dir / "event_data" / "forecast" / "sync_status.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        forecast = date.fromisoformat(str(receipt.get("end_date"))[:10])
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        forecast = None
    if daily is None or forecast is None:
        return None
    return min(daily, forecast)


def _rolling_path(data_dir: Path, end: date) -> Path:
    return (
        data_dir
        / "research"
        / "rolling"
        / f"risk_admitted_idiosyncratic_forecast_{end.isoformat()}.json"
    )


def _research_script() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[3] / "research" / "run_risk_admitted_rolling_replay.py",
        here.parents[2] / "research" / "run_risk_admitted_rolling_replay.py",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("延伸回放执行程序未随服务部署")


def _generate_rolling_artifact(
    data_dir: Path,
    *,
    end: date,
    output: Path,
    contract_sha256: str,
    progress_cb: Callable[[dict[str, Any]], None] | None,
    cancel_event: Any,
) -> None:
    _check_cancel(cancel_event)
    _progress(progress_cb, "正在按冻结合同生成最新数据回放", 1, 3, end)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_research_script()),
        "--data-dir",
        str(data_dir),
        "--end",
        end.isoformat(),
        "--contract-sha256",
        contract_sha256,
        "--output",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_text = ""
    while True:
        try:
            output_text, _ = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancel_event is None or not cancel_event.is_set():
                continue
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise ValueError("cancelled") from None
    output_lines = output_text.splitlines()
    if process.returncode != 0 or not output.is_file():
        detail = output_lines[-1] if output_lines else f"exit={process.returncode}"
        raise ValueError(f"冻结合同延伸回放生成失败：{detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curves(
    rows: list[dict[str, Any]], base_equity: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    equity_curve: list[dict[str, Any]] = []
    drawdown_curve: list[dict[str, Any]] = []
    returns: list[float] = []
    previous = base_equity
    peak = base_equity
    for row in rows:
        value = float(row["equity"])
        daily_return = value / previous - 1.0 if previous > 0 else 0.0
        returns.append(daily_return)
        previous = value
        peak = max(peak, value)
        day = str(row["date"])[:10]
        equity_curve.append(
            {
                "date": day,
                "value": round(value, 4),
                "cash": round(float(row.get("cash") or 0.0), 4),
                "positions": int(row.get("position_count") or 0),
                "exposure": round(1.0 - float(row.get("cash_ratio") or 0.0), 6),
            }
        )
        drawdown_curve.append(
            {"date": day, "value": round(value / peak - 1.0 if peak > 0 else 0.0, 6)}
        )
    return equity_curve, drawdown_curve, returns


def _closed_trades(
    orders: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    open_buys: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    trading_index = {str(row["date"])[:10]: index for index, row in enumerate(daily)}
    exits = [row for row in orders if row.get("status") == "FILLED"] + [
        {
            "date": row.get("date"),
            "symbol": row.get("symbol"),
            "side": "SETTLEMENT",
            "status": row.get("status"),
            "gross": row.get("recovery_value") or 0.0,
            "commission": 0.0,
            "stamp_tax": 0.0,
            "slippage": 0.0,
            "exit_trigger": row.get("status"),
        }
        for row in settlements
    ]
    for row in sorted(
        exits, key=lambda item: (str(item.get("date"))[:10], item.get("side") != "SELL")
    ):
        symbol = str(row.get("symbol") or "")
        side = row.get("side")
        if side == "BUY":
            open_buys[symbol] = row
            continue
        if side not in {"SELL", "SETTLEMENT"}:
            continue
        buy = open_buys.pop(symbol, None)
        if buy is None:
            continue
        exit_day = str(row.get("date"))[:10]
        if not (start.isoformat() <= exit_day <= end.isoformat()):
            continue
        entry_day = str(buy.get("date"))[:10]
        shares = int(buy.get("raw_shares") or 0)
        if shares <= 0:
            continue
        entry_gross = float(buy.get("gross") or 0.0)
        exit_gross = float(row.get("gross") or 0.0)
        entry_value = (
            entry_gross + float(buy.get("commission") or 0.0) + float(buy.get("slippage") or 0.0)
        )
        exit_value = (
            exit_gross
            - float(row.get("commission") or 0.0)
            - float(row.get("stamp_tax") or 0.0)
            - float(row.get("slippage") or 0.0)
        )
        pnl_amount = exit_value - entry_value
        entry_idx = trading_index.get(entry_day, 0)
        exit_idx = trading_index.get(exit_day, entry_idx)
        trigger = str(row.get("exit_trigger") or "portfolio_rebalance")
        reason_map = {
            "max_holding_sessions": "达到冻结持有期",
            "forced_rebalance": "冻结组合强制调仓",
            "portfolio_rebalance": "冻结组合目标调仓",
            "DELISTED_WRITE_OFF": "退市结算",
        }
        completed.append(
            {
                "symbol": symbol,
                "entry_date": entry_day,
                "exit_date": exit_day,
                "entry_price": entry_gross / shares,
                "exit_price": exit_gross / shares,
                "pnl_pct": pnl_amount / entry_value if entry_value > 0 else 0.0,
                "duration": max(1, exit_idx - entry_idx),
                "exit_reason": reason_map.get(trigger, trigger),
                "shares": shares,
                "lots": shares // 100,
                "position_pct": buy.get("target_weight"),
                "entry_value": entry_value,
                "exit_value": exit_value,
                "pnl_amount": pnl_amount,
                "entry_score": None,
                "entry_signal_date": str(buy.get("signal_date"))[:10]
                if buy.get("signal_date")
                else None,
                "exit_signal_date": exit_day,
                "blocked_exit_days": 0,
                "entry_signal_id": buy.get("family"),
                "exit_signal_id": trigger,
            }
        )
    return completed


def _stats(
    daily: list[dict[str, Any]],
    returns: list[float],
    trades: list[dict[str, Any]],
    base_equity: float,
    orders: list[dict[str, Any]],
    start: date,
    end: date,
) -> dict[str, Any]:
    final_equity = float(daily[-1]["equity"])
    total_return = final_equity / base_equity - 1.0 if base_equity > 0 else 0.0
    annual_return = (
        (1.0 + total_return) ** (252.0 / len(returns)) - 1.0
        if returns and total_return > -1.0
        else None
    )
    mean = statistics.fmean(returns) if returns else 0.0
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    downside = [value for value in returns if value < 0]
    downside_dev = statistics.stdev(downside) if len(downside) > 1 else 0.0
    trade_returns = [float(row["pnl_pct"]) for row in trades]
    wins = [value for value in trade_returns if value > 0]
    losses = [value for value in trade_returns if value <= 0]
    scoped_orders = [
        row for row in orders if start.isoformat() <= str(row.get("date"))[:10] <= end.isoformat()
    ]
    execution = defaultdict(int)
    reason_keys = {
        "limit_up": "buy_limit_up",
        "limit_down": "sell_limit_down",
        "suspended": "buy_suspended",
        "signal_capacity": "buy_exposure",
        "zero_lot_or_cash": "buy_exposure",
    }
    for row in scoped_orders:
        reason = row.get("reason")
        if reason in reason_keys:
            execution[reason_keys[reason]] += 1
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": _max_drawdown(returns),
        "sharpe": mean / stdev * math.sqrt(252.0) if stdev > 0 else None,
        "sortino": mean / downside_dev * math.sqrt(252.0) if downside_dev > 0 else None,
        "win_rate": len(wins) / len(trade_returns) if trade_returns else None,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else None,
        "n_trades": len(trades),
        "final_equity": final_equity,
        "avg_return": statistics.fmean(trade_returns) if trade_returns else None,
        "median_return": statistics.median(trade_returns) if trade_returns else None,
        "avg_duration": statistics.fmean(float(row["duration"]) for row in trades)
        if trades
        else None,
        "best": max(trade_returns) if trade_returns else None,
        "worst": min(trade_returns) if trade_returns else None,
        "n_days": len(daily),
        "n_candidates": sum(row.get("side") == "BUY" for row in scoped_orders),
        "avg_daily_candidates": (
            sum(row.get("side") == "BUY" for row in scoped_orders) / len(daily) if daily else 0.0
        ),
        "mean_cash_ratio": statistics.fmean(float(row.get("cash_ratio") or 0.0) for row in daily),
        "execution": dict(execution),
        "selection": {
            "strategy_matches": sum(row.get("side") == "BUY" for row in scoped_orders),
            "entry_candidates": sum(row.get("side") == "BUY" for row in scoped_orders),
            "entry_trigger_enabled": True,
        },
    }


def _per_symbol_stats(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in trades:
        grouped[str(row["symbol"])].append(float(row["pnl_pct"]))
    result = []
    for symbol, values in grouped.items():
        total = math.prod(1.0 + value for value in values) - 1.0
        result.append(
            {
                "symbol": symbol,
                "n_trades": len(values),
                "total_return": total,
                "win_rate": sum(value > 0 for value in values) / len(values),
                "best": max(values),
                "worst": min(values),
            }
        )
    return sorted(result, key=lambda row: row["total_return"], reverse=True)


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def _verified(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    return _cached_sha256(str(path), stat.st_mtime_ns, stat.st_size) == expected_sha256


@lru_cache(maxsize=8)
def _cached_sha256(path: str, _mtime_ns: int, _size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ValueError("cancelled")


def _progress(
    callback: Callable[[dict[str, Any]], None] | None,
    message: str,
    day: int,
    total: int,
    current: date,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": "simulation",
                "message": message,
                "day": day,
                "total": total,
                "date": current.isoformat(),
                "equity": 0,
            }
        )
