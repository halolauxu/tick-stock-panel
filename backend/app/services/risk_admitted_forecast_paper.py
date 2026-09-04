"""Forward-only paper execution for the frozen micro-cap + forecast portfolio.

This module deliberately owns portfolio targets instead of pretending the
single-strategy screener can express 20% event positions mixed with 5% weekly
micro-cap positions.  It never replays fills before the account baseline.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from app.market_time import cn_now
from app.price_limits import (
    polars_is_risk_warning_name,
    polars_limit_price,
    polars_price_limit_pct,
)

STRATEGY_ID = "risk_admitted_idiosyncratic_forecast_v1"
ACCOUNT_ID = "risk-forecast-v1"
ACCOUNT_NAME = "主板微盘 × 特异性业绩预告（前向）"
INITIAL_CAPITAL = 200_000.0
TOTAL_SLOTS = 20
MICROCAP_WEIGHT = 0.05
EVENT_WEIGHT = 0.20
MAX_EVENT_POSITIONS = 5
EVENT_LIFETIME = 10
MIN_LISTING_DAYS = 180
MAIN_BOARD_PATTERN = r"^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$"
RESULT_SHA256 = "6c70333c3c07543a9240a86ae3166fd75f4afaf13a418167e2ef394e89964145"
RESULT_FILE = "p0_risk_admitted_idiosyncratic_forecast_overlay_v1.json"
STATE_SCHEMA = "risk-admitted-idiosyncratic-forecast-forward-v1"
THRESHOLDS = {
    "microcap_excess_5d_p10": -0.02012421350845861,
    "microcap_breadth_3d_p10": 0.2822831103242099,
    "microcap_limit_down_3d_p90": 0.011011011011011013,
    "microcap_liquidity_5d_60d_p10": 0.6401950242931338,
    "microcap_limit_down_3d_p95": 0.06341463414634146,
}
FORWARD_OBSERVATION_DAYS = 60
HISTORICAL_RESULTS = (
    {
        "id": "validation",
        "label": "2021–2023 独立验证",
        "annualized": 0.4341,
        "total_return": 1.8295,
        "max_drawdown": -0.2080,
        "yearly": (0.4598, 0.3695, 0.4153),
    },
    {
        "id": "known_stress",
        "label": "2024–2026-08 压力期",
        "annualized": 0.3218,
        "total_return": 1.0402,
        "max_drawdown": -0.3262,
        "yearly": (0.2314, 0.6296, 0.0167),
    },
)


def ensure_account(paper_service, baseline_date: date) -> dict[str, Any]:
    """Create the one immutable 200k forward account, idempotently."""
    _require_frozen_result(paper_service.repo.store.data_dir)
    try:
        account = paper_service.ledger.get_account(ACCOUNT_ID)
        config = account["config"]
        if (
            config.get("strategy_id") != STRATEGY_ID
            or config.get("research_result_sha256") != RESULT_SHA256
            or float(config.get("initial_capital") or 0.0) != INITIAL_CAPITAL
        ):
            raise ValueError("现有前向账户合同与冻结策略不一致，拒绝静默覆盖")
        return account
    except KeyError:
        account = paper_service.ledger.create_account(
            name=ACCOUNT_NAME,
            baseline_date=baseline_date,
            account_id=ACCOUNT_ID,
            config={
                "strategy_id": STRATEGY_ID,
                "strategy_name": ACCOUNT_NAME,
                "asset_type": "stock",
                "symbols": None,
                "params": {},
                "overrides": {},
                "entry_fill": "open_t+1",
                "exit_fill": "open_t+1",
                "commission_pct": 0.0002,
                "stamp_tax_pct": 0.0005,
                "slippage_bps": 5.0,
                "max_positions": TOTAL_SLOTS,
                "max_exposure_pct": 1.0,
                "initial_capital": INITIAL_CAPITAL,
                "position_sizing": "frozen_target_weight",
                "holding_days": EVENT_LIFETIME,
                "minute_fill": False,
                "exit_mode": "eod",
                "enforce_t_plus_one": True,
                "forward_only": True,
                "research_result_sha256": RESULT_SHA256,
            },
        )
        paper_service.ledger.record_account_event(
            ACCOUNT_ID,
            event_key=f"{ACCOUNT_ID}:FROZEN_CONTRACT:{RESULT_SHA256}",
            event_type="FORWARD_CONTRACT_FROZEN",
            trading_date=baseline_date,
            title="前向合同已冻结",
            detail="仅记录部署后的真实信号、次日开盘订单与成交；禁止历史回填和账户内调参",
            payload={
                "strategy_id": STRATEGY_ID,
                "result_sha256": RESULT_SHA256,
                "capital": INITIAL_CAPITAL,
                "event_weight": EVENT_WEIGHT,
                "microcap_weight": MICROCAP_WEIGHT,
            },
        )
        return account


def is_managed_account(config: dict[str, Any]) -> bool:
    return config.get("strategy_id") == STRATEGY_ID


def managed_strategy_snapshot(
    paper_service,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return product-facing provenance and live state for the managed account.

    The managed portfolio intentionally does not implement ``StrategyDef``.
    Exposing it through a separate read model keeps ordinary screening and
    monitoring from running a materially different contract; Backtest uses a
    dedicated account-level adapter for the verified frozen evidence.
    """
    current = now or cn_now()
    data_dir = paper_service.repo.store.data_dir
    result_path = data_dir / "research" / RESULT_FILE
    artifact_verified = _artifact_verified(result_path)
    try:
        account = paper_service.ledger.get_account(ACCOUNT_ID)
    except KeyError:
        account = None

    latest_enriched = paper_service.repo.latest_enriched_date("stock")
    receipt = _read_json(data_dir / "event_data" / "forecast" / "sync_status.json")
    forecast_covered = _as_date(receipt.get("end_date")) if receipt else None
    try:
        state = _load_state(_state_path(data_dir)) or {}
    except (OSError, ValueError, json.JSONDecodeError):
        state = {}
    last_signal = _as_date(
        (account or {}).get("last_processed_date") or state.get("last_signal_date")
    )
    schedule = _pipeline_schedule()
    lifecycle = _managed_lifecycle(
        account=account,
        now=current,
        schedule=schedule,
        latest_enriched=latest_enriched,
        forecast_covered=forecast_covered,
        last_signal=last_signal,
    )
    nav = (account or {}).get("nav") or []
    last_settlement = _as_date(nav[-1].get("trading_date")) if nav else None
    summary = (account or {}).get("summary") or {}
    return {
        "id": STRATEGY_ID,
        "name": ACCOUNT_NAME.removesuffix("（前向）"),
        "version": "V1",
        "kind": "managed_forward",
        "source": "frozen_research",
        "account_id": ACCOUNT_ID,
        "description": (
            "沪深主板微盘周调仓为底仓；微盘风险关闭时，使用公司特异性正向业绩预告"
            "替换部分微盘暴露。"
        ),
        "provenance": {
            "created_by": "自动研究部署",
            "introduced_commit": "1f2ef35",
            "frozen_at": "2026-09-03",
            "research_result_sha256": RESULT_SHA256,
            "artifact_verified": artifact_verified,
            "note": "V1 独立前向观察账户；不代表主板短周期 V2 路线通过。",
        },
        "contract": {
            "initial_capital": INITIAL_CAPITAL,
            "observation_trading_days": FORWARD_OBSERVATION_DAYS,
            "total_slots": TOTAL_SLOTS,
            "microcap_weight": MICROCAP_WEIGHT,
            "event_weight": EVENT_WEIGHT,
            "max_event_positions": MAX_EVENT_POSITIONS,
            "event_lifetime_days": EVENT_LIFETIME,
            "rebalance": "每周五盘后",
            "execution": "下一交易日开盘 · T+1 · 100股整数手 · 含费用/滑点/容量约束",
        },
        "historical_results": [
            {**row, "yearly": list(row["yearly"])} for row in HISTORICAL_RESULTS
        ],
        "live": {
            "account_exists": account is not None,
            "account_status": (account or {}).get("status"),
            "lifecycle": lifecycle,
            "pipeline_schedule": f"{schedule['hour']:02d}:{schedule['minute']:02d}",
            "latest_enriched_date": _date_text(latest_enriched),
            "forecast_covered_through": _date_text(forecast_covered),
            "last_signal_date": _date_text(last_signal),
            "last_settlement_date": _date_text(last_settlement),
            "signal_count": len((account or {}).get("signals") or []),
            "order_count": len((account or {}).get("orders") or []),
            "fill_count": len((account or {}).get("fills") or []),
            "position_count": int(summary.get("position_count") or 0),
            "pending_order_count": int(summary.get("pending_order_count") or 0),
            "open_incident_count": int(summary.get("open_incident_count") or 0),
            "observed_settlement_days": len(nav),
        },
    }


def _managed_lifecycle(
    *,
    account: dict[str, Any] | None,
    now: datetime,
    schedule: dict[str, int],
    latest_enriched: date | None,
    forecast_covered: date | None,
    last_signal: date | None,
) -> dict[str, Any]:
    code = "WAITING_PIPELINE"
    label = "等待盘后数据"
    detail = "盘后数据完成后自动生成当日不可变目标。"
    next_action = f"{schedule['hour']:02d}:{schedule['minute']:02d} 自动同步并封板"
    stage = "data"

    if account is None:
        code, label = "NOT_STARTED", "前向账户尚未创建"
        detail = "冻结研究文件校验通过后，服务启动会自动创建账户。"
        next_action = "核对冻结研究文件并创建账户"
    elif account.get("status") != "active":
        code, label = "PAUSED", "账户已暂停"
        detail = "暂停期间不会生成新信号或订单。"
        next_action = "恢复账户后继续按真实时钟运行"
    else:
        open_incidents = [
            row for row in (account.get("incidents") or []) if row.get("status") == "open"
        ]
        pending_orders = [
            row
            for row in (account.get("orders") or [])
            if row.get("status") in {"PLANNED", "PREFLIGHT_OK"}
        ]
        if open_incidents:
            code, label = "BLOCKED", "存在阻断异常"
            detail = str(open_incidents[0].get("detail") or open_incidents[0].get("title"))
            next_action = "先处理异常，系统不会制造成交"
            stage = "blocked"
        elif pending_orders:
            code, label = "WAITING_OPEN", "等待下一交易日开盘"
            scheduled = next(
                (str(row.get("scheduled_date")) for row in pending_orders if row.get("scheduled_date")),
                "下一交易日",
            )
            detail = f"{len(pending_orders)} 笔订单已经冻结，等待盘前校验和真实开盘行情。"
            next_action = f"{scheduled} 09:25 校验，09:30 执行"
            stage = "execution"
        elif latest_enriched is None:
            code, label = "WAITING_PIPELINE", "等待首个完整数据日"
            detail = "尚无可用于策略判定的完整 Enriched 数据。"
        elif (
            now.weekday() < 5
            and latest_enriched < now.date()
            and now.time() >= dt_time(schedule["hour"], schedule["minute"])
        ):
            code, label = "DATA_DELAYED", "盘后数据同步中或延期"
            detail = f"完整数据仍停留在 {_date_text(latest_enriched)}。"
            next_action = "等待自动补偿重试；数据完整后自动封板"
        elif now.weekday() < 5 and latest_enriched < now.date():
            code, label = "WAITING_PIPELINE", "等待今日盘后数据"
            detail = f"最近完整数据为 {_date_text(latest_enriched)}，今天尚未到盘后同步时间。"
        elif forecast_covered is None or forecast_covered < latest_enriched:
            code, label = "INPUT_DELAYED", "业绩预告输入未齐"
            detail = (
                f"业绩预告覆盖到 {_date_text(forecast_covered)}，"
                f"落后完整行情 {_date_text(latest_enriched)}。"
            )
            next_action = "等待业绩预告自动补采；输入未齐不会生成信号"
        elif last_signal is None or last_signal < latest_enriched:
            code, label = "WAITING_SEAL", "数据已齐，等待信号封板"
            detail = f"{_date_text(latest_enriched)} 输入已完整，尚未写入不可变目标。"
            next_action = "自动生成目标并写入审计账本"
            stage = "signal"
        else:
            fills = account.get("fills") or []
            positions = account.get("positions") or []
            code = "OBSERVING" if fills or positions else "SEALED_NO_ORDER"
            label = "前向观察中" if code == "OBSERVING" else "本轮信号已封板"
            detail = (
                "真实成交、持仓和结算正在持续记录。"
                if code == "OBSERVING"
                else f"{_date_text(last_signal)} 已完成判定，本轮没有待执行订单。"
            )
            next_action = "继续记录下一真实交易日" if code == "OBSERVING" else "等待下一盘后信号"
            stage = "observe"

    return {
        "code": code,
        "label": label,
        "detail": detail,
        "next_action": next_action,
        "stage": stage,
    }


def _pipeline_schedule() -> dict[str, int]:
    from app.services import preferences

    return preferences.get_pipeline_schedule()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_verified(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        stat = path.stat()
        return _cached_artifact_sha256(str(path), stat.st_mtime_ns, stat.st_size) == RESULT_SHA256
    except OSError:
        return False


@lru_cache(maxsize=4)
def _cached_artifact_sha256(path: str, _mtime_ns: int, _size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def seal_account(paper_service, account_id: str, signal_date: date) -> dict[str, int]:
    account = paper_service.ledger.get_account(account_id)
    if account["status"] != "active":
        return {"signals": 0, "orders": 0}
    if account.get("last_processed_date") == signal_date.isoformat():
        return {"signals": 0, "orders": 0}
    baseline = date.fromisoformat(account["baseline_date"])
    if signal_date < baseline:
        return {"signals": 0, "orders": 0}

    state_path = _state_path(paper_service.repo.store.data_dir)
    previous = _load_state(state_path)
    plan, next_state = build_forward_plan(
        paper_service.repo.store.data_dir,
        signal_date,
        baseline_date=baseline,
        previous_state=previous,
    )
    positions = {row["symbol"]: row for row in account["positions"]}
    live_orders = [
        row
        for row in account["orders"]
        if row["status"]
        not in {
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
        }
    ]
    pending_buy = {row["symbol"] for row in live_orders if row["side"] == "BUY"}
    pending_sell = {row["symbol"] for row in live_orders if row["side"] == "SELL"}
    targets = {row["symbol"]: row for row in plan["targets"]}
    signals = 0
    orders = 0

    for symbol in sorted(set(positions) - set(targets)):
        if symbol in pending_sell:
            continue
        position = positions[symbol]
        _, _, created = paper_service.ledger.record_signal_and_order(
            account_id=account_id,
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            name=str(position.get("name") or symbol),
            side="SELL",
            signal_date=signal_date,
            score=None,
            reason="frozen_target_exit",
            signal_ref=plan["decision_id"],
            requested_qty=int(position["quantity"]),
            target_amount=float(position["market_value"]),
            target_weight=0.0,
            planned_session="NEXT_OPEN",
            payload={"family": position.get("family"), "decision": plan["decision_id"]},
        )
        signals += int(created)
        orders += int(created)

    equity = float(account["summary"]["equity"])
    for symbol, target in sorted(targets.items(), key=lambda item: (item[1]["rank"], item[0])):
        if symbol in positions or symbol in pending_buy:
            continue
        target_amount = equity * float(target["target_weight"])
        capacity = float(target.get("signal_amount") or 0.0) * 0.01
        payload = {
            "family": target["family"],
            "rank": target["rank"],
            "decision": plan["decision_id"],
            "capacity": capacity,
            "risk_on": plan["risk"]["risk_on"],
        }
        if target_amount > capacity:
            _, created = paper_service.ledger.record_skipped_signal(
                account_id=account_id,
                strategy_id=STRATEGY_ID,
                symbol=symbol,
                name=str(target.get("name") or symbol),
                side="BUY",
                signal_date=signal_date,
                score=float(-target["rank"]),
                reason="frozen_target_entry",
                signal_ref=plan["decision_id"],
                skip_code="SIGNAL_CAPACITY",
                detail="目标金额超过信号日成交额 1% 容量上限",
                payload=payload,
            )
            signals += int(created)
            continue
        _, _, created = paper_service.ledger.record_signal_and_order(
            account_id=account_id,
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            name=str(target.get("name") or symbol),
            side="BUY",
            signal_date=signal_date,
            score=float(-target["rank"]),
            reason="frozen_target_entry",
            signal_ref=plan["decision_id"],
            # Zero tells the opening clock to size from the frozen CNY target
            # at the observed next-open price rather than today's close.
            requested_qty=0,
            target_amount=target_amount,
            target_weight=float(target["target_weight"]),
            planned_session="NEXT_OPEN",
            payload=payload,
        )
        signals += int(created)
        orders += int(created)

    paper_service.ledger.record_account_event(
        account_id,
        event_key=f"{account_id}:TARGETS:{signal_date}",
        event_type="FORWARD_TARGETS_FROZEN",
        trading_date=signal_date,
        title="专用组合目标已冻结",
        detail=(
            f"目标 {len(targets)} 只：事件 {plan['event_count']} 只、"
            f"微盘 {plan['microcap_count']} 只；次日订单 {orders} 笔"
        ),
        payload={
            "decision_id": plan["decision_id"],
            "risk": plan["risk"],
            "target_count": len(targets),
            "event_count": plan["event_count"],
            "microcap_count": plan["microcap_count"],
            "signals": signals,
            "orders": orders,
        },
    )
    paper_service.ledger.mark_signal_day(account_id, signal_date)
    _atomic_json(state_path, next_state)
    _write_decision(paper_service.repo.store.data_dir, plan)
    return {"signals": signals, "orders": orders}


def build_forward_plan(
    data_dir: Path,
    signal_date: date,
    *,
    baseline_date: date,
    previous_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_forecast_receipt(data_dir, signal_date)
    panel = _load_recent_panel(data_dir, signal_date)
    dates = panel.get_column("date").unique().sort().to_list()
    if not dates or dates[-1] != signal_date:
        raise ValueError(f"缺少 {signal_date} 完整 enriched 数据")
    features = build_daily_features(panel)
    feature_by_date = {row["date"]: row for row in features.to_dicts()}
    current = panel.filter(pl.col("date") == signal_date)

    state = previous_state or _bootstrap_state(data_dir, baseline_date)
    last_signal = _as_date(state.get("last_signal_date"))
    new_dates = [day for day in dates if last_signal is None or day > last_signal]
    if signal_date not in new_dates:
        raise ValueError("前向状态日期不早于待封板日期，拒绝覆盖")

    active_events = [dict(row) for row in state.get("active_events", [])]
    microcap_targets = [dict(row) for row in state.get("microcap_targets", [])]
    skipped_gap_dates: list[str] = []
    risk = {
        "risk_on": bool(state.get("risk_on", True)),
        "off_days": int(state.get("off_days", 0)),
        "clean_days": int(state.get("clean_days", 0)),
    }
    last_risk: dict[str, Any] | None = None
    for day in new_dates:
        active_events = [
            {**event, "age": int(event.get("age", 0)) + 1}
            for event in active_events
            if int(event.get("age", 0)) + 1 < EVENT_LIFETIME
        ]
        feature = feature_by_date.get(day)
        if feature is None:
            raise ValueError(f"缺少 {day} 风险状态特征")
        risk, last_risk = advance_risk_state(risk, feature)
        if day != signal_date:
            skipped_gap_dates.append(day.isoformat())

    if signal_date >= baseline_date and not risk["risk_on"]:
        new_events = _idiosyncratic_events_for_date(data_dir, signal_date, current)
        by_key = {(row["symbol"], row["ann_date"]): row for row in active_events}
        for row in new_events:
            by_key[(row["symbol"], row["ann_date"])] = {**row, "age": 0}
        active_events = list(by_key.values())

    # The historical contract rebalances on the last observed session of an
    # ISO week.  Forward execution uses Friday; a holiday-shortened week is
    # explicitly left as a missed rebalance rather than guessed.
    weekly_rebalance = signal_date.weekday() == 4
    if weekly_rebalance:
        microcap_targets = _microcap_targets(current)

    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in active_events:
        existing = latest_by_symbol.get(row["symbol"])
        candidate_key = (
            str(row.get("ann_date") or ""),
            float(row.get("p_change_min") or -math.inf),
            float(row.get("p_change_max") or -math.inf),
        )
        existing_key = (
            (
                str(existing.get("ann_date") or ""),
                float(existing.get("p_change_min") or -math.inf),
                float(existing.get("p_change_max") or -math.inf),
            )
            if existing is not None
            else ("", -math.inf, -math.inf)
        )
        if existing is None or candidate_key > existing_key:
            latest_by_symbol[row["symbol"]] = row
    active_events = sorted(
        latest_by_symbol.values(),
        key=lambda row: (
            -float(row.get("p_change_min") or -math.inf),
            -float(row.get("p_change_max") or -math.inf),
            row["symbol"],
        ),
    )[:MAX_EVENT_POSITIONS]
    event_symbols = {row["symbol"] for row in active_events}
    targets: list[dict[str, Any]] = []
    rank = 0
    for row in active_events:
        rank += 1
        targets.append(
            {
                **row,
                "rank": rank,
                "family": "idiosyncratic_forecast",
                "target_weight": EVENT_WEIGHT,
            }
        )
    micro_slots = TOTAL_SLOTS - len(active_events) * int(EVENT_WEIGHT / MICROCAP_WEIGHT)
    for row in microcap_targets:
        if row["symbol"] in event_symbols:
            continue
        if micro_slots <= 0:
            break
        rank += 1
        targets.append(
            {
                **row,
                "rank": rank,
                "family": "main_board_microcap",
                "target_weight": MICROCAP_WEIGHT,
            }
        )
        micro_slots -= 1

    decision_payload = {
        "schema_version": STATE_SCHEMA,
        "signal_date": signal_date.isoformat(),
        "baseline_date": baseline_date.isoformat(),
        "research_result_sha256": RESULT_SHA256,
        "risk": {**risk, "features": last_risk},
        "weekly_rebalance": weekly_rebalance,
        "skipped_gap_dates": skipped_gap_dates,
        "targets": targets,
        "event_count": len(active_events),
        "microcap_count": sum(row["family"] == "main_board_microcap" for row in targets),
    }
    decision_id = hashlib.sha256(
        json.dumps(decision_payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    plan = {**decision_payload, "decision_id": decision_id}
    next_state = {
        "schema_version": STATE_SCHEMA,
        "baseline_date": baseline_date.isoformat(),
        "last_signal_date": signal_date.isoformat(),
        **risk,
        "active_events": active_events,
        "microcap_targets": microcap_targets,
        "last_decision_id": decision_id,
    }
    return plan, next_state


def build_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    cross_section = (
        panel.filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"),
        )
        .with_columns(
            (((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count")).floor())
            .clip(0, 9)
            .cast(pl.UInt8)
            .alias("cap_decile")
        )
        .group_by("date")
        .agg(
            pl.col("daily_return")
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_daily_return"),
            pl.col("daily_return").mean().alias("market_daily_return"),
            (pl.col("daily_return") > 0)
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_breadth"),
            pl.col("is_limit_down")
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_limit_down"),
            pl.col("amount")
            .filter(pl.col("cap_decile") == 0)
            .median()
            .alias("microcap_median_amount"),
        )
        .sort("date")
    )
    return (
        cross_section.with_columns(
            (
                (pl.col("microcap_daily_return") + 1.0).rolling_map(
                    lambda values: values.product(), window_size=5, min_samples=5
                )
                / (pl.col("market_daily_return") + 1.0).rolling_map(
                    lambda values: values.product(), window_size=5, min_samples=5
                )
                - 1.0
            ).alias("microcap_excess_5d"),
            pl.col("microcap_breadth")
            .rolling_mean(window_size=3, min_samples=3)
            .alias("microcap_breadth_3d"),
            pl.col("microcap_limit_down")
            .rolling_mean(window_size=3, min_samples=3)
            .alias("microcap_limit_down_3d"),
            (
                pl.col("microcap_median_amount").rolling_mean(5, min_samples=5)
                / pl.col("microcap_median_amount").rolling_mean(60, min_samples=60)
            ).alias("microcap_liquidity_5d_60d"),
        )
        .with_columns(
            (pl.col("microcap_excess_5d") <= THRESHOLDS["microcap_excess_5d_p10"])
            .fill_null(False)
            .alias("excess_alarm"),
            (pl.col("microcap_breadth_3d") <= THRESHOLDS["microcap_breadth_3d_p10"])
            .fill_null(False)
            .alias("breadth_alarm"),
            (pl.col("microcap_limit_down_3d") >= THRESHOLDS["microcap_limit_down_3d_p90"])
            .fill_null(False)
            .alias("limit_down_alarm"),
            (pl.col("microcap_liquidity_5d_60d") <= THRESHOLDS["microcap_liquidity_5d_60d_p10"])
            .fill_null(False)
            .alias("liquidity_alarm"),
            (pl.col("microcap_limit_down_3d") >= THRESHOLDS["microcap_limit_down_3d_p95"])
            .fill_null(False)
            .alias("severe_limit_down"),
        )
        .with_columns(
            pl.sum_horizontal(
                "excess_alarm",
                "breadth_alarm",
                "limit_down_alarm",
                "liquidity_alarm",
            ).alias("ordinary_alarm_count")
        )
    )


def advance_risk_state(
    state: dict[str, Any], feature: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk_on = bool(state.get("risk_on", True))
    off_days = int(state.get("off_days", 0))
    clean_days = int(state.get("clean_days", 0))
    count = int(feature.get("ordinary_alarm_count") or 0)
    severe = bool(feature.get("severe_limit_down"))
    switch = None
    if risk_on:
        if severe or count >= 2:
            risk_on = False
            off_days = 0
            clean_days = 0
            switch = "RISK_OFF"
    else:
        off_days += 1
        clean_days = clean_days + 1 if count == 0 else 0
        if off_days >= 5 and clean_days >= 3:
            risk_on = True
            off_days = 0
            clean_days = 0
            switch = "RISK_ON"
    audit = {
        "decision_date": str(feature["date"]),
        "risk_on": risk_on,
        "switch": switch,
        "ordinary_alarm_count": count,
        "severe_limit_down": severe,
        **{
            key: feature.get(key)
            for key in (
                "microcap_excess_5d",
                "microcap_breadth_3d",
                "microcap_limit_down_3d",
                "microcap_liquidity_5d_60d",
            )
        },
    }
    return {"risk_on": risk_on, "off_days": off_days, "clean_days": clean_days}, audit


def _load_recent_panel(data_dir: Path, signal_date: date) -> pl.DataFrame:
    start = signal_date - timedelta(days=180)
    paths = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= day <= signal_date:
            paths.append(path)
    if not paths:
        raise ValueError("缺少近 180 日 enriched 数据")
    panel = (
        pl.scan_parquet(sorted(paths))
        .select("symbol", "date", "open", "close", "volume", "amount", "raw_close")
        .collect(engine="streaming")
    )
    calendar = panel.select("date").unique().sort("date").with_row_index("trade_index")
    research = data_dir / "research"
    universe_path = research / "historical_stock_universe_all_a.parquet"
    names_path = research / "historical_stock_names_all_a.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError("缺少点时全 A 证券主表或历史名称")
    universe = (
        pl.read_parquet(universe_path)
        .with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "list_date", "delist_date")
    )
    names = (
        pl.read_parquet(names_path)
        .with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "name", "start_date", "end_date")
        .sort(["symbol", "start_date"])
    )
    shares_paths = sorted((data_dir / "financials" / "shares").glob("*.parquet"))
    if not shares_paths:
        raise ValueError("缺少点时股本数据")
    shares_source = pl.concat(
        [pl.read_parquet(path) for path in shares_paths], how="diagonal_relaxed"
    )
    announce = (
        pl.col("announce_date").cast(pl.Utf8).str.to_date(strict=False)
        if "announce_date" in shares_source.columns
        else pl.lit(None, dtype=pl.Date)
    )
    period = (
        pl.col("period_end").cast(pl.Utf8).str.to_date(strict=False)
        if "period_end" in shares_source.columns
        else pl.lit(None, dtype=pl.Date)
    )
    shares = (
        shares_source.with_columns(
            pl.coalesce(announce, period).alias("available_date"),
            period.alias("period_date"),
            pl.col("total_shares").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("available_date").is_not_null() & (pl.col("total_shares") > 0))
        .sort(["symbol", "available_date", "period_date"])
        .unique(subset=["symbol", "available_date"], keep="last")
        .select("symbol", "available_date", "total_shares")
        .sort(["symbol", "available_date"])
    )
    work = (
        panel.join(calendar, on="date", how="left")
        .join(universe, on="symbol", how="left")
        .filter(
            pl.col("list_date").is_not_null()
            & (pl.col("date") >= pl.col("list_date"))
            & (pl.col("delist_date").is_null() | (pl.col("date") <= pl.col("delist_date")))
            & ((pl.col("date") - pl.col("list_date")).dt.total_days() >= MIN_LISTING_DAYS)
        )
        .sort(["symbol", "date"])
        .join_asof(
            names,
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("name").is_not_null()
            & (pl.col("end_date").is_null() | (pl.col("date") <= pl.col("end_date")))
            & ~polars_is_risk_warning_name(pl.col("name"))
            & ~pl.col("name").str.contains("退", literal=True)
        )
        .sort(["symbol", "date"])
        .join_asof(
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(pl.col("total_shares") > 0)
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("raw_close")).alias("adjustment_factor"),
            pl.col("trade_index").shift(1).over("symbol").alias("previous_index"),
            pl.col("close").shift(1).over("symbol").alias("previous_close"),
            pl.col("raw_close").shift(1).over("symbol").alias("previous_raw_close"),
            (
                pl.col("close").shift(1).over("symbol")
                / pl.col("raw_close").shift(1).over("symbol")
            ).alias("previous_adjustment_factor"),
            polars_price_limit_pct(
                pl.col("symbol"),
                pl.col("date"),
                polars_is_risk_warning_name(pl.col("name")),
            ).alias("limit_pct"),
        )
        .with_columns(
            (pl.col("trade_index") == pl.col("previous_index") + 1).alias("adjacent"),
            (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
            pl.when(pl.col("trade_index") == pl.col("previous_index") + 1)
            .then(pl.col("close") / pl.col("previous_close") - 1.0)
            .otherwise(None)
            .alias("daily_return"),
            pl.when(pl.col("trade_index") == pl.col("previous_index") + 1)
            .then(
                pl.when(
                    (pl.col("adjustment_factor") - pl.col("previous_adjustment_factor")).abs()
                    > 1e-6
                )
                .then(pl.col("previous_close"))
                .otherwise(pl.col("previous_raw_close"))
            )
            .otherwise(None)
            .alias("reference_close"),
        )
        .with_columns(
            polars_limit_price(pl.col("reference_close"), pl.col("limit_pct"), up=False).alias(
                "limit_down_price"
            ),
        )
        .with_columns(
            (
                pl.col("adjacent") & (pl.col("raw_close") <= pl.col("limit_down_price") + 0.005)
            ).alias("is_limit_down")
        )
    )
    return work


def _microcap_targets(current: pl.DataFrame) -> list[dict[str, Any]]:
    rows = (
        current.filter(
            pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
            & (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").alias("cap_rank"),
        )
        .with_columns(
            (((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count")).floor())
            .clip(0, 9)
            .cast(pl.UInt8)
            .alias("cap_decile")
        )
        .filter(pl.col("cap_decile") == 0)
        .sort(["cap_rank", "symbol"])
        .select(
            "symbol",
            "name",
            pl.col("amount").alias("signal_amount"),
            "market_cap",
            "cap_rank",
        )
        .to_dicts()
    )
    return rows


def _idiosyncratic_events_for_date(
    data_dir: Path, signal_date: date, current: pl.DataFrame
) -> list[dict[str, Any]]:
    path = data_dir / "event_data" / "forecast" / f"year={signal_date.year}" / "part.parquet"
    if not path.exists():
        return []
    events = pl.read_parquet(path).filter(pl.col("ann_date") == signal_date)
    if events.is_empty():
        return []
    event_type = pl.col("type").fill_null("")
    is_first = pl.col("first_ann_date").is_null() | (pl.col("first_ann_date") == pl.col("ann_date"))
    positive_profit = pl.col("net_profit_min").fill_null(0) > 0
    turnaround = event_type.str.contains("扭亏", literal=True) & positive_profit
    negative = (
        pl.col("p_change_max").is_not_null() & (pl.col("p_change_max") < 0)
    ) | event_type.str.contains(r"(?:预减|首亏|续亏|略减)")
    category = (
        pl.when(turnaround)
        .then(pl.lit("turnaround"))
        .when(
            positive_profit & pl.col("p_change_min").is_not_null() & (pl.col("p_change_min") >= 100)
        )
        .then(pl.lit("growth_ge_100"))
        .when(positive_profit & pl.col("p_change_min").is_between(50, 100, closed="left"))
        .then(pl.lit("growth_50_100"))
        .when(positive_profit & pl.col("p_change_min").is_between(0, 50, closed="left"))
        .then(pl.lit("growth_0_50"))
        .when(negative)
        .then(pl.lit("negative_control"))
        .otherwise(None)
    )
    priority = pl.when(turnaround).then(4).when(negative).then(0).otherwise(1)
    events = (
        events.filter(is_first)
        .with_columns(
            category.alias("_source_category"),
            priority.alias("_priority"),
        )
        .filter(pl.col("_source_category").is_not_null())
        .sort(
            ["symbol", "ann_date", "_priority", "p_change_min", "period_end"],
            descending=[False, False, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["symbol", "ann_date"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("_source_category")
            .is_in(["turnaround", "growth_ge_100", "growth_50_100", "growth_0_50"])
            .cast(pl.Int64)
            .alias("_positive"),
            pl.col("_source_category").is_in(["growth_0_50", "growth_50_100"]).alias("_target"),
        )
        .filter(pl.col("symbol").str.contains(MAIN_BOARD_PATTERN))
    )
    membership_path = data_dir / "research" / "sw_l1_membership.parquet"
    if not membership_path.exists():
        raise ValueError("缺少点时申万一级行业归属")
    membership = (
        pl.read_parquet(membership_path)
        .with_columns(
            pl.col("in_date").cast(pl.Date, strict=False),
            pl.col("out_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "l1_code", "in_date", "out_date")
        .sort(["symbol", "in_date"])
    )
    events = (
        events.sort(["symbol", "ann_date"])
        .join_asof(
            membership,
            left_on="ann_date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("l1_code").is_not_null()
            & (pl.col("out_date").is_null() | (pl.col("ann_date") <= pl.col("out_date")))
        )
    )
    if events.is_empty():
        return []
    industry = events.group_by("l1_code").agg(
        pl.len().alias("_industry_count"),
        pl.col("_positive").sum().alias("_industry_positive"),
    )
    market_count = events.height
    market_positive = int(events.get_column("_positive").sum())
    scored = (
        events.join(industry, on="l1_code", how="left")
        .with_columns(
            (pl.col("_industry_count") - 1).alias("industry_peer_count"),
            (
                (pl.col("_industry_positive") - pl.col("_positive"))
                / (pl.col("_industry_count") - 1)
            ).alias("industry_peer_positive_share"),
            ((market_positive - pl.col("_positive")) / max(market_count - 1, 1)).alias(
                "market_peer_positive_share"
            ),
        )
        .with_columns(
            (pl.col("industry_peer_positive_share") - pl.col("market_peer_positive_share")).alias(
                "industry_positive_share_excess"
            )
        )
        .filter(
            pl.col("_target")
            & (pl.col("industry_peer_count") >= 5)
            & (pl.col("industry_peer_positive_share") <= 0.40)
            & (pl.col("industry_positive_share_excess") <= -0.10)
        )
        .join(
            current.select(
                "symbol",
                "name",
                "raw_close",
                "amount",
                "market_cap",
            ),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("amount") >= 50_000_000.0)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .sort(
            ["p_change_min", "p_change_max", "symbol"],
            descending=[True, True, False],
            nulls_last=True,
        )
        .select(
            "symbol",
            "name",
            pl.col("ann_date").cast(pl.Utf8),
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
        )
    )
    return scored.to_dicts()


def _bootstrap_state(data_dir: Path, baseline_date: date) -> dict[str, Any]:
    result_path = _require_frozen_result(data_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    decisions = payload["risk"]["known_stress"]["decisions"]
    if not decisions:
        raise ValueError("冻结研究结果缺少风险状态审计")
    last = decisions[-1]
    switch_index = max(
        (index for index, row in enumerate(decisions) if row.get("switch")),
        default=-1,
    )
    if switch_index >= 0:
        start = decisions[switch_index]
        risk = {
            "risk_on": bool(start["risk_on"]),
            "off_days": 0,
            "clean_days": 0,
        }
        for row in decisions[switch_index + 1 :]:
            risk, _ = advance_risk_state(
                risk,
                {
                    "date": row["decision_date"],
                    "ordinary_alarm_count": row["ordinary_alarm_count"],
                    "severe_limit_down": row["severe_limit_down"],
                },
            )
    else:
        risk = {"risk_on": bool(last["risk_on"]), "off_days": 0, "clean_days": 0}
    return {
        "schema_version": STATE_SCHEMA,
        "baseline_date": baseline_date.isoformat(),
        "last_signal_date": last["decision_date"],
        **risk,
        "active_events": [],
        "microcap_targets": [],
        "last_decision_id": None,
    }


def _require_frozen_result(data_dir: Path) -> Path:
    result_path = data_dir / "research" / RESULT_FILE
    if (
        not result_path.exists()
        or hashlib.sha256(result_path.read_bytes()).hexdigest() != RESULT_SHA256
    ):
        raise ValueError("冻结研究结果缺失或哈希不一致，拒绝启动前向账户")
    return result_path


def _require_forecast_receipt(data_dir: Path, signal_date: date) -> None:
    path = data_dir / "event_data" / "forecast" / "sync_status.json"
    if not path.exists():
        raise ValueError("业绩预告增量采集尚无成功回执")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    covered = _as_date(receipt.get("end_date"))
    if covered is None or covered < signal_date:
        raise ValueError(f"业绩预告只同步到 {covered}，未覆盖信号日 {signal_date}")


def _state_path(data_dir: Path) -> Path:
    return data_dir / "research" / "forward" / STRATEGY_ID / "state.json"


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATE_SCHEMA:
        raise ValueError("前向状态版本不匹配")
    return payload


def _write_decision(data_dir: Path, plan: dict[str, Any]) -> None:
    path = (
        data_dir
        / "research"
        / "forward"
        / STRATEGY_ID
        / "decisions"
        / f"{plan['signal_date']}.json"
    )
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("decision_id") != plan["decision_id"]:
            raise ValueError("同一信号日的不可变决策哈希冲突")
        return
    _atomic_json(path, plan)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])
