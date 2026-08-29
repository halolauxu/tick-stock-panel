"""Forward-only Alpha candidate accounts backed by the event-driven paper ledger."""
# Requirements: AM-S8-001 through AM-S8-010.
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

from app.alpha_mining.config_store import AlphaConfigStore
from app.alpha_mining.evidence import AlphaEvidenceStore
from app.alpha_mining.lifecycle import is_strict_full_history_request


class AlphaShadowService:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.evidence = AlphaEvidenceStore(self.data_dir)
        self.config = AlphaConfigStore(self.data_dir)

    def start(self, candidate_id: str, paper_service, baseline_date: date) -> dict[str, Any]:
        candidate = self.evidence.get_candidate(candidate_id)
        if candidate["state"]["state"] != "research_candidate":
            raise ValueError("只有通过全部历史与压力门槛的研究候选可以进入前向模拟")
        experiment = self.evidence.read_experiment(candidate["run_id"])
        request = dict(experiment["contract"].get("request") or {})
        if not is_strict_full_history_request(self.data_dir, request):
            raise ValueError("候选没有完成全部可用历史的严格验证; 不能进入前向模拟")
        frozen = dict(candidate["candidate"])
        definition = dict(frozen.get("definition") or {})
        if definition.get("kind") != "factor_rank":
            raise ValueError("当前前向账户只接受公共factor_rank执行合同")
        parameters = dict(definition.get("parameters") or {})
        params = {
            **parameters,
            "scoring": dict(definition.get("scoring") or {}),
            "directions": dict(definition.get("directions") or {}),
        }
        account_id = f"alpha-{candidate_id[3:15]}"
        try:
            account = paper_service.ledger.get_account(account_id)
        except KeyError:
            account = paper_service.ledger.create_account(
                name=f"Alpha前向 · {candidate['engine_id']}",
                baseline_date=baseline_date,
                account_id=account_id,
                config={
                    "strategy_id": "factor_rank_research",
                    "strategy_name": f"Alpha候选 {candidate_id}",
                    "asset_type": request.get("asset_type", "stock"),
                    "symbols": request.get("symbols"),
                    "params": params,
                    "overrides": {},
                    "entry_fill": "open_t+1",
                    "exit_fill": "open_t+1",
                    "commission_pct": float(request.get("commission_pct", 0.0002)),
                    "stamp_tax_pct": float(request.get("stamp_tax_pct", 0.0005)),
                    "slippage_bps": float(request.get("slippage_bps", 5.0)),
                    "max_positions": int(request.get("max_positions", 10)),
                    "max_exposure_pct": 1.0,
                    "initial_capital": 1_000_000.0,
                    "position_sizing": "equal",
                    "holding_days": int(request.get("forward_horizon", 5)),
                    "minute_fill": False,
                    "exit_mode": "eod",
                    "enforce_t_plus_one": True,
                    "alpha_candidate_id": candidate_id,
                    "alpha_candidate_sha256": candidate["content_sha256"],
                },
            )
        receipt = self.evidence.write_shadow_receipt(candidate_id, {
            "paper_account_id": account_id,
            "baseline_date": baseline_date.isoformat(),
            "candidate_sha256": candidate["content_sha256"],
            "execution_policy": "event_driven_open_t_plus_one",
        })
        if candidate["state"]["state"] == "research_candidate":
            self.evidence.transition(candidate_id, "shadow", {
                "paper_account_id": account_id,
                "receipt_sha256": receipt["content_sha256"],
            })
        return {"candidate": self.evidence.get_candidate(candidate_id), "account": account}

    def status(self, candidate_id: str, paper_service) -> dict[str, Any]:
        candidate = self.evidence.get_candidate(candidate_id)
        shadow = candidate.get("shadow")
        if not shadow:
            raise ValueError("候选尚未创建前向账户")
        account_id = shadow["receipt"]["paper_account_id"]
        account = paper_service.ledger.get_account(account_id)
        account["reconcile"] = paper_service.ledger.reconcile(
            account_id,
            open_incident=False,
        )
        evaluation = self._evaluate_account(account)
        return {"candidate": candidate, "account": account, "evaluation": evaluation}

    def evaluate_and_advance(self, candidate_id: str, paper_service) -> dict[str, Any]:
        status = self.status(candidate_id, paper_service)
        account = status["account"]
        evaluation = status["evaluation"]
        state = status["candidate"]["state"]["state"]
        if evaluation["drift_detected"] and state == "shadow":
            paper_service.ledger.set_status(account["id"], "paused")
            self.evidence.write_forward_evaluation(candidate_id, {
                **evaluation,
                "verdict": "failed",
                "action": "paused_without_retuning",
            })
            self.evidence.transition(candidate_id, "rejected", {
                "reason": "forward_drift",
                "paper_account_id": account["id"],
            })
        elif evaluation["qualified"] and state == "shadow":
            self.evidence.write_forward_evaluation(candidate_id, {
                **evaluation,
                "verdict": "passed",
                "action": "advance_to_challenger",
            })
            self.evidence.transition(candidate_id, "challenger", {
                "paper_account_id": account["id"],
                "trading_days": evaluation["trading_days"],
                "fills": evaluation["fills"],
            })
        return self.status(candidate_id, paper_service)

    def _evaluate_account(self, account: dict[str, Any]) -> dict[str, Any]:
        settings = self.config.get()
        fills = list(account.get("fills") or [])
        orders = list(account.get("orders") or [])
        signals = list(account.get("signals") or [])
        nav = list(account.get("nav") or [])
        incidents = list(account.get("incidents") or [])
        reconcile = account.get("reconcile")
        if reconcile is None:
            reconcile = {"ok": not any(item.get("severity") == "critical" for item in incidents)}
        configured_slippage = float(account.get("config", {}).get("slippage_bps", 5.0))
        observed_slippage = []
        for fill in fills:
            reference = _number(fill.get("reference_price"))
            price = _number(fill.get("price"))
            if reference and price and reference > 0:
                observed_slippage.append(abs(price / reference - 1.0) * 10_000.0)
        average_slippage = (
            sum(observed_slippage) / len(observed_slippage) if observed_slippage else None
        )
        total_return = float(account.get("summary", {}).get("total_return") or 0.0)
        values = [float(item["value"]) for item in nav if _number(item.get("value"))]
        max_drawdown = _max_drawdown(values)
        no_synthetic_profit = bool(fills) or abs(total_return) < 1e-12
        signal_order_parity = all(
            signal.get("order_id") or signal.get("skipped")
            for signal in signals
        ) and all(
            order.get("signal_id") and order.get("signal_date")
            for order in orders
        )
        factor_decay = _factor_decay(
            fills,
            orders,
            min_round_trips=int(settings["shadow_min_factor_round_trips"]),
            min_rank_ic=float(settings["shadow_min_rank_ic"]),
        )
        critical = sum(
            item.get("status") == "open" and item.get("severity") == "critical"
            for item in incidents
        )
        enough = (
            len(nav) >= int(settings["shadow_min_trading_days"])
            and len(fills) >= int(settings["shadow_min_fills"])
        )
        drift = bool(
            enough
            and (
                critical
                or not bool(reconcile.get("ok", False))
                or not no_synthetic_profit
                or not signal_order_parity
                or factor_decay["status"] == "failed"
                or (
                    average_slippage is not None
                    and average_slippage > max(configured_slippage * 2.0, configured_slippage + 5.0)
                )
                or total_return <= -0.10
            )
        )
        qualified = bool(
            enough
            and not drift
            and total_return > 0
            and max_drawdown >= -0.25
            and no_synthetic_profit
            and signal_order_parity
            and factor_decay["status"] == "passed"
        )
        return {
            "trading_days": len(nav),
            "fills": len(fills),
            "orders": len(orders),
            "signals": len(signals),
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "average_slippage_bps": average_slippage,
            "reconcile_ok": bool(reconcile.get("ok", False)),
            "signal_order_parity": signal_order_parity,
            "factor_decay": factor_decay,
            "no_synthetic_profit": no_synthetic_profit,
            "critical_incidents": critical,
            "drift_detected": drift,
            "qualified": qualified,
            "required_trading_days": int(settings["shadow_min_trading_days"]),
            "required_fills": int(settings["shadow_min_fills"]),
            "required_factor_round_trips": int(settings["shadow_min_factor_round_trips"]),
        }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _factor_decay(
    fills: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    *,
    min_round_trips: int,
    min_rank_ic: float,
) -> dict[str, Any]:
    """Measure whether frozen entry scores still rank realised forward returns."""
    order_by_id = {str(item.get("id")): item for item in orders}
    open_lots: dict[str, list[dict[str, float]]] = {}
    observations: list[tuple[float, float]] = []
    for fill in sorted(fills, key=lambda item: (str(item.get("executed_at") or ""), str(item.get("id") or ""))):
        order = order_by_id.get(str(fill.get("order_id")))
        if not order:
            continue
        symbol = str(fill.get("symbol") or order.get("symbol") or "")
        quantity = int(fill.get("quantity") or 0)
        price = _number(fill.get("price"))
        fee = _number(fill.get("fee_amount")) or 0.0
        if not symbol or quantity <= 0 or not price or price <= 0:
            continue
        if str(fill.get("side")) == "BUY":
            score = _number(order.get("score"))
            if score is None:
                continue
            open_lots.setdefault(symbol, []).append({
                "quantity": float(quantity),
                "cost_per_share": (price * quantity + fee) / quantity,
                "score": score,
            })
            continue
        remaining = float(quantity)
        net_per_share = (price * quantity - fee) / quantity
        lots = open_lots.setdefault(symbol, [])
        while remaining > 0 and lots:
            lot = lots[0]
            matched = min(remaining, lot["quantity"])
            if lot["cost_per_share"] > 0:
                observations.append((
                    lot["score"],
                    net_per_share / lot["cost_per_share"] - 1.0,
                ))
            lot["quantity"] -= matched
            remaining -= matched
            if lot["quantity"] <= 1e-9:
                lots.pop(0)
    rank_ic = _spearman(observations)
    if len(observations) < min_round_trips or rank_ic is None:
        status = "pending"
    elif rank_ic >= min_rank_ic:
        status = "passed"
    else:
        status = "failed"
    return {
        "status": status,
        "completed_round_trips": len(observations),
        "rank_ic": rank_ic,
        "minimum_rank_ic": min_rank_ic,
        "minimum_round_trips": min_round_trips,
    }


def _spearman(observations: list[tuple[float, float]]) -> float | None:
    if len(observations) < 2:
        return None
    left = _ranks([item[0] for item in observations])
    right = _ranks([item[1] for item in observations])
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator > 0 else None


def _ranks(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            ranked[original] = average
        index = end
    return ranked
