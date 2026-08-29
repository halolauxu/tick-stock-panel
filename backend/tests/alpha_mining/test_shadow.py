from __future__ import annotations

# Requirements: AM-S8-001 through AM-S8-010.
from datetime import date, timedelta

import pytest

from app.alpha_mining.contracts import CandidateSpec, FrozenSignalSpec
from app.alpha_mining.providers import DeclarativeCandidateRenderer
from app.alpha_mining.shadow import AlphaShadowService, _factor_decay
from app.services.paper_ledger import PaperLedger


def _completed_round_trips(*, inverse: bool = False) -> tuple[list[dict], list[dict]]:
    fills: list[dict] = []
    orders: list[dict] = []
    for index in range(10):
        buy_id = f"buy-{index}"
        sell_id = f"sell-{index}"
        score = float(index + 1)
        gain = (10 - index if inverse else index + 1) / 100.0
        orders.extend((
            {"id": buy_id, "score": score, "symbol": f"{index:06d}.SZ"},
            {"id": sell_id, "score": None, "symbol": f"{index:06d}.SZ"},
        ))
        fills.extend((
            {
                "id": buy_id,
                "order_id": buy_id,
                "symbol": f"{index:06d}.SZ",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0,
                "fee_amount": 0.0,
                "executed_at": f"2026-01-{index + 1:02d}T09:30:00",
            },
            {
                "id": sell_id,
                "order_id": sell_id,
                "symbol": f"{index:06d}.SZ",
                "side": "SELL",
                "quantity": 100,
                "price": 10.0 * (1.0 + gain),
                "fee_amount": 0.0,
                "executed_at": f"2026-02-{index + 1:02d}T09:30:00",
            },
        ))
    return fills, orders


def test_factor_decay_passes_monotonic_scores_and_rejects_inversion() -> None:
    fills, orders = _completed_round_trips()
    passed = _factor_decay(fills, orders, min_round_trips=10, min_rank_ic=0.02)
    assert passed["status"] == "passed"
    assert passed["rank_ic"] == 1.0

    inverse_fills, inverse_orders = _completed_round_trips(inverse=True)
    failed = _factor_decay(
        inverse_fills,
        inverse_orders,
        min_round_trips=10,
        min_rank_ic=0.02,
    )
    assert failed["status"] == "failed"
    assert failed["rank_ic"] == -1.0


def test_forward_evaluation_requires_signal_lifecycle_and_factor_decay(tmp_path) -> None:
    service = AlphaShadowService(tmp_path)
    service.config.update({
        "shadow_min_trading_days": 20,
        "shadow_min_fills": 20,
        "shadow_min_factor_round_trips": 10,
    })
    fills, orders = _completed_round_trips()
    for order in orders:
        order.update({"signal_id": f"signal-{order['id']}", "signal_date": "2026-01-01"})
    account = {
        "fills": fills,
        "orders": orders,
        "signals": [
            {"id": f"signal-{order['id']}", "order_id": order["id"], "skipped": False}
            for order in orders
        ],
        "nav": [
            {"date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(), "value": 1_000_000 + index * 1_000}
            for index in range(20)
        ],
        "incidents": [],
        "reconcile": {"ok": True},
        "config": {"slippage_bps": 5.0},
        "summary": {"total_return": 0.02},
    }
    evaluation = service._evaluate_account(account)
    assert evaluation["qualified"] is True
    assert evaluation["factor_decay"]["status"] == "passed"

    account["signals"][0]["order_id"] = None
    broken = service._evaluate_account(account)
    assert broken["signal_order_parity"] is False
    assert broken["drift_detected"] is True


def test_paper_projection_exposes_ordered_and_skipped_signal_lifecycle(tmp_path) -> None:
    ledger = PaperLedger(tmp_path)
    ledger.create_account(
        account_id="alpha-test",
        name="Alpha test",
        baseline_date=date(2026, 1, 5),
        config={"initial_capital": 200_000.0, "strategy_id": "factor_rank_research"},
    )
    ledger.record_signal_and_order(
        account_id="alpha-test",
        strategy_id="factor_rank_research",
        symbol="000001.SZ",
        name="测试一",
        side="BUY",
        signal_date=date(2026, 1, 5),
        score=88.0,
        reason="score",
        signal_ref="ref-1",
        requested_qty=100,
        target_amount=1_000.0,
        target_weight=0.1,
        planned_session="NEXT_OPEN",
    )
    ledger.record_skipped_signal(
        account_id="alpha-test",
        strategy_id="factor_rank_research",
        symbol="000002.SZ",
        name="测试二",
        side="BUY",
        signal_date=date(2026, 1, 5),
        score=77.0,
        reason="score",
        signal_ref="ref-2",
        skip_code="INSUFFICIENT_BUYING_POWER",
        detail="零手信号保留",
    )
    account = ledger.get_account("alpha-test")
    assert len(account["signals"]) == 2
    assert all(item["order_id"] or item["skipped"] for item in account["signals"])


def test_shadow_account_reuses_frozen_research_execution_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.alpha_mining.shadow.is_strict_full_history_request",
        lambda *_args: True,
    )
    service = AlphaShadowService(tmp_path)
    service.config.update({"enabled": True})
    service.evidence.create_experiment("alpha-shadow", {
        "request": {
            "asset_type": "stock",
            "forward_horizon": 10,
            "commission_pct": 0.0003,
            "stamp_tax_pct": 0.0005,
            "slippage_bps": 8.0,
            "max_positions": 12,
        }
    })
    spec = CandidateSpec(
        recipe_id="shadow.factor",
        engine_id="cross_sectional_rank",
        engine_version="1.0.0",
        name="Shadow",
        thesis="parity",
        signal_kind="factor_rank",
        features=("momentum_5d",),
        directions=(1,),
        weights=(1.0,),
        parameters={"entry_score": 75.0, "exit_score": 35.0, "top_rank": 15},
        train_evidence={"ic_mean": 0.1},
    )
    frozen = FrozenSignalSpec.from_candidate(spec)
    candidate = service.evidence.freeze_candidate(
        run_id="alpha-shadow",
        engine_id=spec.engine_id,
        candidate=frozen.to_dict(),
        renderer=dict(DeclarativeCandidateRenderer().render(frozen)),
    )
    service.evidence.record_outer_evaluation(
        candidate["candidate_id"],
        {"gates": [], "metrics": {}},
        "research_candidate",
    )

    class Ledger:
        created = None

        def get_account(self, _account_id):
            raise KeyError

        def create_account(self, **kwargs):
            self.created = kwargs
            return {"id": kwargs["account_id"], "config": kwargs["config"]}

    ledger = Ledger()
    paper_service = type("Paper", (), {"ledger": ledger})()
    result = service.start(candidate["candidate_id"], paper_service, date(2026, 1, 5))
    config = ledger.created["config"]
    assert config["strategy_id"] == "factor_rank_research"
    assert config["params"]["scoring"] == {"momentum_5d": 1.0}
    assert config["params"]["directions"] == {"momentum_5d": "high"}
    assert config["entry_fill"] == config["exit_fill"] == "open_t+1"
    assert config["commission_pct"] == 0.0003
    assert config["slippage_bps"] == 8.0
    assert result["candidate"]["state"]["state"] == "shadow"


def test_shadow_rejects_candidate_without_full_history_strict_contract(tmp_path) -> None:
    service = AlphaShadowService(tmp_path)
    service.evidence.create_experiment("alpha-not-strict", {
        "request": {"asset_type": "stock", "budget_profile": "exploratory"},
    })
    spec = CandidateSpec(
        recipe_id="not-strict.factor",
        engine_id="cross_sectional_rank",
        engine_version="1.0.0",
        name="Not strict",
        thesis="must be blocked",
        signal_kind="factor_rank",
        features=("momentum_5d",),
        directions=(1,),
        weights=(1.0,),
        parameters={"top_rank": 20},
        train_evidence={},
    )
    frozen = FrozenSignalSpec.from_candidate(spec)
    candidate = service.evidence.freeze_candidate(
        run_id="alpha-not-strict",
        engine_id=spec.engine_id,
        candidate=frozen.to_dict(),
        renderer=dict(DeclarativeCandidateRenderer().render(frozen)),
    )
    service.evidence.record_outer_evaluation(
        candidate["candidate_id"], {"gates": [], "metrics": {}}, "research_candidate",
    )

    with pytest.raises(ValueError, match="全部可用历史"):
        service.start(candidate["candidate_id"], object(), date(2026, 1, 5))
