from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import risk_admitted_forecast_paper as strategy
from app.services.paper_ledger import PaperLedger


class _PaperService:
    def __init__(self, data_dir, *, latest_enriched: date | None = None) -> None:
        self.repo = SimpleNamespace(
            store=SimpleNamespace(data_dir=data_dir),
            latest_enriched_date=lambda _asset_type: latest_enriched,
        )
        self.ledger = PaperLedger(data_dir)


def test_risk_clock_requires_two_alarms_and_three_clean_days_after_minimum_off() -> None:
    state = {"risk_on": True, "off_days": 0, "clean_days": 0}
    state, audit = strategy.advance_risk_state(
        state,
        {
            "date": date(2026, 8, 20),
            "ordinary_alarm_count": 2,
            "severe_limit_down": False,
        },
    )
    assert state == {"risk_on": False, "off_days": 0, "clean_days": 0}
    assert audit["switch"] == "RISK_OFF"

    for offset, alarms in enumerate((1, 1, 0, 0, 0), start=1):
        state, audit = strategy.advance_risk_state(
            state,
            {
                "date": date(2026, 8, 20 + offset),
                "ordinary_alarm_count": alarms,
                "severe_limit_down": False,
            },
        )

    assert state == {"risk_on": True, "off_days": 0, "clean_days": 0}
    assert audit["switch"] == "RISK_ON"


def test_forward_account_is_idempotent_and_freezes_contract(tmp_path, monkeypatch) -> None:
    service = _PaperService(tmp_path)
    monkeypatch.setattr(strategy, "_require_result", lambda *_args, **_kwargs: tmp_path)

    created = strategy.ensure_account(service, date(2026, 9, 3))
    repeated = strategy.ensure_account(service, date(2026, 9, 4))

    assert created["id"] == repeated["id"] == strategy.ACCOUNT_ID
    assert created["baseline_date"] == "2026-09-03"
    assert repeated["config"]["research_result_sha256"] == strategy.RESULT_SHA256
    assert repeated["config"]["position_sizing"] == "frozen_target_weight"
    assert len(service.ledger.list_accounts()) == 1


def test_forward_account_rejects_any_frozen_contract_drift(tmp_path, monkeypatch) -> None:
    service = _PaperService(tmp_path)
    monkeypatch.setattr(strategy, "_require_result", lambda *_args, **_kwargs: tmp_path)
    service.ledger.create_account(
        name=strategy.ACCOUNT_NAME,
        baseline_date=date(2026, 9, 3),
        account_id=strategy.ACCOUNT_ID,
        config={
            **strategy._FROZEN_ACCOUNT_CONTRACT,
            "commission_pct": 0.0,
        },
    )

    with pytest.raises(ValueError, match="账户合同与冻结策略不一致"):
        strategy.ensure_account(service, date(2026, 9, 4))


def test_managed_account_dispatches_to_dedicated_sealer(tmp_path, monkeypatch) -> None:
    from app.services.paper_trading import PaperTradingService

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    service = PaperTradingService(SimpleNamespace(repo=repo))
    account = service.ledger.create_account(
        name="专用前向账户",
        baseline_date=date(2026, 9, 3),
        account_id=strategy.ACCOUNT_ID,
        config={
            "strategy_id": strategy.STRATEGY_ID,
            "initial_capital": strategy.INITIAL_CAPITAL,
        },
    )
    called: list[tuple[str, date]] = []
    monkeypatch.setattr(
        strategy,
        "seal_account",
        lambda _service, account_id, signal_date: (
            called.append((account_id, signal_date)) or {"signals": 3, "orders": 2}
        ),
    )

    result = service.seal_account_signals(account["id"], date(2026, 9, 3))

    assert result == {"signals": 3, "orders": 2}
    assert called == [(strategy.ACCOUNT_ID, date(2026, 9, 3))]


def test_managed_snapshot_explains_waiting_pipeline_and_provenance(tmp_path, monkeypatch) -> None:
    service = _PaperService(tmp_path, latest_enriched=date(2026, 9, 3))
    monkeypatch.setattr(strategy, "_require_result", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(strategy, "_pipeline_schedule", lambda: {"hour": 21, "minute": 0})
    strategy.ensure_account(service, date(2026, 9, 3))
    strategy._atomic_json(
        tmp_path / "event_data" / "forecast" / "sync_status.json",
        {"end_date": "2026-09-03"},
    )

    snapshot = strategy.managed_strategy_snapshot(
        service,
        now=datetime(2026, 9, 4, 16, 0, tzinfo=CN_TZ),
    )

    assert snapshot["id"] == strategy.STRATEGY_ID
    assert snapshot["account_id"] == strategy.ACCOUNT_ID
    assert snapshot["provenance"]["introduced_commit"] == "1f2ef35"
    assert snapshot["provenance"]["artifact_verified"] is False
    assert snapshot["live"]["lifecycle"]["code"] == "WAITING_PIPELINE"
    assert snapshot["live"]["lifecycle"]["next_action"] == "21:00 自动同步并封板"
    assert snapshot["historical_results"][0]["label"] == "2021–2023 独立验证"


def test_managed_snapshot_distinguishes_delayed_data_from_waiting_signal(tmp_path, monkeypatch) -> None:
    service = _PaperService(tmp_path, latest_enriched=date(2026, 9, 3))
    monkeypatch.setattr(strategy, "_require_result", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(strategy, "_pipeline_schedule", lambda: {"hour": 21, "minute": 0})
    strategy.ensure_account(service, date(2026, 9, 3))

    delayed = strategy.managed_strategy_snapshot(
        service,
        now=datetime(2026, 9, 4, 21, 5, tzinfo=CN_TZ),
    )
    assert delayed["live"]["lifecycle"]["code"] == "DATA_DELAYED"

    service.repo.latest_enriched_date = lambda _asset_type: date(2026, 9, 4)
    strategy._atomic_json(
        tmp_path / "event_data" / "forecast" / "sync_status.json",
        {"end_date": "2026-09-04"},
    )
    waiting_seal = strategy.managed_strategy_snapshot(
        service,
        now=datetime(2026, 9, 4, 21, 5, tzinfo=CN_TZ),
    )
    assert waiting_seal["live"]["lifecycle"]["code"] == "SEAL_OVERDUE"


def test_managed_sealer_rebalances_retained_position_to_frozen_weight(
    tmp_path, monkeypatch
) -> None:
    service = _PaperService(tmp_path)
    account = service.ledger.create_account(
        name=strategy.ACCOUNT_NAME,
        baseline_date=date(2026, 9, 3),
        account_id=strategy.ACCOUNT_ID,
        config={
            "strategy_id": strategy.STRATEGY_ID,
            "strategy_name": strategy.ACCOUNT_NAME,
            "asset_type": "stock",
            "initial_capital": strategy.INITIAL_CAPITAL,
            "commission_pct": 0.0002,
            "stamp_tax_pct": 0.0005,
            "slippage_bps": 5.0,
        },
    )
    _, order_id, _ = service.ledger.record_signal_and_order(
        account_id=account["id"], strategy_id=strategy.STRATEGY_ID,
        symbol="600000.SH", name="浦发银行", side="BUY",
        signal_date=date(2026, 9, 3), score=1, reason="initial_entry",
        signal_ref="initial", requested_qty=1_000, target_amount=10_000,
        target_weight=0.05, planned_session="NEXT_OPEN",
        payload={"family": "main_board_microcap"},
    )
    service.ledger.assign_due_date(order_id, date(2026, 9, 4), {})
    service.ledger.execute_fill(
        order_id, price=10, quantity=1_000,
        quote_at=datetime(2026, 9, 4, 9, 30, tzinfo=CN_TZ), source="open_quote",
    )
    service.ledger.mark_signal_day(account["id"], date(2026, 9, 3))
    plan = {
        "signal_date": "2026-09-04",
        "decision_id": "decision-1",
        "targets": [{
            "symbol": "600000.SH", "name": "浦发银行", "rank": 1,
            "family": "main_board_microcap", "target_weight": 0.20,
            "signal_amount": 10_000_000,
        }],
        "risk": {"risk_on": True},
        "event_count": 0,
        "microcap_count": 1,
        "weekly_rebalance": True,
        "recovered_gap_dates": [],
    }
    next_state = {
        "schema_version": strategy.STATE_SCHEMA,
        "baseline_date": "2026-09-03",
        "last_signal_date": "2026-09-04",
        "last_decision_id": "decision-1",
        "risk_on": True,
        "off_days": 0,
        "clean_days": 0,
        "active_events": [],
        "microcap_targets": [],
    }
    monkeypatch.setattr(strategy, "_load_state", lambda _path: {
        **next_state, "last_signal_date": "2026-09-03", "last_decision_id": "prior"
    })
    monkeypatch.setattr(strategy, "build_forward_plan", lambda *_args, **_kwargs: (plan, next_state))

    result = strategy.seal_account(service, account["id"], date(2026, 9, 4))
    current = service.ledger.get_account(account["id"])

    assert result == {"signals": 1, "orders": 1}
    rebalance = next(row for row in current["orders"] if row["signal_date"] == "2026-09-04")
    assert rebalance["side"] == "BUY"
    assert rebalance["requested_qty"] == 0
    assert rebalance["target_amount"] == pytest.approx(39_998.60)


def test_managed_sealer_recovers_state_ahead_of_account_checkpoint(tmp_path) -> None:
    service = _PaperService(tmp_path)
    account = service.ledger.create_account(
        name=strategy.ACCOUNT_NAME,
        baseline_date=date(2026, 9, 3),
        account_id=strategy.ACCOUNT_ID,
        config={**strategy._FROZEN_ACCOUNT_CONTRACT, "strategy_name": strategy.ACCOUNT_NAME},
    )
    signal_day = date(2026, 9, 4)
    state = {
        "schema_version": strategy.STATE_SCHEMA,
        "baseline_date": "2026-09-03",
        "last_signal_date": signal_day.isoformat(),
        "last_decision_id": "checkpoint-decision",
        "risk_on": True,
        "off_days": 0,
        "clean_days": 0,
        "active_events": [],
        "microcap_targets": [],
    }
    strategy._atomic_json(strategy._state_path(tmp_path), state)
    strategy._atomic_json(
        strategy._decision_path(tmp_path, signal_day),
        {"signal_date": signal_day.isoformat(), "decision_id": "checkpoint-decision"},
    )

    result = strategy.seal_account(service, account["id"], signal_day)
    recovered = service.ledger.get_account(account["id"])

    assert result == {"signals": 0, "orders": 0}
    assert recovered["last_processed_date"] == signal_day.isoformat()
    assert any(
        event["event_type"] == "SIGNAL_CHECKPOINT_RECOVERED"
        for event in recovered["timeline"]
    )


def test_gap_recovery_advances_each_day_without_backdating_orders(
    tmp_path, monkeypatch
) -> None:
    friday = date(2026, 9, 4)
    monday = date(2026, 9, 7)
    panel = pl.DataFrame({"date": [friday, monday]})
    features = pl.DataFrame({
        "date": [friday, monday],
        "ordinary_alarm_count": [0, 0],
        "severe_limit_down": [False, False],
    })
    event_days: list[date] = []
    monkeypatch.setattr(strategy, "_require_forecast_receipt", lambda *_args: None)
    monkeypatch.setattr(strategy, "_load_recent_panel", lambda *_args: panel)
    monkeypatch.setattr(strategy, "build_daily_features", lambda _panel: features)
    monkeypatch.setattr(
        strategy,
        "advance_risk_state",
        lambda state, feature: (state, {"decision_date": str(feature["date"])}),
    )

    def events(_data_dir, signal_date, _current):
        event_days.append(signal_date)
        if signal_date != friday:
            return []
        return [{
            "symbol": "600001.SH", "name": "邯郸钢铁", "ann_date": "2026-09-04",
            "p_change_min": 60.0, "p_change_max": 80.0,
            "signal_amount": 100_000_000.0,
        }]

    monkeypatch.setattr(strategy, "_idiosyncratic_events_for_date", events)
    monkeypatch.setattr(
        strategy,
        "_microcap_targets",
        lambda _current: [{
            "symbol": "600002.SH", "name": "齐鲁石化",
            "signal_amount": 100_000_000.0, "market_cap": 1_000_000_000.0,
            "cap_rank": 1,
        }],
    )
    previous = {
        "schema_version": strategy.STATE_SCHEMA,
        "baseline_date": "2026-09-03",
        "last_signal_date": "2026-09-03",
        "risk_on": False,
        "off_days": 1,
        "clean_days": 0,
        "active_events": [],
        "microcap_targets": [],
        "last_decision_id": "prior",
    }

    plan, state = strategy.build_forward_plan(
        tmp_path,
        monday,
        baseline_date=date(2026, 9, 3),
        previous_state=previous,
    )

    assert event_days == [friday, monday]
    assert plan["recovered_gap_dates"] == [friday.isoformat()]
    assert plan["rebalance_dates"] == [friday.isoformat()]
    assert plan["weekly_rebalance"] is True
    assert {target["symbol"] for target in plan["targets"]} == {
        "600001.SH", "600002.SH"
    }
    assert state["last_signal_date"] == monday.isoformat()


def test_v2_account_is_separate_and_freezes_allocator_contract(
    tmp_path, monkeypatch
) -> None:
    service = _PaperService(tmp_path)
    monkeypatch.setattr(strategy, "_require_result", lambda *_args, **_kwargs: tmp_path)

    v1 = strategy.ensure_account(service, date(2026, 9, 9))
    v2 = strategy.ensure_v2_account(service, date(2026, 9, 9))

    assert v1["id"] == strategy.ACCOUNT_ID
    assert v2["id"] == strategy.V2_ACCOUNT_ID
    assert v2["config"]["strategy_id"] == strategy.V2_STRATEGY_ID
    assert v2["config"]["position_sizing"] == "participation_evidence_budget"
    assert v2["config"]["research_result_sha256"] == strategy.V2_RESULT_SHA256
    assert len(service.ledger.list_accounts()) == 2


def test_v2_plan_scales_microcap_targets_to_participation_budget(
    tmp_path, monkeypatch
) -> None:
    thursday = date(2026, 9, 3)
    friday = date(2026, 9, 4)
    panel = pl.DataFrame({"date": [thursday, friday]})
    features = pl.DataFrame(
        {
            "date": [thursday, friday],
            "ordinary_alarm_count": [0, 0],
            "severe_limit_down": [False, False],
            "participation_score": [2, 1],
            "microcap_absolute_20d": [0.01, 0.01],
            "microcap_relative_20d": [0.01, -0.01],
            "microcap_breadth_20d": [0.55, 0.40],
            "microcap_liquidity_20d_60d": [0.95, 0.95],
        }
    )
    candidates = [
        {
            "symbol": f"600{index:03d}.SH",
            "name": f"测试{index}",
            "signal_amount": 100_000_000.0,
            "market_cap": 1_000_000_000.0 + index,
            "cap_rank": index,
        }
        for index in range(10)
    ]
    monkeypatch.setattr(strategy, "_require_forecast_receipt", lambda *_args: None)
    monkeypatch.setattr(strategy, "_load_recent_panel", lambda *_args: panel)
    monkeypatch.setattr(strategy, "build_daily_features", lambda _panel: features)
    monkeypatch.setattr(
        strategy.participation,
        "attach_participation_features",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        strategy,
        "_idiosyncratic_events_for_date",
        lambda *_args: [],
    )
    monkeypatch.setattr(strategy, "_microcap_targets", lambda _current: candidates)
    previous = {
        "schema_version": strategy.V2_STATE_SCHEMA,
        "baseline_date": thursday.isoformat(),
        "last_signal_date": thursday.isoformat(),
        "risk_on": True,
        "off_days": 0,
        "clean_days": 0,
        "microcap_slots": 10,
        "pending_microcap_slots": 10,
        "upgrade_days": 0,
        "active_events": [],
        "microcap_targets": [],
        "last_decision_id": "prior",
    }

    plan, next_state = strategy.build_forward_plan(
        tmp_path,
        friday,
        baseline_date=thursday,
        previous_state=previous,
        strategy_id=strategy.V2_STRATEGY_ID,
    )

    assert plan["allocation"]["microcap_slots"] == 5
    assert plan["microcap_count"] == 5
    assert sum(row["target_weight"] for row in plan["targets"]) == pytest.approx(0.25)
    assert next_state["microcap_slots"] == 5
