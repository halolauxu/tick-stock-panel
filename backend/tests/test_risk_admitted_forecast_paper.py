from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.services import risk_admitted_forecast_paper as strategy
from app.services.paper_ledger import PaperLedger


class _PaperService:
    def __init__(self, data_dir) -> None:
        self.repo = SimpleNamespace(store=SimpleNamespace(data_dir=data_dir))
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
    monkeypatch.setattr(strategy, "_require_frozen_result", lambda _data_dir: tmp_path)

    created = strategy.ensure_account(service, date(2026, 9, 3))
    repeated = strategy.ensure_account(service, date(2026, 9, 4))

    assert created["id"] == repeated["id"] == strategy.ACCOUNT_ID
    assert created["baseline_date"] == "2026-09-03"
    assert repeated["config"]["research_result_sha256"] == strategy.RESULT_SHA256
    assert repeated["config"]["position_sizing"] == "frozen_target_weight"
    assert len(service.ledger.list_accounts()) == 1


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
