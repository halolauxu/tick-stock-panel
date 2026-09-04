from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

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


def test_managed_snapshot_explains_waiting_pipeline_and_provenance(tmp_path, monkeypatch) -> None:
    service = _PaperService(tmp_path, latest_enriched=date(2026, 9, 3))
    monkeypatch.setattr(strategy, "_require_frozen_result", lambda _data_dir: tmp_path)
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
    monkeypatch.setattr(strategy, "_require_frozen_result", lambda _data_dir: tmp_path)
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
    assert waiting_seal["live"]["lifecycle"]["code"] == "WAITING_SEAL"
