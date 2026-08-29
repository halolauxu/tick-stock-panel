from __future__ import annotations

import inspect
import json
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import ValidationError

from app.api import paper_trading as paper_api
from app.jobs import daily_pipeline
from app.market_time import CN_TZ
from app.services.paper_ledger import PaperLedger, PaperLedgerError
from app.services.paper_trading import PaperTradingService, PaperTradingStore

SIGNAL_DAY = date(2026, 8, 26)
TRADE_DAY = date(2026, 8, 27)
OPEN_TIME = datetime(2026, 8, 27, 9, 30, 5, tzinfo=CN_TZ)
WEEKEND_DAY = date(2026, 8, 29)


@pytest.fixture(autouse=True)
def _freeze_ledger_clock(monkeypatch):
    """Keep historical execution tests independent of the machine's current date."""
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: OPEN_TIME)


def _config(**overrides) -> dict:
    config = {
        "strategy_id": "n_day_low_reversal",
        "strategy_name": "新低反转",
        "asset_type": "stock",
        "symbols": None,
        "params": {"lookback": 20},
        "overrides": {"max_hold_days": 5},
        "entry_fill": "open_t+1",
        "exit_fill": "open_t+1",
        "exit_mode": "eod",
        "commission_pct": 0.0002,
        "stamp_tax_pct": 0.001,
        "slippage_bps": 5,
        "max_positions": 10,
        "max_exposure_pct": 1,
        "initial_capital": 200_000,
        "position_sizing": "equal",
        "minute_fill": False,
        "regime_filter": None,
        "enforce_t_plus_one": True,
    }
    config.update(overrides)
    return config


class FakeRepo:
    def __init__(
        self,
        data_dir,
        minute_rows: list[dict] | None = None,
        daily_rows: list[dict] | None = None,
    ) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)
        self.minute_rows = minute_rows or []
        self.daily_rows = daily_rows or []

    @staticmethod
    def latest_enriched_date(_asset_type: str) -> date:
        return SIGNAL_DAY

    def get_minute_batch(self, _symbols: list[str], _trading_date: date) -> pl.DataFrame:
        rows = [
            row for row in self.minute_rows
            if row.get("symbol") in _symbols
            and row.get("datetime") is not None
            and (
                row["datetime"].date()
                if row["datetime"].tzinfo is not None
                else row["datetime"].replace(tzinfo=CN_TZ).date()
            ) == _trading_date
        ]
        return pl.DataFrame(rows)

    def get_daily_asset(
        self,
        _asset_type: str,
        symbol: str,
        start: date,
        end: date,
        _columns: list[str],
    ) -> pl.DataFrame:
        rows = [
            row for row in self.daily_rows
            if row.get("symbol") == symbol and start <= row.get("date") <= end
        ]
        if rows:
            return pl.DataFrame(rows)
        return pl.DataFrame({"raw_close": [10.0], "close": [10.0]})


def _service(
    tmp_path,
    *,
    minute_rows: list[dict] | None = None,
    daily_rows: list[dict] | None = None,
) -> PaperTradingService:
    return PaperTradingService(
        SimpleNamespace(repo=FakeRepo(tmp_path, minute_rows, daily_rows))
    )


def _account_with_buy_order(service: PaperTradingService, *, config: dict | None = None):
    account = service.ledger.create_account(
        name="事件驱动模拟盘",
        baseline_date=SIGNAL_DAY,
        config=config or _config(),
        created_at=datetime(2026, 8, 26, 15, 30, tzinfo=CN_TZ),
    )
    _, order_id, _ = service.ledger.record_signal_and_order(
        account_id=account["id"],
        strategy_id="n_day_low_reversal",
        symbol="000001.SZ",
        name="平安银行",
        side="BUY",
        signal_date=SIGNAL_DAY,
        score=88,
        reason="strategy_entry",
        signal_ref="entry-low-reversal",
        requested_qty=1_000,
        target_amount=10_000,
        target_weight=0.05,
        planned_session="NEXT_OPEN",
        frozen_at=datetime(2026, 8, 26, 15, 30, tzinfo=CN_TZ),
    )
    return account, order_id


def _quote(
    *,
    symbol: str = "000001.SZ",
    price: float = 10.0,
    previous: float | None = 9.9,
    volume: float = 1_000,
    at: datetime = OPEN_TIME,
) -> dict:
    return {
        "symbol": symbol,
        "open": price,
        "high": price + 0.02,
        "low": price - 0.02,
        "last_price": price,
        "prev_close": previous,
        "volume": volume,
        "quote_at": at.isoformat(),
        "_quote_dt": at,
        "source": "test_quote",
    }


def test_account_is_persisted_in_transactional_ledger(tmp_path):
    store = PaperTradingStore(tmp_path)
    created = store.create(
        name="新低反转模拟盘",
        start_date=SIGNAL_DAY,
        config=_config(),
        created_at=datetime(2026, 8, 26, 15, 30, tzinfo=CN_TZ),
    )

    loaded = PaperTradingStore(tmp_path).get(created["id"])

    assert loaded["schema_version"] == 6
    assert loaded["execution_policy"] == "event_driven"
    assert loaded["summary"]["cash"] == 200_000
    assert loaded["cash_entries"][0]["event_type"] == "INITIAL_CAPITAL"
    assert loaded["timeline"][0]["event_type"] == "ACCOUNT_CREATED"


def test_capital_contribution_updates_budget_and_cash_ledger_idempotently(tmp_path):
    ledger = PaperLedger(tmp_path)
    account = ledger.create_account(
        name="预算调整账户",
        baseline_date=SIGNAL_DAY,
        config=_config(),
    )

    updated = ledger.increase_capital(
        account["id"],
        100_000,
        reference_id="budget-300000",
        detail="用户将模拟预算调整为 30 万",
    )
    repeated = ledger.increase_capital(
        account["id"],
        100_000,
        reference_id="budget-300000",
        detail="用户将模拟预算调整为 30 万",
    )

    assert updated["config"]["initial_capital"] == 300_000
    assert updated["summary"]["cash"] == 300_000
    assert repeated["summary"]["cash"] == 300_000
    contributions = [
        row for row in repeated["cash_entries"]
        if row["event_type"] == "CAPITAL_CONTRIBUTION"
    ]
    assert len(contributions) == 1
    assert contributions[0]["amount"] == 100_000
    assert ledger.reconcile(account["id"], open_incident=False)["ok"] is True


def _stub_signal_seal(monkeypatch, service: PaperTradingService, rows: list[dict]) -> None:
    class StubScreener:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def build_strategy_context(*_args, **_kwargs):
            return SimpleNamespace(current=pl.DataFrame(rows))

    service.app_state.strategy_engine = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            rows=rows,
            exit_signal_hits=[],
            entry_signal_hits=[
                {"symbol": row["symbol"], "signals": ["signal_n_day_low"]}
                for row in rows
            ],
        )
    )
    monkeypatch.setattr("app.services.paper_trading.ScreenerService", StubScreener)


def test_equal_position_sizing_uses_max_position_slot_budget(tmp_path, monkeypatch):
    service = _service(tmp_path)
    _stub_signal_seal(monkeypatch, service, [
        {"symbol": "000001.SZ", "name": "平安银行", "score": 90.0, "raw_close": 10.0},
        {"symbol": "600000.SH", "name": "浦发银行", "score": 80.0, "raw_close": 10.0},
    ])
    account = service.ledger.create_account(
        name="按槽位等权账户",
        baseline_date=SIGNAL_DAY,
        config=_config(),
    )

    result = service.seal_account_signals(account["id"], SIGNAL_DAY)
    current = service.account(account["id"])

    assert result == {"signals": 2, "orders": 2}
    assert len(current["orders"]) == 2
    assert {row["requested_qty"] for row in current["orders"]} == {1_900}
    assert {row["target_amount"] for row in current["orders"]} == {20_000}
    assert {row["target_weight"] for row in current["orders"]} == {0.1}


def test_zero_lot_candidate_is_frozen_and_explained_in_timeline(tmp_path, monkeypatch):
    service = _service(tmp_path)
    _stub_signal_seal(monkeypatch, service, [
        {"symbol": "002028.SZ", "name": "思源电气", "score": 50.0, "raw_close": 143.18},
    ])
    account = service.ledger.create_account(
        name="小额预算账户",
        baseline_date=SIGNAL_DAY,
        config=_config(initial_capital=10_000),
    )

    result = service.seal_account_signals(account["id"], SIGNAL_DAY)
    current = service.account(account["id"])

    assert result == {"signals": 1, "orders": 0}
    assert current["orders"] == []
    skipped = [row for row in current["timeline"] if row["event_type"] == "SIGNAL_SKIPPED"]
    assert len(skipped) == 1
    assert skipped[0]["payload"]["skip_code"] == "INSUFFICIENT_BUYING_POWER"
    assert "不足买入一手" in skipped[0]["detail"]
    with service.ledger._connect() as conn:  # verify immutable domain row
        signal_count = conn.execute(
            "SELECT count(*) FROM signal_intents WHERE account_id=?", (account["id"],)
        ).fetchone()[0]
    assert signal_count == 1


def test_paper_service_has_no_backtest_replay_dependency():
    source = inspect.getsource(PaperTradingService)

    assert "run_worker_task" not in source
    assert '"kind": "backtest"' not in source


def test_scheduler_registers_and_dispatches_all_exchange_clock_boundaries():
    jobs: dict[str, dict] = {}

    class Scheduler:
        @staticmethod
        def add_job(func, **kwargs):
            jobs[kwargs["id"]] = {"func": func, **kwargs}

    calls: list[str] = []
    service = SimpleNamespace(
        probe_quote_chain=lambda quotes=None: calls.append("canary"),
        preflight_all=lambda: calls.append("preflight"),
        execute_open_orders=lambda: calls.append("open"),
        finalize_open_window=lambda: calls.append("deadline"),
        settle_all=lambda: calls.append("settlement"),
        recover_missed_open=lambda: calls.append("recovery"),
    )
    daily_pipeline.set_app_state(SimpleNamespace(paper_trading_service=service))

    daily_pipeline._register_paper_clock_jobs(Scheduler())
    for job_id in (
        "paper_quote_canary",
        "paper_preflight",
        "paper_open_execution",
        "paper_open_deadline",
        "paper_settlement",
        "paper_evidence_recovery",
    ):
        jobs[job_id]["func"]()

    assert calls == ["canary", "preflight", "open", "deadline", "settlement", "recovery"]
    assert "hour='9', minute='20'" in str(jobs["paper_quote_canary"]["trigger"])
    assert str(jobs["paper_preflight"]["trigger"]).startswith("cron[day_of_week='mon-fri'")
    assert "hour='9', minute='30', second='5,25,45'" in str(
        jobs["paper_open_execution"]["trigger"]
    )
    assert "interval[0:00:10]" in str(jobs["paper_quote_tick"]["trigger"])
    assert "interval[0:01:00]" in str(jobs["paper_evidence_recovery"]["trigger"])


def test_clock_passes_targeted_quotes_directly_to_preflight():
    received: list[dict] = []
    records = [{"symbol": "000001.SH", "last_price": 3_900, "timestamp": OPEN_TIME}]
    service = SimpleNamespace(
        subscription_symbols=lambda: {"000001.SZ"},
        quotes_from_records=lambda values, source: {row["symbol"]: row for row in values},
        preflight_all=lambda *, quotes: received.append(quotes) or {"assigned": 0},
    )
    quote_service = SimpleNamespace(
        refresh_paper_symbols=lambda *, notify: {"records": records}
    )
    daily_pipeline.set_app_state(SimpleNamespace(
        paper_trading_service=service,
        quote_service=quote_service,
    ))

    daily_pipeline._paper_clock_call("preflight_all")

    assert received == [{"000001.SH": records[0]}]


def test_quote_canary_opens_and_resolves_a_visible_incident(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    at = datetime(2026, 8, 27, 9, 20, tzinfo=CN_TZ)
    index_quote = _quote(symbol="000001.SH", price=3_900, previous=3_880, at=at)

    failed = service.probe_quote_chain(
        now=at,
        quotes={"000001.SH": index_quote},
    )
    after_failure = service.account(account["id"])

    assert failed["ready"] is False
    assert failed["missing"] == ["000001.SZ"]
    assert after_failure["summary"]["critical_incident_count"] == 1
    assert any(
        incident["code"] == "QUOTE_CHAIN_NOT_READY"
        and incident["status"] == "open"
        for incident in after_failure["incidents"]
    )

    ready = service.probe_quote_chain(
        now=at,
        quotes={
            "000001.SH": index_quote,
            "000001.SZ": _quote(at=at),
        },
    )
    after_recovery = service.account(account["id"])

    assert ready == {"ready": True, "required": 2, "available": 2, "missing": []}
    assert after_recovery["summary"]["critical_incident_count"] == 0


def test_preflight_requires_each_orders_own_current_quote(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    at = datetime(2026, 8, 27, 9, 25, tzinfo=CN_TZ)

    result = service.preflight_all(
        now=at,
        quotes={"000001.SH": _quote(symbol="000001.SH", at=at)},
    )
    current = service.account(account["id"])

    assert result == {"checked": 1, "deferred": 1, "assigned": 0}
    assert current["orders"][0]["status"] == "PLANNED"
    assert any(
        incident["code"] == "ORDER_QUOTE_NOT_READY"
        and incident["status"] == "open"
        for incident in current["incidents"]
    )


def test_weekend_quotes_cannot_schedule_or_terminalize_next_open_order(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    saturday_open = datetime(2026, 8, 29, 9, 31, tzinfo=CN_TZ)

    result = service.execute_open_orders(
        now=saturday_open,
        quotes={"000001.SZ": _quote(at=saturday_open)},
        finalize_missing=True,
    )
    current = service.account(account["id"])

    assert result == {
        "filled": 0,
        "partial": 0,
        "rejected": 0,
        "unknown": 0,
        "waiting": 1,
    }
    assert current["orders"][0]["status"] == "PLANNED"
    assert current["orders"][0]["scheduled_date"] is None
    assert current["fills"] == []
    assert current["summary"]["critical_incident_count"] == 0


def test_weekend_preflight_does_not_unlock_friday_t_plus_one_position(tmp_path):
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})
    service.ledger.execute_fill(
        order_id,
        price=10,
        quantity=1_000,
        quote_at=OPEN_TIME,
        source="open_quote",
    )
    saturday_open = datetime(2026, 8, 29, 9, 25, tzinfo=CN_TZ)

    service.preflight_all(
        now=saturday_open,
        quotes={"000001.SZ": _quote(at=saturday_open)},
    )
    position = service.account(account["id"])["positions"][0]

    assert position["available_qty"] == 0
    assert position["locked_qty"] == 1_000


def test_cache_fetch_time_never_masquerades_as_quote_time(tmp_path):
    service = _service(tmp_path)
    saturday_fetch = datetime(2026, 8, 29, 9, 31, tzinfo=CN_TZ)
    service.app_state.quote_service = SimpleNamespace(
        get_enriched_today=lambda: (
            pl.DataFrame([{
                "symbol": "000001.SZ",
                "close": 10.0,
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "volume": 1_000,
            }]),
            WEEKEND_DAY,
        ),
        status=lambda: {"last_fetch_ms": saturday_fetch.timestamp() * 1_000},
    )

    assert service._quotes_from_cache() == {}


def test_weekday_without_original_same_day_quote_fails_closed(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    monday_open = datetime(2026, 8, 31, 9, 31, tzinfo=CN_TZ)
    stale_friday = datetime(2026, 8, 28, 15, 0, tzinfo=CN_TZ)

    result = service.execute_open_orders(
        now=monday_open,
        quotes={"000001.SZ": _quote(at=stale_friday)},
        finalize_missing=True,
    )
    current = service.account(account["id"])

    assert result["filled"] == 0
    assert result["unknown"] == 0
    assert current["orders"][0]["status"] == "PLANNED"
    assert current["orders"][0]["scheduled_date"] is None
    assert current["fills"] == []
    assert current["summary"]["critical_incident_count"] == 0
    assert any(
        incident["code"] == "TRADING_DAY_UNCONFIRMED"
        and incident["status"] == "open"
        for incident in current["incidents"]
    )


def test_quote_tick_refreshes_tracked_paper_symbols_during_market(monkeypatch):
    calls: list[str] = []
    state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"}
        ),
        quote_service=SimpleNamespace(refresh=lambda: calls.append("refresh")),
    )
    daily_pipeline.set_app_state(state)
    monkeypatch.setattr(
        daily_pipeline,
        "cn_now",
        lambda: datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ),
    )

    daily_pipeline._paper_quote_tick()

    assert calls == ["refresh"]


def test_quote_tick_skips_weekend_market_clock(monkeypatch):
    calls: list[str] = []
    state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"}
        ),
        quote_service=SimpleNamespace(refresh=lambda: calls.append("refresh")),
    )
    daily_pipeline.set_app_state(state)
    monkeypatch.setattr(
        daily_pipeline,
        "cn_now",
        lambda: datetime(2026, 8, 29, 11, 25, tzinfo=CN_TZ),
    )

    daily_pipeline._paper_quote_tick()

    assert calls == []


def test_quote_tick_does_not_duplicate_a_running_global_poll(monkeypatch):
    calls: list[str] = []
    state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"}
        ),
        quote_service=SimpleNamespace(
            refresh=lambda: calls.append("refresh"),
            status=lambda: {"enabled": True, "running": True, "paused": False},
            global_poll_covers_paper_symbols=lambda: True,
        ),
    )
    daily_pipeline.set_app_state(state)
    monkeypatch.setattr(
        daily_pipeline,
        "cn_now",
        lambda: datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ),
    )

    daily_pipeline._paper_quote_tick()

    assert calls == []


def test_quote_tick_finalizes_close_marks_without_hammering_provider(monkeypatch):
    calls: list[str] = []
    claims: list[datetime] = []

    def claim(*, now):
        claims.append(now)
        return len(claims) == 1

    state = SimpleNamespace(
        paper_trading_service=SimpleNamespace(
            subscription_symbols=lambda: {"000001.SZ"},
            claim_close_quote_refresh=claim,
        ),
        quote_service=SimpleNamespace(refresh=lambda: calls.append("refresh")),
    )
    daily_pipeline.set_app_state(state)
    monkeypatch.setattr(
        daily_pipeline,
        "cn_now",
        lambda: datetime(2026, 8, 27, 16, 0, tzinfo=CN_TZ),
    )

    daily_pipeline._paper_quote_tick()
    daily_pipeline._paper_quote_tick()

    assert len(claims) == 2
    assert calls == ["refresh"]


def test_system_does_not_report_stale_quotes_when_nothing_requires_quotes(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    service.app_state.quote_service = SimpleNamespace(
        status=lambda: {"quote_age_ms": None, "mode": "watchlist", "enabled": False}
    )
    monkeypatch.setattr(
        "app.services.paper_trading.cn_now",
        lambda: datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ),
    )

    status = service.system_status()

    assert status["tracked_symbol_count"] == 0
    assert status["quote_stale"] is False
    assert status["executor_health"] == "HEALTHY"


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (datetime(2026, 8, 27, 12, 0, tzinfo=CN_TZ), "LUNCH_BREAK"),
        (datetime(2026, 8, 27, 17, 0, tzinfo=CN_TZ), "CLOSED"),
        (datetime(2026, 8, 29, 10, 0, tzinfo=CN_TZ), "CLOSED"),
    ],
)
def test_system_market_phase_describes_closed_sessions(tmp_path, monkeypatch, at, expected):
    service = _service(tmp_path)
    monkeypatch.setattr("app.services.paper_trading.cn_now", lambda: at)

    assert service.system_status()["market_phase"] == expected


def test_duplicate_execution_cannot_duplicate_fill_cash_or_position(tmp_path):
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})

    first_fill = service.ledger.execute_fill(
        order_id, price=10, quantity=1_000, quote_at=OPEN_TIME, source="open_quote"
    )
    second_fill = PaperLedger(tmp_path).execute_fill(
        order_id, price=10, quantity=1_000, quote_at=OPEN_TIME, source="open_quote"
    )
    current = service.account(account["id"])

    assert second_fill == first_fill
    assert len(current["fills"]) == 1
    assert current["positions"][0]["quantity"] == 1_000
    assert current["positions"][0]["locked_qty"] == 1_000
    assert len([row for row in current["cash_entries"] if row["event_type"] == "BUY_FILL"]) == 1
    assert current["reconciliation"]["ok"] is True


def test_today_pnl_uses_previous_close_and_buy_cost_without_backfill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.paper_ledger.cn_now",
        lambda: datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ),
    )
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})
    service.ledger.execute_fill(
        order_id,
        price=10,
        quantity=1_000,
        quote_at=OPEN_TIME,
        source="open_quote",
        previous_close=9.9,
    )
    service.ledger.update_marks(
        {
            "000001.SZ": {
                "last_price": 10.5,
                "prev_close": 9.9,
                "quote_at": datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ).isoformat(),
            }
        },
        source="test_quote",
    )

    current = service.account(account["id"])

    assert current["summary"]["today_pnl_available"] is True
    assert current["summary"]["today_pnl"] == pytest.approx(493.0)
    assert current["positions"][0]["pnl_date"] == "2026-08-27"
    assert current["positions"][0]["today_bought_qty"] == 1_000


def test_revised_close_mark_restates_settlement_once(tmp_path, monkeypatch):
    observed = datetime(2026, 8, 27, 16, 10, tzinfo=CN_TZ)
    monkeypatch.setattr("app.services.paper_trading.cn_now", lambda: observed)
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: observed)
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})
    service.ledger.execute_fill(
        order_id,
        price=10,
        quantity=1_000,
        quote_at=OPEN_TIME,
        source="open_quote",
        previous_close=9.9,
    )
    service.ledger.settle_account(account["id"], TRADE_DAY, source="15:05_close")
    final_quote = _quote(
        price=10.2,
        at=datetime(2026, 8, 27, 15, 0, 4, tzinfo=CN_TZ),
    )

    first = service.on_quote_records([final_quote], source="close_final")
    second = service.on_quote_records([final_quote], source="close_final")
    current = service.account(account["id"])

    assert first["marked"] == 1
    assert second["marked"] == 0
    assert current["positions"][0]["last_price"] == 10.2
    assert current["nav"][0]["market_value"] == 10_200
    assert current["nav"][0]["equity"] == pytest.approx(current["summary"]["equity"])
    assert len([
        event for event in current["timeline"]
        if event["event_type"] == "SETTLEMENT_RESTATED"
    ]) == 1


def test_today_pnl_keeps_realized_sell_after_position_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.paper_ledger.cn_now",
        lambda: datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ),
    )
    service = _service(tmp_path)
    account, buy_order = _account_with_buy_order(service)
    service.ledger.assign_due_date(buy_order, SIGNAL_DAY, {})
    service.ledger.execute_fill(
        buy_order,
        price=10,
        quantity=1_000,
        quote_at=datetime(2026, 8, 26, 9, 30, tzinfo=CN_TZ),
        source="open_quote",
        previous_close=9.8,
    )
    service.ledger.unlock_positions(TRADE_DAY)
    _, sell_order, _ = service.ledger.record_signal_and_order(
        account_id=account["id"],
        strategy_id="n_day_low_reversal",
        symbol="000001.SZ",
        name="平安银行",
        side="SELL",
        signal_date=SIGNAL_DAY,
        score=None,
        reason="strategy_exit",
        signal_ref=None,
        requested_qty=1_000,
        target_amount=10_500,
        target_weight=0,
        planned_session="NEXT_OPEN",
    )
    service.ledger.assign_due_date(sell_order, TRADE_DAY, {})
    service.ledger.execute_fill(
        sell_order,
        price=10.5,
        quantity=1_000,
        quote_at=OPEN_TIME,
        source="open_quote",
        previous_close=10.0,
    )

    current = service.account(account["id"])
    sell_fill = next(fill for fill in current["fills"] if fill["side"] == "SELL")

    assert current["positions"] == []
    assert current["summary"]["today_pnl"] == pytest.approx(482.15)
    assert sell_fill["day_pnl"] == pytest.approx(482.15)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        (_quote(volume=0), "REJECTED_SUSPENDED"),
        (
            {**_quote(price=11.0, previous=10.0), "high": 11.0, "low": 11.0},
            "REJECTED_LIMIT_UP",
        ),
    ],
)
def test_open_executor_has_explicit_blocked_terminal_states(tmp_path, quote, expected):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)

    result = service.execute_open_orders(
        now=datetime(2026, 8, 27, 9, 31, tzinfo=CN_TZ),
        quotes={"000001.SZ": quote},
        finalize_missing=True,
    )

    assert result["rejected"] == 1
    assert service.account(account["id"])["orders"][0]["status"] == expected


def test_zero_volume_at_0930_waits_until_deadline_before_suspension(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    quote = _quote(volume=0)

    result = service.execute_open_orders(now=OPEN_TIME, quotes={"000001.SZ": quote})

    assert result["waiting"] == 1
    assert service.account(account["id"])["orders"][0]["status"] == "PREFLIGHT_OK"


def test_open_executor_rejects_insufficient_cash(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service, config=_config(initial_capital=1_000))

    service.execute_open_orders(now=OPEN_TIME, quotes={"000001.SZ": _quote(price=10)})

    assert service.account(account["id"])["orders"][0]["status"] == (
        "REJECTED_INSUFFICIENT_CASH"
    )


def test_0931_missing_symbol_quote_becomes_unknown_not_a_fill(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    other = _quote(symbol="000002.SZ")

    result = service.execute_open_orders(
        now=datetime(2026, 8, 27, 9, 31, tzinfo=CN_TZ),
        quotes={"000002.SZ": other},
        finalize_missing=True,
    )
    current = service.account(account["id"])

    assert result["unknown"] == 1
    assert current["orders"][0]["status"] == "UNKNOWN_MARKET_DATA"
    assert current["fills"] == []


def test_restart_requeues_weekend_misclassification_with_compensating_audit(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    saturday = datetime(2026, 8, 29, 9, 31, tzinfo=CN_TZ)
    service.ledger.assign_due_date(
        order_id,
        WEEKEND_DAY,
        {"market_observed": True, "quote_at": saturday.isoformat()},
    )
    service.ledger.terminal_order(
        order_id,
        status="UNKNOWN_MARKET_DATA",
        reason="09:31 当时缺少可靠开盘行情",
        quality="NO_RELIABLE_OPEN_DATA",
        scheduled_date=WEEKEND_DAY,
        severity="critical",
    )
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: saturday)

    restarted = _service(tmp_path)
    current = restarted.account(account["id"])
    event_types = [event["event_type"] for event in current["timeline"]]

    assert current["orders"][0]["status"] == "PLANNED"
    assert current["orders"][0]["scheduled_date"] is None
    assert current["orders"][0]["execution_quality"] is None
    assert current["fills"] == []
    assert current["summary"]["critical_incident_count"] == 0
    assert "INVALID_SESSION_REQUEUED" in event_types
    assert "UNKNOWN_MARKET_DATA" in event_types


def test_late_recovery_preserves_missed_event_and_marks_fill_recovered_late(tmp_path):
    minute_rows = [{
        "symbol": "000001.SZ",
        "datetime": datetime(2026, 8, 27, 9, 30, tzinfo=CN_TZ),
        "open": 10.0,
        "high": 10.05,
        "low": 9.98,
        "close": 10.02,
        "volume": 12_000,
    }]
    service = _service(tmp_path, minute_rows=minute_rows)
    account, _ = _account_with_buy_order(service)

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])
    event_types = {event["event_type"] for event in current["timeline"]}

    assert result == {
        "missed": 1,
        "recovered": 1,
        "resolved": 0,
        "unknown": 0,
        "waiting_evidence": 0,
    }
    assert current["orders"][0]["status"] == "FILLED"
    assert current["orders"][0]["execution_quality"] == "RECOVERED_LATE"
    assert current["fills"][0]["quality"] == "RECOVERED_LATE"
    assert "MISSED_EXECUTION" in event_types
    assert "FILLED" in event_types
    assert current["summary"]["critical_incident_count"] == 0


def test_late_recovery_without_reliable_minute_data_stays_unknown(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    market_quote = _quote(
        symbol="000002.SZ", at=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    service._quotes_from_cache = lambda: {"000002.SZ": market_quote}  # type: ignore[method-assign]

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert result == {
        "missed": 1,
        "recovered": 0,
        "resolved": 0,
        "unknown": 1,
        "waiting_evidence": 1,
    }
    assert current["orders"][0]["status"] == "UNKNOWN_MARKET_DATA"
    assert current["orders"][0]["execution_quality"] == "NO_RELIABLE_OPEN_DATA"
    assert current["fills"] == []


def test_completed_day_quote_recovers_open_fill_even_with_other_cached_quotes(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    service.ledger.settle_account(account["id"], TRADE_DAY, source="15:05_close")
    other = _quote(
        symbol="000002.SZ",
        at=datetime(2026, 8, 27, 15, 0, 5, tzinfo=CN_TZ),
    )
    service._quotes_from_cache = lambda: {"000002.SZ": other}  # type: ignore[method-assign]
    refresh_calls: list[str] = []
    service.app_state.quote_service = SimpleNamespace(
        refresh_paper_symbols=lambda: refresh_calls.append("paper") or {
            "records": [{
                "symbol": "000001.SZ",
                "timestamp": datetime(2026, 8, 27, 15, 0, 4, tzinfo=CN_TZ),
                "open": 10.0,
                "high": 10.4,
                "low": 9.8,
                "last_price": 10.2,
                "prev_close": 9.9,
                "volume": 88_000,
            }],
        },
        refresh=lambda: {},
        status=lambda: {"quote_age_ms": 0, "mode": "paper", "enabled": True},
    )

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 16, 5, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert refresh_calls == ["paper"]
    assert result["recovered"] == 1
    assert current["fills"][0]["price"] == 10.0
    assert current["fills"][0]["source"] == (
        "realtime_close_snapshot_open_recovery"
    )
    assert current["orders"][0]["execution_quality"] == "RECOVERED_LATE"
    assert len(current["nav"]) == 1
    assert current["nav"][0]["equity"] == pytest.approx(
        current["summary"]["equity"]
    )
    assert any(
        event["event_type"] == "SETTLEMENT_RESTATED"
        for event in current["timeline"]
    )


def test_intraday_snapshot_does_not_masquerade_as_completed_open_evidence(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    current_quote = _quote(at=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ))
    service._quotes_from_cache = lambda: {"000001.SZ": current_quote}  # type: ignore[method-assign]

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert result["waiting_evidence"] == 1
    assert result["recovered"] == 0
    assert current["orders"][0]["status"] == "UNKNOWN_MARKET_DATA"
    assert current["fills"] == []


def test_cached_after_close_quote_requires_a_fresh_targeted_snapshot(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    cached_quote = _quote(at=datetime(2026, 8, 27, 16, 5, tzinfo=CN_TZ))
    service._quotes_from_cache = lambda: {"000001.SZ": cached_quote}  # type: ignore[method-assign]

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 16, 5, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert result["waiting_evidence"] == 1
    assert result["recovered"] == 0
    assert current["orders"][0]["status"] == "UNKNOWN_MARKET_DATA"
    assert current["fills"] == []


def test_unknown_order_is_reconciled_when_opening_evidence_arrives_later(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    market_quote = _quote(
        symbol="000002.SZ", at=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    service._quotes_from_cache = lambda: {"000002.SZ": market_quote}  # type: ignore[method-assign]

    first = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    assert first["waiting_evidence"] == 1
    assert service.account(account["id"])["orders"][0]["status"] == (
        "UNKNOWN_MARKET_DATA"
    )
    assert service.subscription_symbols() == {"000001.SZ"}

    service.repo.minute_rows = [{
        "symbol": "000001.SZ",
        "datetime": datetime(2026, 8, 27, 9, 30, tzinfo=CN_TZ),
        "open": 10.0,
        "high": 10.05,
        "low": 9.98,
        "close": 10.02,
        "volume": 12_000,
    }]
    second = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 26, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert second["recovered"] == 1
    assert current["orders"][0]["status"] == "FILLED"
    assert current["orders"][0]["execution_quality"] == "RECOVERED_LATE"
    assert len(current["fills"]) == 1
    assert current["summary"]["critical_incident_count"] == 0
    assert service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 27, tzinfo=CN_TZ)
    )["recovered"] == 0
    assert len(service.account(account["id"])["fills"]) == 1


def test_unknown_recovery_survives_restart_and_unlocks_historical_buy(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    market_quote = _quote(
        symbol="000002.SZ", at=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    service._quotes_from_cache = lambda: {"000002.SZ": market_quote}  # type: ignore[method-assign]
    service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )

    next_day = datetime(2026, 8, 28, 8, 0, tzinfo=CN_TZ)
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: next_day)
    minute_rows = [{
        "symbol": "000001.SZ",
        "datetime": datetime(2026, 8, 27, 9, 30, tzinfo=CN_TZ),
        "open": 10.0,
        "high": 10.05,
        "low": 9.98,
        "close": 10.02,
        "volume": 12_000,
    }]
    restarted = _service(tmp_path, minute_rows=minute_rows)

    result = restarted.recover_missed_open(now=next_day)
    current = restarted.account(account["id"])

    assert result["recovered"] == 1
    assert len(current["fills"]) == 1
    assert current["positions"][0]["available_qty"] == 1_000
    assert current["positions"][0]["locked_qty"] == 0
    assert _service(tmp_path, minute_rows=minute_rows).recover_missed_open(
        now=next_day
    )["recovered"] == 0
    assert len(restarted.account(account["id"])["fills"]) == 1


def test_full_day_downtime_keeps_original_next_session_instead_of_deployment_day(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    daily_partition = tmp_path / "kline_daily" / "date=2026-08-27"
    daily_partition.mkdir(parents=True)
    (daily_partition / "part.parquet").touch()
    minute_rows = [{
        "symbol": "000001.SZ",
        "datetime": datetime(2026, 8, 27, 9, 30, tzinfo=CN_TZ),
        "open": 10.0,
        "high": 10.05,
        "low": 9.98,
        "close": 10.02,
        "volume": 12_000,
    }]
    restarted_at = datetime(2026, 8, 28, 8, 0, tzinfo=CN_TZ)
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: restarted_at)
    restarted = _service(tmp_path, minute_rows=minute_rows)

    result = restarted.recover_missed_open(now=restarted_at)
    current = restarted.account(account["id"])

    assert result["missed"] == 1
    assert result["recovered"] == 1
    assert current["orders"][0]["scheduled_date"] == "2026-08-27"
    assert current["orders"][0]["status"] == "FILLED"
    assert current["orders"][0]["execution_quality"] == "RECOVERED_LATE"
    assert current["positions"][0]["available_qty"] == 1_000
    assert current["positions"][0]["locked_qty"] == 0
    assert any(
        event["event_type"] == "MISSED_EXECUTION"
        and event["trading_date"] == "2026-08-27"
        for event in current["timeline"]
    )


def test_historical_daily_ohlcv_recovers_without_full_market_minute_partition(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    daily_partition = tmp_path / "kline_daily" / "date=2026-08-27"
    daily_partition.mkdir(parents=True)
    (daily_partition / "part.parquet").touch()
    next_day = datetime(2026, 8, 28, 8, 0, tzinfo=CN_TZ)
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: next_day)
    restarted = _service(
        tmp_path,
        daily_rows=[{
            "symbol": "000001.SZ",
            "date": TRADE_DAY,
            "raw_open": 10.0,
            "raw_high": 10.4,
            "raw_low": 9.8,
            "raw_close": 10.2,
            "raw_volume": 88_000,
            "prev_close": 9.9,
        }],
    )

    result = restarted.recover_missed_open(now=next_day)
    current = restarted.account(account["id"])

    assert result["missed"] == 1
    assert result["recovered"] == 1
    assert current["fills"][0]["price"] == 10.0
    assert current["fills"][0]["source"] == "daily_close_snapshot_open_recovery"
    assert current["positions"][0]["available_qty"] == 1_000
    assert current["positions"][0]["locked_qty"] == 0


def test_partial_live_daily_partition_is_not_historical_recovery_evidence(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)
    daily_partition = tmp_path / "kline_daily" / "date=2026-08-27"
    daily_partition.mkdir(parents=True)
    (daily_partition / "part.parquet").touch()
    next_day = datetime(2026, 8, 28, 8, 0, tzinfo=CN_TZ)
    monkeypatch.setattr("app.services.paper_ledger.cn_now", lambda: next_day)
    restarted = _service(
        tmp_path,
        daily_rows=[{
            "symbol": "000001.SZ",
            "date": TRADE_DAY,
            "open": 10.0,
            "high": 10.4,
            "low": 9.8,
            "close": 10.2,
            "volume": 88_000,
            "prev_close": 9.9,
            "quote_ts": datetime(2026, 8, 27, 14, 13, tzinfo=CN_TZ),
        }],
    )

    result = restarted.recover_missed_open(now=next_day)
    current = restarted.account(account["id"])

    assert result["recovered"] == 0
    assert result["waiting_evidence"] == 1
    assert current["orders"][0]["status"] == "UNKNOWN_MARKET_DATA"
    assert current["fills"] == []


def test_unknown_order_only_allows_compensating_late_fill(tmp_path):
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {})
    service.ledger.terminal_order(
        order_id,
        status="UNKNOWN_MARKET_DATA",
        reason="09:31 时没有可靠开盘行情",
        quality="NO_RELIABLE_OPEN_DATA",
    )

    with pytest.raises(PaperLedgerError, match="订单已终结"):
        service.ledger.execute_fill(
            order_id,
            price=10,
            quantity=1_000,
            quote_at=OPEN_TIME,
            source="minute_k",
        )

    service.ledger.execute_fill(
        order_id,
        price=10,
        quantity=1_000,
        quote_at=OPEN_TIME,
        source="minute_k_recovery",
        quality="RECOVERED_LATE",
    )
    assert service.account(account["id"])["orders"][0]["status"] == "FILLED"


def test_recovery_fetches_only_missing_symbol_opening_minutes(
    tmp_path, monkeypatch
):
    calls: list[tuple[str, date, str]] = []

    def fetch(symbol: str, trading_date: date, asset_type: str) -> pl.DataFrame:
        calls.append((symbol, trading_date, asset_type))
        return pl.DataFrame([{
            "symbol": symbol,
            "datetime": datetime(2026, 8, 27, 9, 30, tzinfo=CN_TZ),
            "open": 10.0,
            "high": 10.05,
            "low": 9.98,
            "close": 10.02,
            "volume": 12_000,
        }])

    monkeypatch.setattr("app.services.kline_sync.fetch_minute_single", fetch)
    state = SimpleNamespace(
        repo=FakeRepo(tmp_path),
        capabilities=SimpleNamespace(has=lambda _capability: True),
    )
    service = PaperTradingService(state)
    account, _ = _account_with_buy_order(service)
    service._quotes_from_cache = lambda: {}  # type: ignore[method-assign]

    result = service.recover_missed_open(
        now=datetime(2026, 8, 27, 11, 25, tzinfo=CN_TZ)
    )
    current = service.account(account["id"])

    assert calls == [("000001.SZ", TRADE_DAY, "stock")]
    assert result["recovered"] == 1
    assert current["fills"][0]["source"] == "minute_k_targeted_recovery"


def test_intraday_stop_on_same_day_buy_is_t1_locked_and_not_sold(tmp_path):
    config = _config(exit_mode="intraday", overrides={"stop_loss": 0.05})
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service, config=config)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})
    service.ledger.execute_fill(
        order_id, price=10, quantity=1_000, quote_at=OPEN_TIME, source="open_quote"
    )
    stop_time = datetime(2026, 8, 27, 10, 5, tzinfo=CN_TZ)

    service.on_quote_records(
        [{
            "symbol": "000001.SZ",
            "timestamp": stop_time,
            "open": 9.4,
            "high": 9.45,
            "low": 9.35,
            "close": 9.4,
            "volume": 20_000,
        }],
        source="minute_k",
    )
    current = service.account(account["id"])

    assert current["positions"][0]["pending_exit_reason"] == "stop_loss"
    assert current["positions"][0]["pending_exit_date"] == "2026-08-27"
    assert current["positions"][0]["available_qty"] == 0
    assert current["positions"][0]["locked_qty"] == 1_000
    assert len(current["fills"]) == 1
    assert any(
        event["event_type"] == "EXIT_TRIGGERED_T1_LOCKED"
        for event in current["timeline"]
    )


def test_settlement_and_reconciliation_are_idempotent(tmp_path):
    service = _service(tmp_path)
    account, order_id = _account_with_buy_order(service)
    service.ledger.assign_due_date(order_id, TRADE_DAY, {"market_observed": True})
    service.ledger.execute_fill(
        order_id, price=10, quantity=1_000, quote_at=OPEN_TIME, source="open_quote"
    )
    settlement_day = date(2026, 8, 28)

    service.ledger.settle_account(account["id"], settlement_day, source="15:05_close")
    service.ledger.settle_account(account["id"], settlement_day, source="15:05_close")
    current = service.account(account["id"])

    assert current["positions"][0]["hold_days"] == 1
    assert len(current["nav"]) == 1
    assert len([e for e in current["timeline"] if e["event_type"] == "ACCOUNT_SETTLED"]) == 1
    assert current["reconciliation"]["ok"] is True


def test_signal_day_without_orders_is_marked_once(tmp_path):
    ledger = PaperLedger(tmp_path)
    account = ledger.create_account(
        name="空信号模拟盘", baseline_date=SIGNAL_DAY, config=_config()
    )

    ledger.mark_signal_day(account["id"], SIGNAL_DAY)
    ledger.mark_signal_day(account["id"], SIGNAL_DAY)

    assert ledger.get_account(account["id"])["last_processed_date"] == "2026-08-26"


def test_delete_only_hides_selected_account_and_retains_audit_ledger(tmp_path):
    store = PaperTradingStore(tmp_path)
    first = store.create(name="待删除模拟盘", start_date=SIGNAL_DAY, config=_config())
    second = store.create(name="保留模拟盘", start_date=SIGNAL_DAY, config=_config())

    receipt = store.delete(first["id"])

    assert receipt["id"] == first["id"]
    with pytest.raises(KeyError):
        store.get(first["id"])
    assert store.get(second["id"])["name"] == "保留模拟盘"
    assert PaperLedger(tmp_path).account_row(
        first["id"], include_deleted=True
    )["status"] == "deleted"


def test_legacy_migration_keeps_snapshot_but_does_not_import_fake_fills(tmp_path):
    legacy_root = tmp_path / "paper_trading" / "accounts"
    legacy_root.mkdir(parents=True)
    payload = {
        "id": "legacy123456",
        "name": "旧模拟盘",
        "created_at": "2026-08-27T00:16:00+08:00",
        "baseline_date": "2026-08-26",
        "config": _config(),
        "result": {
            "trades": [{"symbol": "000001.SZ", "entry_price": 9.9}],
            "pending_orders": [{
                "symbol": "000001.SZ",
                "name": "平安银行",
                "signal_date": "2026-08-26",
                "score": 70,
                "status": "pending",
            }],
        },
    }
    (legacy_root / "legacy123456.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    account = _service(tmp_path).account("legacy123456")

    assert len(account["orders"]) == 1
    assert account["orders"][0]["status"] == "PLANNED"
    assert account["fills"] == []
    assert any(
        event["event_type"] == "LEGACY_REPLAY_MIGRATED" for event in account["timeline"]
    )


def test_delete_account_api_targets_route_account_id(tmp_path):
    state = SimpleNamespace(repo=FakeRepo(tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    first = PaperTradingStore(tmp_path).create(
        name="API 删除模拟盘", start_date=SIGNAL_DAY, config=_config()
    )
    second = PaperTradingStore(tmp_path).create(
        name="API 保留模拟盘", start_date=SIGNAL_DAY, config=_config()
    )

    assert paper_api.delete_account(first["id"], request)["id"] == first["id"]
    assert paper_api.get_account(second["id"], request)["id"] == second["id"]
    with pytest.raises(paper_api.HTTPException) as exc:
        paper_api.delete_account(first["id"], request)
    assert exc.value.status_code == 404


def test_create_account_api_freezes_exit_mode_without_running_backtest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paper_api,
        "cn_now",
        lambda: datetime(2026, 8, 27, 10, 0, tzinfo=CN_TZ),
    )
    monkeypatch.setattr(
        paper_api.StrategyEngine,
        "validate_context",
        lambda *_args, **_kwargs: None,
    )
    state = SimpleNamespace(
        repo=FakeRepo(tmp_path),
        strategy_engine=SimpleNamespace(
            get=lambda _strategy_id: SimpleNamespace(name="新低反转")
        ),
        capabilities=SimpleNamespace(has=lambda _cap: False),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    account = paper_api.create_account(
        paper_api.PaperTradingCreateRequest(
            name="API 事件账户",
            strategy_id="n_day_low_reversal",
            initial_capital=200_000,
            exit_mode="eod",
        ),
        request,
    )

    assert account["execution_policy"] == "event_driven"
    assert account["config"]["exit_mode"] == "eod"
    assert account["orders"] == []


def test_intraday_account_requires_realtime_or_minute_capability(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paper_api.StrategyEngine,
        "validate_context",
        lambda *_args, **_kwargs: None,
    )
    state = SimpleNamespace(
        repo=FakeRepo(tmp_path),
        strategy_engine=SimpleNamespace(get=lambda _strategy_id: object()),
        capabilities=SimpleNamespace(has=lambda _cap: False),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    with pytest.raises(paper_api.HTTPException) as exc:
        paper_api.create_account(
            paper_api.PaperTradingCreateRequest(
                name="无行情盘中账户",
                strategy_id="n_day_low_reversal",
                exit_mode="intraday",
            ),
            request,
        )

    assert exc.value.status_code == 403
    assert "实时行情或分钟行情" in exc.value.detail


def test_create_api_requires_an_explicit_exit_mode():
    with pytest.raises(ValidationError):
        paper_api.PaperTradingCreateRequest(
            name="未选择退出模式",
            strategy_id="n_day_low_reversal",
        )


def test_direct_fill_refuses_sell_when_t1_locked(tmp_path):
    service = _service(tmp_path)
    account, buy_order = _account_with_buy_order(service)
    service.ledger.assign_due_date(buy_order, TRADE_DAY, {})
    service.ledger.execute_fill(
        buy_order, price=10, quantity=1_000, quote_at=OPEN_TIME, source="open_quote"
    )
    _, sell_order, _ = service.ledger.record_signal_and_order(
        account_id=account["id"],
        strategy_id="n_day_low_reversal",
        symbol="000001.SZ",
        name="平安银行",
        side="SELL",
        signal_date=TRADE_DAY,
        score=None,
        reason="stop_loss",
        signal_ref=None,
        requested_qty=1_000,
        target_amount=9_400,
        target_weight=0,
        planned_session="NEXT_QUOTE",
    )
    service.ledger.assign_due_date(sell_order, TRADE_DAY, {})

    with pytest.raises(PaperLedgerError, match="T\\+1"):
        service.ledger.execute_fill(
            sell_order,
            price=9.4,
            quantity=1_000,
            quote_at=datetime(2026, 8, 27, 10, 5, tzinfo=CN_TZ),
            source="minute_k",
        )


def test_soft_deleted_account_is_removed_from_quote_subscription(tmp_path):
    service = _service(tmp_path)
    account, _ = _account_with_buy_order(service)

    assert service.subscription_symbols() == {"000001.SZ"}

    service.ledger.delete_account(account["id"])

    assert service.subscription_symbols() == set()
