from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from app.api import paper_trading as paper_api
from app.market_time import CN_TZ
from app.services import paper_trading


def _config() -> dict:
    return {
        "strategy_id": "n_day_low_reversal",
        "asset_type": "stock",
        "symbols": None,
        "params": {"lookback": 20},
        "overrides": {"max_hold_days": 5},
        "entry_fill": "open_t+1",
        "exit_fill": "open_t+1",
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


def test_account_store_persists_frozen_backtest_config(tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)

    created = store.create(
        name="新低反转模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
        created_at=datetime(2026, 8, 26, 16, 0, tzinfo=CN_TZ),
    )

    loaded = paper_trading.PaperTradingStore(tmp_path).get(created["id"])
    assert loaded["status"] == "active"
    assert loaded["schema_version"] == 3
    assert loaded["baseline_date"] == "2026-08-26"
    assert loaded["signal_start_date"] == "2026-08-26"
    assert loaded["start_date"] == "2026-08-26"
    assert loaded["activation_policy"] == "completed_baseline_or_next_forward_day"
    assert loaded["config"] == _config()
    assert loaded["last_processed_date"] is None
    assert loaded["result"] is None


def test_run_account_reuses_backtest_worker_and_is_idempotent(monkeypatch, tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)
    created = store.create(
        name="新低反转模拟盘",
        start_date=date(2026, 8, 20),
        config=_config(),
        created_at=datetime(2026, 8, 20, 16, 0, tzinfo=CN_TZ),
    )
    tasks: list[dict] = []

    def fake_run(task):
        tasks.append(task)
        return {
            "run_id": "paper-run",
            "config": task["config"],
            "stats": {"total_return": 0.05},
            "equity_curve": [{
                "date": "2026-08-26",
                "value": 210_000,
                "cash": 110_000,
                "positions": 1,
                "exposure": 0.4762,
            }],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "trades": [],
            "open_positions": [{"symbol": "000001.SZ"}],
            "pending_orders": [],
            "per_symbol_stats": [],
            "strategy_info": {"id": "n_day_low_reversal", "name": "新低反转"},
            "elapsed_ms": 12,
            "error": None,
        }

    monkeypatch.setattr(paper_trading, "run_worker_task", fake_run)
    state = SimpleNamespace(
        repo=SimpleNamespace(
            store=SimpleNamespace(data_dir=tmp_path),
            latest_enriched_date=lambda asset_type: date(2026, 8, 26),
        ),
    )

    first = paper_trading.run_account(state, created["id"])
    second = paper_trading.run_account(state, created["id"])

    assert first["last_processed_date"] == "2026-08-26"
    assert first["result"]["open_positions"] == [{"symbol": "000001.SZ"}]
    assert second == first
    assert len(tasks) == 1
    assert tasks[0]["kind"] == "backtest"
    assert tasks[0]["config"]["start"] == "2026-08-20"
    assert tasks[0]["config"]["end"] == "2026-08-26"
    assert tasks[0]["config"]["liquidate_on_end"] is False


def test_new_account_waits_for_post_activation_data_without_retroactive_order(monkeypatch, tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)
    created = store.create(
        name="严格前向模拟盘",
        start_date=date(2026, 8, 25),
        config=_config(),
        created_at=datetime(2026, 8, 26, 22, 0, tzinfo=CN_TZ),
    )
    monkeypatch.setattr(
        paper_trading,
        "run_worker_task",
        lambda _task: pytest.fail("基线日不应执行回测或生成历史订单"),
    )
    state = SimpleNamespace(
        repo=SimpleNamespace(
            store=SimpleNamespace(data_dir=tmp_path),
            latest_enriched_date=lambda _asset_type: date(2026, 8, 25),
        ),
    )

    account = paper_trading.run_account(state, created["id"], force=True)

    assert account["last_processed_date"] == "2026-08-25"
    assert account["result"]["pending_orders"] == []
    assert account["result"]["open_positions"] == []
    assert account["execution_state"]["code"] == "waiting_first_data"
    assert "2026-08-26" in account["execution_state"]["detail"]


def test_preopen_account_uses_latest_complete_day_for_next_open(tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)

    created = store.create(
        name="盘前模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
        created_at=datetime(2026, 8, 27, 8, 0, tzinfo=CN_TZ),
    )

    assert created["baseline_date"] == "2026-08-26"
    assert created["signal_start_date"] == "2026-08-26"


def test_v2_account_is_migrated_to_rebuild_complete_baseline_day(tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)
    created = store.create(
        name="旧版模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
        created_at=datetime(2026, 8, 26, 22, 0, tzinfo=CN_TZ),
    )
    legacy = {
        **created,
        "schema_version": 2,
        "signal_start_date": "2026-08-27",
        "start_date": "2026-08-27",
        "last_processed_date": "2026-08-26",
        "result": {"pending_orders": []},
    }
    path = tmp_path / "paper_trading" / "accounts" / f"{created['id']}.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = store.get(created["id"])

    assert migrated["schema_version"] == 3
    assert migrated["signal_start_date"] == "2026-08-26"
    assert migrated["last_processed_date"] is None
    assert migrated["result"] is None
    assert migrated["execution_state"]["code"] == "waiting_rebuild"


def test_corrupt_account_file_fails_closed(tmp_path):
    path = tmp_path / "paper_trading" / "accounts" / "abcdef123456.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(paper_trading.PaperTradingStoreError, match="损坏"):
        paper_trading.PaperTradingStore(tmp_path).get("abcdef123456")
    assert path.read_text(encoding="utf-8") == "{broken"


def test_account_store_delete_removes_only_requested_account(tmp_path):
    store = paper_trading.PaperTradingStore(tmp_path)
    first = store.create(
        name="待删除模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
    )
    second = store.create(
        name="保留模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
    )

    deleted = store.delete(first["id"])

    assert deleted["id"] == first["id"]
    with pytest.raises(KeyError):
        store.get(first["id"])
    assert store.get(second["id"])["name"] == "保留模拟盘"


def test_delete_account_api_returns_receipt_and_missing_account_is_404(tmp_path):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
    )))
    created = paper_trading.PaperTradingStore(tmp_path).create(
        name="API 删除模拟盘",
        start_date=date(2026, 8, 26),
        config=_config(),
    )

    assert paper_api.delete_account(created["id"], request) == {
        "ok": True,
        "id": created["id"],
        "name": "API 删除模拟盘",
    }
    with pytest.raises(paper_api.HTTPException) as exc:
        paper_api.delete_account(created["id"], request)
    assert exc.value.status_code == 404


def test_api_creates_account_from_latest_complete_day(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paper_trading,
        "cn_now",
        lambda: datetime(2026, 8, 26, 22, 0, tzinfo=CN_TZ),
    )
    monkeypatch.setattr(
        paper_api.StrategyEngine,
        "validate_context",
        lambda *_args, **_kwargs: None,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        repo=SimpleNamespace(
            store=SimpleNamespace(data_dir=tmp_path),
            latest_enriched_date=lambda _asset_type: date(2026, 8, 26),
        ),
        strategy_engine=SimpleNamespace(get=lambda _strategy_id: object()),
    )))

    account = paper_api.create_account(
        paper_api.PaperTradingCreateRequest(
            name="新低反转模拟盘",
            strategy_id="n_day_low_reversal",
            initial_capital=200_000,
        ),
        request,
    )

    assert account["baseline_date"] == "2026-08-26"
    assert account["signal_start_date"] == "2026-08-26"
    assert account["start_date"] == "2026-08-26"
    assert account["config"]["strategy_id"] == "n_day_low_reversal"
    assert account["config"]["entry_fill"] == "open_t+1"
    assert paper_api.list_accounts(request)["items"][0]["id"] == account["id"]


def test_manual_api_run_forces_sync_even_when_daily_run_is_paused(monkeypatch):
    calls = []
    expected = {"id": "abcdef123456", "status": "paused"}
    monkeypatch.setattr(
        paper_api,
        "run_account",
        lambda state, account_id, *, force: calls.append((state, account_id, force)) or expected,
    )
    state = SimpleNamespace()
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    assert paper_api.run_paper_account("abcdef123456", request) == expected
    assert calls == [(state, "abcdef123456", True)]


@pytest.mark.parametrize("field", ["entry_fill", "exit_fill"])
def test_after_close_account_rejects_retroactive_close_fill(monkeypatch, tmp_path, field):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        repo=SimpleNamespace(
            store=SimpleNamespace(data_dir=tmp_path),
            latest_enriched_date=lambda _asset_type: date(2026, 8, 26),
        ),
    )))
    body = paper_api.PaperTradingCreateRequest(
        name="不允许倒填",
        strategy_id="n_day_low_reversal",
        **{field: "close_t"},
    )

    with pytest.raises(paper_api.HTTPException, match="收盘") as exc:
        paper_api.create_account(body, request)

    assert exc.value.status_code == 400
