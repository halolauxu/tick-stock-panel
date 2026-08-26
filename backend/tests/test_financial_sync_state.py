from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.api import financials as financials_api
from app.services import financial_sync
from app.tickflow.capabilities import CapabilitySet


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_single_table_trigger_exposes_and_clears_active_table(tmp_path, monkeypatch):
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    monkeypatch.setattr(financial_sync, "_financial_is_custom", lambda: True)

    entered = threading.Event()
    release = threading.Event()

    def fake_run_body(table):
        assert table == "income"
        scheduler._progress_callback(table)(75, 100, 3210, 2)
        entered.set()
        assert release.wait(1)
        return {"income": 12}

    monkeypatch.setattr(scheduler, "_run_body", fake_run_body)

    assert scheduler.trigger("income") == {"started": True}
    assert entered.wait(1)
    assert scheduler.sync_state == {
        "syncing": True,
        "sync_scope": "single",
        "syncing_table": "income",
        "sync_progress": {
            "symbols_done": 75,
            "symbols_total": 100,
            "rows_received": 3210,
            "failures": 2,
        },
    }

    release.set()
    assert _wait_until(lambda: not scheduler.is_syncing)
    assert scheduler.sync_state == {
        "syncing": False,
        "sync_scope": None,
        "syncing_table": None,
        "sync_progress": None,
    }


def test_full_sync_reports_each_active_table(tmp_path, monkeypatch):
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    observed: list[str | None] = []

    monkeypatch.setattr(financial_sync, "_get_symbols", lambda _data_dir: [])
    monkeypatch.setattr(
        financial_sync,
        "_sync_history_table_for_symbols",
        lambda table, *_args, **_kwargs: (
            observed.append(scheduler.sync_state["syncing_table"]) or 0
        ),
    )
    monkeypatch.setattr(financial_sync, "_refresh_financials_views", lambda _data_dir: None)
    monkeypatch.setattr(scheduler, "_record_sync", lambda _table: None)

    with scheduler._lock:
        scheduler._begin_sync(None)
    try:
        scheduler._run_body(None)
    finally:
        with scheduler._lock:
            scheduler._finish_sync()

    assert observed == list(financial_sync.FINANCIAL_TABLES)


def test_trigger_skips_single_table_already_synced_today(tmp_path, monkeypatch):
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    scheduler._last_sync = {"metrics": datetime.now(UTC).isoformat()}
    parquet = tmp_path / "financials" / "metrics" / "part.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"existing-data")
    monkeypatch.setattr(financial_sync, "_financial_is_custom", lambda: True)

    assert scheduler.trigger("metrics") == {
        "started": False,
        "reason": "already up to date",
        "tables": ["metrics"],
    }
    assert scheduler.is_syncing is False


def test_trigger_starts_single_table_synced_on_previous_day(tmp_path, monkeypatch):
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    scheduler._last_sync = {"metrics": (datetime.now(UTC) - timedelta(days=1)).isoformat()}
    parquet = tmp_path / "financials" / "metrics" / "part.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"existing-data")
    monkeypatch.setattr(financial_sync, "_financial_is_custom", lambda: True)
    entered = threading.Event()

    def fake_run_body(table):
        assert table == "metrics"
        entered.set()
        return {"metrics": 1}

    monkeypatch.setattr(scheduler, "_run_body", fake_run_body)

    assert scheduler.trigger("metrics") == {"started": True}
    assert entered.wait(1)
    assert _wait_until(lambda: not scheduler.is_syncing)


def test_full_sync_skips_tables_already_synced_today(tmp_path, monkeypatch):
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    scheduler._last_sync = {"metrics": datetime.now(UTC).isoformat()}
    parquet = tmp_path / "financials" / "metrics" / "part.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"existing-data")
    observed: list[str] = []

    monkeypatch.setattr(financial_sync, "_get_symbols", lambda _data_dir: [])
    monkeypatch.setattr(
        financial_sync,
        "_sync_history_table_for_symbols",
        lambda table, *_args, **_kwargs: observed.append(table) or 0,
    )
    monkeypatch.setattr(financial_sync, "_refresh_financials_views", lambda _data_dir: None)
    monkeypatch.setattr(scheduler, "_record_sync", lambda _table: None)

    with scheduler._lock:
        scheduler._begin_sync(None)
    try:
        scheduler._run_body(None)
    finally:
        with scheduler._lock:
            scheduler._finish_sync()

    assert observed == [table for table in financial_sync.FINANCIAL_TABLES if table != "metrics"]


def test_status_returns_server_sync_scope_and_active_table(tmp_path, monkeypatch):
    monkeypatch.setattr(financials_api, "_financial_allowed", lambda _capset: True)
    scheduler = SimpleNamespace(
        last_sync={"balance_sheet": "2026-08-26T08:00:00+00:00"},
        sync_state={
            "syncing": True,
            "sync_scope": "single",
            "syncing_table": "balance_sheet",
            "sync_progress": {
                "symbols_done": 1200,
                "symbols_total": 5550,
                "rows_received": 55200,
                "failures": 0,
            },
        },
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                capabilities=CapabilitySet(),
                repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
                financial_scheduler=scheduler,
            )
        )
    )

    result = financials_api.financial_status(request)

    assert result["syncing"] is True
    assert result["sync_scope"] == "single"
    assert result["syncing_table"] == "balance_sheet"
    assert result["sync_progress"] == {
        "symbols_done": 1200,
        "symbols_total": 5550,
        "rows_received": 55200,
        "failures": 0,
    }
