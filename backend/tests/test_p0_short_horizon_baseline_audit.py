from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_short_horizon_baseline_audit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_short_horizon_baseline_audit", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_reconstructs_holding_clock_cooldown_and_settlement() -> None:
    trading_dates = [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 12),
        date(2026, 1, 13),
        date(2026, 1, 14),
        date(2026, 1, 15),
        date(2026, 1, 16),
        date(2026, 1, 19),
    ]
    orders = [
        {
            "date": "2026-01-02",
            "symbol": "600001.SH",
            "family": "main_board_microcap",
            "side": "BUY",
            "status": "FILLED",
            "cash_delta": -1000.0,
        },
        {
            "date": "2026-01-15",
            "symbol": "600001.SH",
            "family": "main_board_microcap",
            "side": "SELL",
            "status": "FILLED",
            "cash_delta": 1100.0,
        },
        {
            "date": "2026-01-16",
            "symbol": "600001.SH",
            "family": "main_board_microcap",
            "side": "BUY",
            "status": "FILLED",
            "cash_delta": -1050.0,
        },
        {
            "date": "2026-01-05",
            "symbol": "600002.SH",
            "family": "main_board_microcap",
            "side": "BUY",
            "status": "FILLED",
            "cash_delta": -1000.0,
        },
        {
            "date": "2026-01-19",
            "symbol": "600002.SH",
            "family": "main_board_microcap",
            "side": "SELL",
            "status": "FILLED",
            "cash_delta": 900.0,
        },
        {
            "date": "2026-01-06",
            "symbol": "600003.SH",
            "family": "main_board_microcap",
            "side": "BUY",
            "status": "FILLED",
            "cash_delta": -500.0,
        },
        {
            "date": "2026-01-07",
            "symbol": "600099.SH",
            "family": "main_board_microcap",
            "side": "SELL",
            "status": "FILLED",
            "cash_delta": 100.0,
        },
    ]
    settlements = [
        {
            "date": "2026-01-07",
            "symbol": "600003.SH",
            "status": "DELISTED_WRITE_OFF",
            "recovery_value": 0.0,
        }
    ]

    result = study.reconstruct_lifecycles(orders, settlements, trading_dates)
    cycles = result["cycles"]
    summary = study.summarize_lifecycles(cycles)

    first = next(
        row
        for row in cycles
        if row["symbol"] == "600001.SH" and row["closed"]
    )
    assert first["holding_sessions"] == 10
    assert first["cash_pnl"] == pytest.approx(100.0)

    long_cycle = next(row for row in cycles if row["symbol"] == "600002.SH")
    assert long_cycle["holding_sessions"] == 11
    assert long_cycle["cash_pnl"] == pytest.approx(-100.0)

    settlement = next(row for row in cycles if row["symbol"] == "600003.SH")
    assert settlement["exit_type"] == "DELISTED_WRITE_OFF"
    assert settlement["cash_pnl"] == pytest.approx(-500.0)

    reopened = next(
        row
        for row in cycles
        if row["symbol"] == "600001.SH" and not row["closed"]
    )
    assert reopened["reentry_gap_sessions"] == 1
    assert reopened["observed_sessions"] == 2

    assert summary["closed_cycles"] == 3
    assert summary["open_cycles"] == 1
    assert summary["open_over_10_cycles"] == 0
    assert summary["over_10_cycles"] == 1
    assert summary["reentries_within_10_sessions"] == 1
    assert summary["next_session_reentries"] == 1
    assert result["issues"] == [
        "sell_without_open_position:600099.SH:2026-01-07"
    ]


def test_summary_separates_families_and_exit_years() -> None:
    cycles = [
        {
            "symbol": "600001.SH",
            "family": "main_board_microcap",
            "entry_date": "2025-12-30",
            "exit_date": "2026-01-06",
            "exit_type": "SELL",
            "closed": True,
            "holding_sessions": 5,
            "observed_sessions": 5,
            "reentry_gap_sessions": None,
            "cash_pnl": 120.0,
        },
        {
            "symbol": "600002.SH",
            "family": "idiosyncratic_forecast",
            "entry_date": "2026-02-03",
            "exit_date": "2026-02-16",
            "exit_type": "SELL",
            "closed": True,
            "holding_sessions": 10,
            "observed_sessions": 10,
            "reentry_gap_sessions": None,
            "cash_pnl": -20.0,
        },
    ]

    result = study.build_cycle_audit(cycles)

    assert result["all"]["closed_cycles"] == 2
    assert result["by_family"]["main_board_microcap"]["cash_pnl"] == 120.0
    assert result["by_family"]["idiosyncratic_forecast"]["cash_pnl"] == -20.0
    assert result["by_exit_year"]["2026"]["closed_cycles"] == 2
    assert result["by_entry_year"]["2025"]["closed_cycles"] == 1
    assert result["by_entry_year"]["2026"]["closed_cycles"] == 1
    assert result["by_family_exit_year"]["main_board_microcap"]["2026"][
        "cash_pnl"
    ] == 120.0
    assert result["by_family_exit_year"]["idiosyncratic_forecast"]["2026"][
        "cash_pnl"
    ] == -20.0
