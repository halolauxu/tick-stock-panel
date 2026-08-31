from __future__ import annotations

import importlib.util
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_gold_etf_dispersion_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_gold_etf_dispersion_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _session_times(trade_date: date) -> list[datetime]:
    morning = [
        datetime.combine(trade_date, time(9, 30)) + timedelta(minutes=offset)
        for offset in range(121)
    ]
    afternoon = [
        datetime.combine(trade_date, time(13, 1)) + timedelta(minutes=offset)
        for offset in range(120)
    ]
    return morning + afternoon


def _day_frame(*, negative_basket: bool = False, exit_volume: float = 1_000_000.0) -> pl.DataFrame:
    trade_date = date(2025, 8, 1)
    rows = []
    laggard = study.SYMBOLS[-1]
    for symbol in study.SYMBOLS:
        for timestamp in _session_times(trade_date):
            close = 10.01
            open_price = 10.01
            if timestamp.time() == time(9, 30):
                open_price = close = 10.0
            if timestamp.time() == time(10, 0):
                close = 9.97 if symbol == laggard else 9.99 if negative_basket else 10.01
            if symbol == laggard and timestamp.time() in {time(10, 1), time(10, 2), time(10, 3), time(10, 4)}:
                open_price = 10.0
                close = 9.98
            if symbol == laggard and timestamp.time() == time(10, 5):
                open_price = 10.0
                close = 10.01
            if symbol == laggard and timestamp.time() == time(10, 6):
                open_price = close = 10.1
            volume = 1_000_000.0
            if symbol == laggard and timestamp.time() >= time(10, 6):
                volume = exit_volume
            rows.append(
                {
                    "symbol": symbol,
                    "datetime": timestamp,
                    "open": open_price,
                    "high": max(open_price, close),
                    "low": min(open_price, close),
                    "close": close,
                    "volume": volume,
                    "amount": 400_000.0,
                }
            )
    return pl.DataFrame(rows)


def test_signal_uses_nonnegative_median_and_selects_only_laggard() -> None:
    signals = study.build_signals(_day_frame())

    assert len(signals) == 1
    assert signals[0]["symbol"] == study.SYMBOLS[-1]
    assert signals[0]["dispersion"] == pytest.approx(0.004)


def test_signal_rejects_negative_gold_basket_even_with_dispersion() -> None:
    assert study.build_signals(_day_frame(negative_basket=True)) == []


def test_simulation_uses_next_minute_open_volume_cap_tick_fees_and_ledger() -> None:
    minutes = _day_frame()
    signals = study.build_signals(minutes)
    result = study.simulate_account(minutes, signals, 200_000.0)

    record = result["records"][0]
    assert record["status"] == "CLOSED_INTRADAY"
    assert record["shares"] == 10_000
    assert record["gross_pnl"] == pytest.approx(1_000.0)
    assert record["net_pnl"] == pytest.approx(919.7)
    assert record["cost"] == pytest.approx(80.3)
    assert result["ending_equity"] == pytest.approx(200_919.7)
    assert result["total_slippage"] == pytest.approx(20.0)
    assert result["total_commission"] == pytest.approx(60.3)
    assert result["ledger_error"] == pytest.approx(0.0)


def test_zero_exit_capacity_keeps_real_residual_and_fails_overnight_gate() -> None:
    minutes = _day_frame(exit_volume=0.0)
    signals = study.build_signals(minutes)
    result = study.simulate_account(minutes, signals, 200_000.0)

    assert result["overnight_failures"] == 1
    assert result["ending_market_value"] > 0
    assert result["records"][-1]["status"] == "OPEN_RESIDUAL"
    assert study.evaluate_gate(result)["no_overnight_residual"] is False


def test_gate_requires_every_frozen_condition() -> None:
    account = {
        "annualized_return": 0.50,
        "max_drawdown": -0.25,
        "intraday_trades": 40,
        "positive_months": 4,
        "signal_execution_rate": 0.80,
        "overnight_failures": 0,
        "ending_market_value": 0.0,
        "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
