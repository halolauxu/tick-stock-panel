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
        / "run_p0_qdii_etf_intraday_momentum_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_qdii_etf_momentum_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _session_times(trade_date: date) -> list[datetime]:
    return [
        datetime.combine(trade_date, time(9, 30)) + timedelta(minutes=offset)
        for offset in range(121)
    ] + [
        datetime.combine(trade_date, time(13, 1)) + timedelta(minutes=offset)
        for offset in range(120)
    ]


def _full_day(*, negative_median: bool = False) -> pl.DataFrame:
    trade_date = date(2025, 8, 1)
    winner = study.SYMBOLS[-1]
    rows = []
    for index, symbol in enumerate(study.SYMBOLS):
        for timestamp in _session_times(trade_date):
            open_price = close = 2.0
            if timestamp.time() == time(10, 0):
                if symbol == winner:
                    close = 2.016
                elif index < 8:
                    close = 1.998 if negative_median else 2.004
            if symbol == winner and timestamp.time() in {
                time(10, 1), time(10, 2), time(10, 3), time(10, 4), time(10, 5)
            }:
                open_price = close = 2.0
            if symbol == winner and timestamp.time() == time(14, 55):
                open_price = close = 2.02
            rows.append(
                {
                    "symbol": symbol, "datetime": timestamp, "open": open_price,
                    "high": max(open_price, close), "low": min(open_price, close),
                    "close": close, "volume": 10_000_000.0,
                    "amount": 200_000.0 if index < 8 or symbol == winner else 10_000.0,
                }
            )
    return pl.DataFrame(rows)


def test_signal_requires_global_risk_on_and_selects_strongest_liquid_fund() -> None:
    signals = study.build_signals(_full_day())

    assert len(signals) == 1
    assert signals[0]["symbol"] == study.SYMBOLS[-1]
    assert signals[0]["winner_return"] == pytest.approx(0.008)
    assert signals[0]["median_return"] == pytest.approx(0.002)


def test_signal_rejects_negative_cross_section_median() -> None:
    assert study.build_signals(_full_day(negative_median=True)) == []


def test_execution_uses_next_minute_capacity_ticks_fees_and_scheduled_exit() -> None:
    minutes = _full_day()
    signal = {
        "date": date(2025, 8, 1), "symbol": study.SYMBOLS[-1],
        "eligible_funds": 9, "median_return": 0.002, "winner_return": 0.008,
        "winner_lead": 0.006, "signal_close": 2.0,
    }
    result = study.simulate_account(minutes, [signal], 200_000.0)

    record = result["records"][0]
    assert record["status"] == "CLOSED_INTRADAY"
    assert record["shares"] == 79_900
    assert record["gross_pnl"] == pytest.approx(1_598.0)
    assert record["net_pnl"] == pytest.approx(1_341.8406)
    assert result["total_slippage"] == pytest.approx(159.8)
    assert result["ledger_error"] == pytest.approx(0.0)


def test_gate_requires_all_frozen_conditions() -> None:
    account = {
        "annualized_return": 0.50, "max_drawdown": -0.25, "intraday_trades": 40,
        "positive_months": 4, "signal_execution_rate": 0.80,
        "overnight_failures": 0, "ending_market_value": 0.0, "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
