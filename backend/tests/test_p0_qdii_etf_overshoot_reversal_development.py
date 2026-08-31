from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))


def _load_module():
    path = RESEARCH / "run_p0_qdii_etf_overshoot_reversal_development.py"
    spec = importlib.util.spec_from_file_location("p0_qdii_etf_reversal_dev", path)
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


def _day(positive_median: bool = False) -> pl.DataFrame:
    trade_date = date(2025, 8, 1)
    laggard = study.engine.SYMBOLS[-1]
    rows = []
    for index, symbol in enumerate(study.engine.SYMBOLS):
        for timestamp in _session_times(trade_date):
            close = 2.0
            if timestamp.time() == time(10, 0):
                if symbol == laggard:
                    close = 1.984
                elif index < 8:
                    close = 2.004 if positive_median else 1.996
            rows.append(
                {
                    "symbol": symbol, "datetime": timestamp, "open": 2.0,
                    "high": max(2.0, close), "low": min(2.0, close), "close": close,
                    "volume": 1_000_000.0,
                    "amount": 200_000.0 if index < 8 or symbol == laggard else 10_000.0,
                }
            )
    return pl.DataFrame(rows)


def test_signal_selects_relative_laggard_only_in_negative_cross_section() -> None:
    signals = study.build_signals(_day())

    assert len(signals) == 1
    assert signals[0]["symbol"] == study.engine.SYMBOLS[-1]
    assert signals[0]["laggard_return"] == pytest.approx(-0.008)
    assert signals[0]["median_return"] == pytest.approx(-0.002)
    assert signals[0]["lag"] == pytest.approx(0.006)


def test_signal_rejects_positive_cross_section() -> None:
    assert study.build_signals(_day(positive_median=True)) == []


def test_reversal_reuses_the_same_strict_execution_gate() -> None:
    account = {
        "annualized_return": 0.50, "max_drawdown": -0.25, "intraday_trades": 40,
        "positive_months": 4, "signal_execution_rate": 0.80,
        "overnight_failures": 0, "ending_market_value": 0.0, "ledger_error": 0.0,
    }

    assert all(study.engine.evaluate_gate(account).values())
