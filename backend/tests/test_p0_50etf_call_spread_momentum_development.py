from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_50etf_call_spread_momentum_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_50etf_call_spread_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _master(maturity: date) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "contract": f"C-{strike}", "call_put": "C", "exercise_price": strike,
                "opt_multiplier": 10_000.0, "min_price_chg": 0.0001,
                "maturity_date": maturity, "list_date": date(2015, 1, 1),
                "delist_date": maturity,
            }
            for strike in (100.0, 105.0, 110.0)
        ]
    )


def test_build_cycles_uses_prior_ma_and_fixed_next_higher_call() -> None:
    dates = [date(2015, 2, 9) + timedelta(days=index) for index in range(100)]
    closes = [100.0] * 100
    closes[79] = 104.0
    fund = pl.DataFrame({"date": dates, "close": closes})
    maturity = dates[-1] + timedelta(days=1)
    prior_date = dates[78]
    options = pl.DataFrame(
        [
            {
                "contract": contract, "date": prior_date, "volume": 100.0,
                "open_interest": 500.0,
            }
            for contract in ("C-105.0", "C-110.0")
        ]
    )

    cycles = study.build_cycles(_master(maturity), fund, options)

    assert len(cycles) == 1
    assert cycles[0]["status"] == "READY"
    assert cycles[0]["signal_date"] == dates[79]
    assert cycles[0]["prior_ma60"] == 100.0
    assert cycles[0]["legs"]["long_call"]["strike"] == 105.0
    assert cycles[0]["legs"]["short_call"]["strike"] == 110.0


def _ready_cycle(entry: date, exit_date: date) -> dict:
    return {
        "maturity_date": exit_date + timedelta(days=5), "signal_date": entry - timedelta(days=1),
        "entry_date": entry, "exit_date": exit_date, "status": "READY", "reason": None,
        "legs": {
            "long_call": {"contract": "LC", "strike": 2.3, "tick": 0.0001, "multiplier": 10_000.0},
            "short_call": {"contract": "SC", "strike": 2.4, "tick": 0.0001, "multiplier": 10_000.0},
        },
    }


def _prices(entry: date, exit_date: date, omit_exit: bool = False) -> pl.DataFrame:
    rows = [
        {"contract": "LC", "date": entry, "open": 0.12, "close": 0.12, "settle": 0.12},
        {"contract": "SC", "date": entry, "open": 0.05, "close": 0.05, "settle": 0.05},
    ]
    if not omit_exit:
        rows.extend(
            [
                {"contract": "LC", "date": exit_date, "open": 0.18, "close": 0.18, "settle": 0.18},
                {"contract": "SC", "date": exit_date, "open": 0.06, "close": 0.06, "settle": 0.06},
            ]
        )
    return pl.DataFrame(rows)


def test_simulation_applies_debit_risk_ticks_fees_marks_and_ledger() -> None:
    entry = date(2020, 1, 2)
    exit_date = date(2020, 1, 3)
    result = study.simulate_account(
        [_ready_cycle(entry, exit_date)], _prices(entry, exit_date), [entry, exit_date], 200_000.0
    )

    record = result["records"][0]
    assert record["spreads"] == 27
    assert record["gross_pnl"] == pytest.approx(13_500.0)
    assert record["cost"] == pytest.approx(648.0)
    assert record["net_pnl"] == pytest.approx(12_852.0)
    assert result["ending_equity"] == pytest.approx(212_852.0)
    assert result["max_drawdown"] == pytest.approx(-0.00162)
    assert result["ledger_error"] == pytest.approx(0.0)


def test_missing_exit_open_is_rejected_without_backfill() -> None:
    entry = date(2020, 1, 2)
    exit_date = date(2020, 1, 3)
    result = study.simulate_account(
        [_ready_cycle(entry, exit_date)],
        _prices(entry, exit_date, omit_exit=True),
        [entry, exit_date],
        200_000.0,
    )

    assert result["trades"] == 0
    assert result["records"][0]["reason"] == "ENTRY_OR_EXIT_OPEN_MISSING"


def test_gate_requires_every_frozen_condition() -> None:
    account = {
        "annualized_return": 0.50, "max_drawdown": -0.25, "trades": 20,
        "positive_years": 4, "execution_rate": 0.90, "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
