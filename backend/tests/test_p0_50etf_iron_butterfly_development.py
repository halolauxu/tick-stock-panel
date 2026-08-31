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
        / "run_p0_50etf_iron_butterfly_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_50etf_iron_butterfly_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _master(maturity: date) -> pl.DataFrame:
    rows = []
    for contract, call_put, strike in (
        ("ATM-C", "C", 100.0),
        ("ATM-P", "P", 100.0),
        ("WING-P", "P", 90.0),
        ("WING-C", "C", 110.0),
    ):
        rows.append(
            {
                "contract": contract,
                "call_put": call_put,
                "exercise_price": strike,
                "opt_multiplier": 10_000.0,
                "min_price_chg": 0.0001,
                "maturity_date": maturity,
                "list_date": date(2015, 1, 1),
                "delist_date": maturity,
            }
        )
    return pl.DataFrame(rows)


def _fund() -> pl.DataFrame:
    dates = [date(2015, 2, 9) + timedelta(days=offset) for offset in range(30)]
    return pl.DataFrame({"date": dates, "close": [100.0] * len(dates)})


def _liquidity_rows(trade_date: date, failing_contract: str | None = None) -> list[dict]:
    return [
        {
            "contract": contract,
            "date": trade_date,
            "open": 0.1,
            "close": 0.1,
            "settle": 0.1,
            "volume": 99.0 if contract == failing_contract else 100.0,
            "open_interest": 500.0,
        }
        for contract in ("ATM-C", "ATM-P", "WING-P", "WING-C")
    ]


def test_build_cycles_selects_atm_wings_and_prior_day_liquidity() -> None:
    fund = _fund()
    maturity = fund["date"][-1] + timedelta(days=1)
    prior_date = fund["date"][8]
    result = study.build_cycles(
        _master(maturity), fund, pl.DataFrame(_liquidity_rows(prior_date))
    )

    assert len(result) == 1
    assert result[0]["status"] == "READY"
    assert result[0]["signal_date"] == fund["date"][9]
    assert result[0]["entry_date"] == fund["date"][10]
    assert result[0]["exit_date"] == fund["date"][25]
    assert result[0]["legs"]["short_call"]["strike"] == 100.0
    assert result[0]["legs"]["long_put"]["strike"] == 90.0
    assert result[0]["legs"]["long_call"]["strike"] == 110.0


def test_build_cycles_does_not_replace_failed_prior_day_liquidity_with_signal_day() -> None:
    fund = _fund()
    maturity = fund["date"][-1] + timedelta(days=1)
    prior_date = fund["date"][8]
    signal_date = fund["date"][9]
    rows = _liquidity_rows(prior_date, failing_contract="ATM-C")
    rows.extend(_liquidity_rows(signal_date))

    result = study.build_cycles(_master(maturity), fund, pl.DataFrame(rows))

    assert result[0]["status"] == "REJECTED"
    assert result[0]["reason"] == "PRIOR_DAY_LIQUIDITY_FAILED"


def _ready_cycle(entry_date: date, exit_date: date) -> dict:
    legs = {
        "short_call": {"contract": "SC", "strike": 2.3, "tick": 0.0001, "multiplier": 10_000.0},
        "short_put": {"contract": "SP", "strike": 2.3, "tick": 0.0001, "multiplier": 10_000.0},
        "long_put": {"contract": "LP", "strike": 2.1, "tick": 0.0001, "multiplier": 10_000.0},
        "long_call": {"contract": "LC", "strike": 2.5, "tick": 0.0001, "multiplier": 10_000.0},
    }
    return {
        "maturity_date": exit_date + timedelta(days=5),
        "signal_date": entry_date - timedelta(days=1),
        "entry_date": entry_date,
        "exit_date": exit_date,
        "status": "READY",
        "reason": None,
        "legs": legs,
    }


def _price_rows(entry_date: date, exit_date: date, omit_exit: bool = False) -> pl.DataFrame:
    rows = []
    for contract, entry_open, exit_open in (
        ("SC", 0.10, 0.03),
        ("SP", 0.10, 0.03),
        ("LP", 0.02, 0.005),
        ("LC", 0.02, 0.005),
    ):
        rows.append(
            {
                "contract": contract,
                "date": entry_date,
                "open": entry_open,
                "close": entry_open,
                "settle": entry_open,
            }
        )
        if not omit_exit:
            rows.append(
                {
                    "contract": contract,
                    "date": exit_date,
                    "open": exit_open,
                    "close": exit_open,
                    "settle": exit_open,
                }
            )
    return pl.DataFrame(rows)


def test_simulation_applies_capped_risk_slippage_fees_daily_marks_and_ledger() -> None:
    entry_date = date(2020, 1, 2)
    exit_date = date(2020, 1, 3)
    result = study.simulate_account(
        [_ready_cycle(entry_date, exit_date)],
        _price_rows(entry_date, exit_date),
        [entry_date, exit_date],
        200_000.0,
    )

    record = result["records"][0]
    assert record["sets"] == 45
    assert record["gross_pnl"] == pytest.approx(49_500.0)
    assert record["cost"] == pytest.approx(2_160.0)
    assert record["net_pnl"] == pytest.approx(47_340.0)
    assert result["ending_equity"] == pytest.approx(247_340.0)
    assert result["max_drawdown"] == pytest.approx(-0.0054)
    assert result["ledger_error"] == pytest.approx(0.0)


def test_simulation_rejects_missing_exit_open_without_backfill() -> None:
    entry_date = date(2020, 1, 2)
    exit_date = date(2020, 1, 3)
    result = study.simulate_account(
        [_ready_cycle(entry_date, exit_date)],
        _price_rows(entry_date, exit_date, omit_exit=True),
        [entry_date, exit_date],
        200_000.0,
    )

    assert result["trades"] == 0
    assert result["records"][0]["reason"] == "ENTRY_OR_EXIT_OPEN_MISSING"


def test_gate_requires_every_frozen_condition() -> None:
    account = {
        "annualized_return": 0.50,
        "max_drawdown": -0.25,
        "trades": 30,
        "positive_years": 4,
        "execution_rate": 0.90,
        "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
