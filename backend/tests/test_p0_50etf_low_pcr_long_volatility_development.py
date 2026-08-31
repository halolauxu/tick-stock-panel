from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_50etf_low_pcr_long_volatility_development.py"
)
SPEC = importlib.util.spec_from_file_location("low_pcr_long_volatility", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _business_dates(start: date, count: int) -> list[date]:
    output = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return output


def test_pcr_zscore_uses_prior_twenty_days_and_open_interest() -> None:
    dates = _business_dates(date(2015, 2, 9), 22)
    master = pl.DataFrame(
        {
            "contract": ["C", "P"],
            "call_put": ["C", "P"],
            "opt_multiplier": [10_000.0, 10_000.0],
        }
    )
    rows = []
    ratios = [1.0, 2.0] * 10 + [1.5, 0.1]
    for trade_date, ratio in zip(dates, ratios, strict=True):
        rows.extend(
            [
                {
                    "contract": "C",
                    "date": trade_date,
                    "open_interest": 1_000.0,
                },
                {
                    "contract": "P",
                    "date": trade_date,
                    "open_interest": 1_000.0 * ratio,
                },
            ]
        )
    fund = pl.DataFrame({"date": dates, "close": [2.5] * len(dates)})
    result = study.build_pcr_signals(master, fund, pl.DataFrame(rows))
    last = result.tail(1).to_dicts()[0]
    assert last["pcr_oi"] == pytest.approx(0.1)
    assert last["pcr_mean_20"] == pytest.approx(1.525)
    assert last["regime"] == "LOW"


def _master(maturity: date) -> pl.DataFrame:
    return pl.DataFrame(
        [
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
            for contract, call_put, strike in (
                ("C24", "C", 2.4),
                ("P24", "P", 2.4),
                ("C25", "C", 2.5),
                ("P25", "P", 2.5),
            )
        ]
    )


def test_trade_selection_uses_signal_close_nearest_eligible_atm_and_liquidity() -> None:
    dates = _business_dates(date(2015, 3, 2), 10)
    signal_date = dates[1]
    maturity = dates[8]
    signals = pl.DataFrame(
        {
            "date": [signal_date],
            "underlying_close": [2.47],
            "pcr_oi": [0.2],
            "pcr_mean_20": [1.0],
            "pcr_std_20": [0.3],
            "pcr_z": [-2.5],
            "regime": ["LOW"],
        }
    )
    option_rows = [
        {
            "contract": contract,
            "date": signal_date,
            "volume": 100.0,
            "open_interest": 500.0,
        }
        for contract in ("C24", "P24", "C25", "P25")
    ]
    trades = study.build_trades(
        signals,
        _master(maturity),
        pl.DataFrame({"date": dates, "close": [2.47] * len(dates)}),
        pl.DataFrame(option_rows),
        regime="LOW",
    )
    assert trades[0]["status"] == "READY"
    assert trades[0]["entry_date"] == dates[2]
    assert trades[0]["maturity_date"] == maturity
    assert trades[0]["legs"]["call"]["strike"] == 2.5
    assert trades[0]["legs"]["put"]["strike"] == 2.5


def _ready_trade(entry_date: date) -> dict:
    return {
        "date": entry_date - timedelta(days=1),
        "signal_date": entry_date - timedelta(days=1),
        "entry_date": entry_date,
        "maturity_date": entry_date + timedelta(days=10),
        "status": "READY",
        "reason": None,
        "legs": {
            "call": {
                "contract": "C",
                "strike": 2.5,
                "tick": 0.0001,
                "multiplier": 10_000.0,
            },
            "put": {
                "contract": "P",
                "strike": 2.5,
                "tick": 0.0001,
                "multiplier": 10_000.0,
            },
        },
    }


def test_account_uses_adverse_ticks_fees_capped_risk_and_balanced_ledger() -> None:
    entry_date = date(2020, 1, 2)
    options = pl.DataFrame(
        [
            {
                "contract": "C",
                "date": entry_date,
                "open": 0.05,
                "close": 0.08,
            },
            {
                "contract": "P",
                "date": entry_date,
                "open": 0.05,
                "close": 0.08,
            },
        ]
    )
    result = study.simulate_account([_ready_trade(entry_date)], options, [entry_date], 200_000.0)
    record = result["records"][0]
    assert record["contracts"] == 9
    assert record["gross_pnl"] == pytest.approx(5_400.0)
    assert record["cost"] == pytest.approx(216.0)
    assert record["net_pnl"] == pytest.approx(5_184.0)
    assert result["ending_equity"] == pytest.approx(205_184.0)
    assert result["max_ledger_error"] == pytest.approx(0.0)


def test_account_rejects_missing_close_without_backfill() -> None:
    entry_date = date(2020, 1, 2)
    options = pl.DataFrame(
        [
            {"contract": "C", "date": entry_date, "open": 0.05, "close": 0.08},
            {"contract": "P", "date": entry_date, "open": 0.05, "close": None},
        ]
    )
    result = study.simulate_account([_ready_trade(entry_date)], options, [entry_date], 200_000.0)
    assert result["trades"] == 0
    assert result["reject_reasons"] == {"ENTRY_OPEN_OR_CLOSE_MISSING": 1}


def test_gate_requires_return_control_increment_and_concentration() -> None:
    candidate = {
        "annualized_return": 0.60,
        "max_drawdown": -0.20,
        "positive_years": 5,
        "trades": 40,
        "execution_rate": 0.95,
        "largest_positive_trade_share": 0.30,
        "max_ledger_error": 0.0,
    }
    control = {"annualized_return": 0.55}
    benchmark = {"annualized_return": 0.20}
    decision = study.evaluate_gate(candidate, control, benchmark)
    assert decision["passed"] is False
    assert decision["checks"]["high_pcr_control_increment_at_least_10pp"] is False
    assert decision["checks"]["largest_positive_trade_share_at_most_25pct"] is False
