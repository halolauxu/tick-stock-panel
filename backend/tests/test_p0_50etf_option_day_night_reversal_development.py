from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_50etf_option_day_night_reversal_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_50etf_option_day_night_reversal_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _master() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "contract": f"{maturity}-{side}-{strike}",
                "call_put": side,
                "exercise_price": strike,
                "opt_multiplier": 10_000.0,
                "min_price_chg": 0.0001,
                "maturity_date": maturity,
                "list_date": date(2015, 1, 1),
                "delist_date": maturity,
            }
            for maturity in (date(2015, 2, 25), date(2015, 3, 25))
            for side in ("C", "P")
            for strike in (2.4, 2.5, 2.6)
        ]
    )


def _fund() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2015, 2, 9), date(2015, 2, 10)],
            "open": [2.48, 2.51],
            "high": [2.51, 2.70],
            "low": [2.45, 2.40],
            "close": [2.49, 2.68],
            "pre_close": [2.47, 2.49],
        }
    )


def _option_rows() -> pl.DataFrame:
    rows = []
    for side in ("C", "P"):
        contract = f"2015-02-25-{side}-2.5"
        rows.extend(
            [
                {
                    "contract": contract,
                    "date": date(2015, 2, 9),
                    "pre_settle": 0.09,
                    "open": 0.09,
                    "close": 0.10,
                    "volume": 100.0,
                    "open_interest": 500.0,
                },
                {
                    "contract": contract,
                    "date": date(2015, 2, 10),
                    "pre_settle": 0.10,
                    "open": 0.12 if side == "C" else 0.08,
                    "close": 0.09 if side == "C" else 0.11,
                    "volume": 200.0,
                    "open_interest": 600.0,
                },
            ]
        )
    return pl.DataFrame(rows)


def test_build_sessions_uses_prior_close_nearest_maturity_and_prior_liquidity() -> None:
    sessions = study.build_sessions(_master(), _fund(), _option_rows())

    assert len(sessions) == 1
    session = sessions[0]
    assert session["status"] == "READY"
    assert session["maturity_date"] == date(2015, 2, 25)
    assert session["strike"] == pytest.approx(2.5)
    assert session["legs"]["call"]["contract"] == "2015-02-25-C-2.5"
    assert session["legs"]["put"]["contract"] == "2015-02-25-P-2.5"


def test_build_sessions_does_not_use_next_day_liquidity_to_rescue_signal() -> None:
    options = _option_rows().with_columns(
        pl.when((pl.col("date") == date(2015, 2, 9)) & pl.col("contract").str.contains("-C-"))
        .then(pl.lit(99.0))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )

    sessions = study.build_sessions(_master(), _fund(), options)

    assert sessions[0]["status"] == "REJECTED"
    assert sessions[0]["reason"] == "SIGNAL_LIQUIDITY_FAILED"


def test_short_option_margin_matches_sse_minimum_formulas() -> None:
    call = study.short_option_margin(
        call_put="C",
        pre_settle=0.10,
        underlying_pre_close=2.50,
        strike=2.60,
        multiplier=10_000.0,
    )
    put = study.short_option_margin(
        call_put="P",
        pre_settle=0.09,
        underlying_pre_close=2.48,
        strike=2.50,
        multiplier=10_000.0,
    )

    assert call == pytest.approx(3_000.0)
    assert put == pytest.approx(3_876.0)


def test_account_applies_two_segments_slippage_fees_limits_and_ledger() -> None:
    sessions = study.build_sessions(_master(), _fund(), _option_rows())

    result = study.simulate_account(
        sessions,
        [date(2015, 2, 9), date(2015, 2, 10)],
        200_000.0,
    )

    record = result["records"][0]
    assert record["status"] == "FILLED"
    assert record["overnight_sets"] == 8
    assert record["intraday_sets"] == 8
    assert record["overnight"]["net_pnl"] == pytest.approx(3_008.0)
    assert record["intraday"]["net_pnl"] == pytest.approx(4_608.0)
    assert record["gross_pnl"] == pytest.approx(8_000.0)
    assert record["cost"] == pytest.approx(384.0)
    assert result["ending_equity"] == pytest.approx(207_616.0)
    assert result["ledger_error"] == pytest.approx(0.0, abs=1e-9)
    assert result["option_account_eligible"] is False
    assert result["eligibility_reason"] == "NOT_ELIGIBLE_FOR_OPTION_ACCOUNT"


def test_new_account_position_limit_caps_large_accounts_at_twenty_sets() -> None:
    sessions = study.build_sessions(_master(), _fund(), _option_rows())

    result = study.simulate_account(
        sessions,
        [date(2015, 2, 9), date(2015, 2, 10)],
        1_000_000.0,
    )

    assert result["records"][0]["overnight_sets"] == 20
    assert result["records"][0]["intraday_sets"] == 20
    assert result["option_account_eligible"] is True


def test_gate_requires_eligible_capital_and_every_frozen_condition() -> None:
    account = {
        "option_account_eligible": True,
        "annualized_return": 0.50,
        "max_drawdown": -0.25,
        "complete_sessions": 1_000,
        "positive_years": 4,
        "execution_rate": 0.90,
        "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
    account["option_account_eligible"] = False
    assert study.evaluate_gate(account)["option_account_eligible"] is False
