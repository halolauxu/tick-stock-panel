from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_50etf_ivs_direction_screen.py"
)
SPEC = importlib.util.spec_from_file_location("p0_50etf_ivs_direction_screen", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _business_dates(start: date, count: int) -> list[date]:
    output: list[date] = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return output


def test_implied_volatility_recovers_call_and_put_inputs() -> None:
    for side in ("C", "P"):
        price = study.bs_price(
            spot=2.5,
            strike=2.5,
            time_years=30 / 365.25,
            rate=0.02,
            volatility=0.25,
            call_put=side,
        )
        recovered = study.implied_volatility(
            price=price,
            spot=2.5,
            strike=2.5,
            time_years=30 / 365.25,
            rate=0.02,
            call_put=side,
        )
        assert recovered == pytest.approx(0.25, abs=1e-8)


def test_daily_ivs_pairs_identical_terms_and_uses_open_interest_weights() -> None:
    dates = _business_dates(date(2024, 1, 2), 12)
    trade_date = dates[0]
    maturity = dates[-1]
    master = pl.DataFrame(
        [
            {
                "contract": side,
                "call_put": side,
                "exercise_price": 2.5,
                "opt_multiplier": 10_000.0,
                "maturity_date": maturity,
                "list_date": date(2023, 12, 1),
                "delist_date": maturity,
            }
            for side in ("C", "P")
        ]
    )
    time_years = (maturity - trade_date).days / 365.25
    options = pl.DataFrame(
        [
            {
                "contract": side,
                "date": trade_date,
                "close": study.bs_price(
                    spot=2.5,
                    strike=2.5,
                    time_years=time_years,
                    rate=0.02,
                    volatility=volatility,
                    call_put=side,
                ),
                "open_interest": 1_000.0,
            }
            for side, volatility in (("C", 0.30), ("P", 0.20))
        ]
    )
    fund = pl.DataFrame(
        {
            "date": dates,
            "open": [2.5] * len(dates),
            "close": [2.5] * len(dates),
        }
    )
    shibor = pl.DataFrame(
        {
            "date": [trade_date],
            "on": [2.0],
            "1w": [2.0],
            "2w": [2.0],
            "1m": [2.0],
            "3m": [2.0],
            "6m": [2.0],
            "9m": [2.0],
            "1y": [2.0],
        }
    )
    daily, audit = study.build_daily_ivs(master, fund, options, shibor)
    assert daily.height == 1
    assert daily["ivs"][0] == pytest.approx(0.10, abs=1e-8)
    assert audit["valid_ivs_days"] == 1


def test_recursive_forecast_never_uses_unfinished_label() -> None:
    observations = []
    start = date(2016, 1, 1)
    for index in range(180):
        signal = start + timedelta(days=7 * index)
        observations.append(
            {
                "signal_date": signal,
                "target_date": signal + timedelta(days=7),
                "ivs": float(index % 7),
                "realized_log_return": 0.01 - 0.001 * (index % 7),
            }
        )
    forecasts = study.recursive_oos_forecasts(observations, min_training=100)
    assert forecasts
    assert all(
        row["training_last_target_date"] <= row["signal_date"] for row in forecasts
    )


def test_weekly_timing_respects_next_open_lots_costs_and_ledger() -> None:
    dates = _business_dates(date(2019, 1, 4), 7)
    forecasts = [
        {
            "signal_date": dates[0],
            "target_date": dates[-1],
            "forecast": 0.01,
            "history_mean_forecast": 0.01,
        }
    ]
    fund = pl.DataFrame(
        {
            "date": dates,
            "open": [10.0] * len(dates),
            "close": [10.1] * len(dates),
        }
    )
    result = study.simulate_weekly_timing(forecasts, fund, 200_000.0)
    record = result["records"][0]
    assert record["entry_date"] == dates[1]
    assert record["shares"] % 100 == 0
    assert record["fees"] > 0
    assert result["max_ledger_error"] <= 1e-8


def test_gate_requires_both_horizons_and_both_halves() -> None:
    data = {"valid_day_coverage": 0.99, "shibor_zero_rate_correlation": 0.99}
    weekly = {
        "oos_r2": 0.01,
        "negative_beta_share": 0.95,
        "low_minus_high_ivs_mean_return": 0.01,
        "half_correlations": {"2019_2021": 0.1, "2022_2024": -0.1},
    }
    monthly = {"oos_r2": 0.02}
    timing = {"annualized_return": 0.10, "max_ledger_error": 0.0}
    buy_hold = {"annualized_return": 0.10}
    result = study.evaluate_gate(data, weekly, monthly, timing, buy_hold)
    assert result["passed"] is False
    assert result["checks"]["both_oos_halves_positive_correlation"] is False
