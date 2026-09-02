from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_main_board_forecast_drift_account.py"
    )
    spec = importlib.util.spec_from_file_location("p0_forecast_account", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_candidates_start_next_day_and_expire_after_ten_days() -> None:
    all_dates = [date(2020, 1, 2) + timedelta(days=index) for index in range(30)]
    events = pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "ann_date": [all_dates[0]],
            "p_change_min": [20.0],
            "p_change_max": [30.0],
            "net_profit_min": [1.0],
            "net_profit_max": [2.0],
        }
    )
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "date": [all_dates[0]],
            "raw_close": [10.0],
            "amount": [100_000_000.0],
            "market_cap": [10_000_000_000.0],
        }
    )

    candidates, audit = study.build_candidates(events, panel, all_dates)

    assert candidates.height == study.SIGNAL_LIFETIME_TRADING_DAYS
    assert candidates.get_column("entry_date")[0] == all_dates[1]
    assert candidates.get_column("entry_date")[-1] == all_dates[10]
    assert audit["eligible_unique_events"] == 1


def test_candidates_rank_surprise_then_symbol() -> None:
    all_dates = [date(2020, 1, 2) + timedelta(days=index) for index in range(30)]
    symbols = [f"600{index:03d}.SH" for index in range(12)]
    events = pl.DataFrame(
        {
            "symbol": symbols,
            "ann_date": [all_dates[0]] * 12,
            "p_change_min": [float(index) for index in range(12)],
            "p_change_max": [float(index + 1) for index in range(12)],
            "net_profit_min": [1.0] * 12,
            "net_profit_max": [2.0] * 12,
        }
    )
    panel = pl.DataFrame(
        {
            "symbol": symbols,
            "date": [all_dates[0]] * 12,
            "raw_close": [10.0] * 12,
            "amount": [100_000_000.0] * 12,
            "market_cap": [10_000_000_000.0] * 12,
        }
    )

    candidates, _ = study.build_candidates(events, panel, all_dates)
    first_day = candidates.filter(pl.col("entry_date") == all_dates[1]).sort("cap_rank")

    assert first_day.height == study.TARGET_POSITIONS
    assert first_day.get_column("symbol")[0] == "600011.SH"


def test_gate_keeps_promising_account_below_final_50pct_target() -> None:
    primary = {
        "metrics": {
            "annualized": 0.25,
            "max_drawdown": -0.20,
            "positive_years": 6,
            "mean_cash_ratio": 0.25,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }

    decision = study.evaluate(primary, {"annualized": 0.12})

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_VALIDATION"

