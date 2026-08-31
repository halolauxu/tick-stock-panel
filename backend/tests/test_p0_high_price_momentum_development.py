from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "run_p0_high_price_momentum_development.py"
)
SPEC = importlib.util.spec_from_file_location("high_price_momentum", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_monthly_momentum_requires_twelve_consecutive_month_ends() -> None:
    dates = [date(2019, month, 20) for month in range(1, 13)] + [date(2020, 1, 20)]
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * 13,
            "date": dates,
            "close": [10.0 + index for index in range(13)],
        }
    )
    monthly, _ = study.build_monthly_signal_panel(panel)
    last = monthly.filter(pl.col("date") == date(2019, 12, 20)).to_dicts()[0]
    assert last["momentum_12m"] == 21.0 / 10.0 - 1.0


def test_independent_price_and_momentum_sorts_select_only_intersection() -> None:
    count = 100
    ranked = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * count,
            "entry_date": [date(2020, 2, 3)] * count,
            "symbol": [f"{index:06d}.SH" for index in range(count)],
            "raw_close": [float(index + 3) for index in range(count)],
            "close": [float(index + 3) for index in range(count)],
            "market_cap": [float(index + 1) * 1_000_000_000 for index in range(count)],
            "amount": [100_000_000.0] * count,
            "mean_amount_20d": [100_000_000.0] * count,
            "momentum_12m": [float(index) / 100 for index in range(count)],
        }
    )
    scored = study.rank_monthly_universe(ranked)
    candidates = study.build_candidates(scored, require_high_price=True)
    assert candidates.height == 9
    assert candidates.get_column("price_decile").unique().to_list() == [9]
    assert candidates.get_column("momentum_quintile").unique().to_list() == [4]
    assert candidates.get_column("market_cap_decile").min() > 0


def test_smallest_cap_decile_is_computed_before_price_and_liquidity_filters() -> None:
    count = 10
    monthly = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * count,
            "entry_date": [date(2020, 2, 3)] * count,
            "symbol": [f"{index:06d}.SH" for index in range(count)],
            "raw_close": [2.0] + [10.0 + index for index in range(1, count)],
            "close": [2.0] + [10.0 + index for index in range(1, count)],
            "market_cap": [float(index + 1) for index in range(count)],
            "amount": [100_000_000.0] * count,
            "mean_amount_20d": [100_000_000.0] * count,
            "momentum_12m": [float(index) / 10 for index in range(count)],
        }
    )
    ranked = study.rank_monthly_universe(monthly)
    assert "000000.SH" not in ranked.get_column("symbol").to_list()
    assert "000001.SH" in ranked.get_column("symbol").to_list()
    assert (
        ranked.filter(pl.col("symbol") == "000001.SH").get_column("market_cap_decile").item() == 1
    )


def test_gate_requires_increment_over_plain_momentum_and_closed_account() -> None:
    candidate = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_full_years": 6,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_open_positions": 1,
            "max_cash_reconciliation_error": 0.0,
        },
        "completed_trades": 400,
        "profit_concentration": {"largest_positive_symbol_share": 0.10},
    }
    control = {"metrics": {"annualized": 0.55}}
    benchmark = {"annualized": 0.10}
    decision = study.evaluate_gate(candidate, control, benchmark)
    assert decision["passed"] is False
    assert decision["checks"]["plain_momentum_increment_at_least_10pp"] is False
    assert decision["checks"]["no_ending_open_positions"] is False


def test_profit_concentration_uses_closed_cash_flows_by_symbol() -> None:
    result = study._profit_concentration(
        [
            {"symbol": "A", "cash_delta": -100.0},
            {"symbol": "A", "cash_delta": 130.0},
            {"symbol": "B", "cash_delta": -100.0},
            {"symbol": "B", "cash_delta": 110.0},
        ]
    )
    assert result["largest_positive_symbol"] == "A"
    assert result["largest_positive_symbol_share"] == 0.75
