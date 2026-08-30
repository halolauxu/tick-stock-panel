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
        / "run_p0_convertible_bond_momentum_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_cb_momentum", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_momentum_uses_exactly_prior_twenty_closes_and_excludes_eb() -> None:
    days = [date(2019, 1, 1) + timedelta(days=index) for index in range(21)]
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH"] * 21 + ["132001.SH"] * 21,
            "date": days + days,
            "open": [100.0] * 42,
            "close": ([100.0] * 20 + [110.0]) * 2,
            "volume": [100_000.0] * 42,
            "amount": [50_000_000.0] * 42,
            "cb_value": [100.0] * 42,
            "cb_over_rate": [20.0] * 42,
        }
    )
    master = pl.DataFrame(
        {
            "symbol": ["110001.SH", "132001.SH"],
            "list_date": [date(2018, 1, 1)] * 2,
            "maturity_date": [date(2025, 1, 1)] * 2,
        }
    )

    result = study.prepare_panel(daily, master)

    assert result["symbol"].unique().to_list() == ["110001.SH"]
    assert result["momentum_20d"][-1] == pytest.approx(0.10)


def test_candidates_rank_positive_momentum_first() -> None:
    signal_date = date(2020, 1, 3)
    entry_date = date(2020, 1, 6)
    panel = pl.DataFrame(
        {
            "date": [signal_date] * 3,
            "symbol": ["LOW", "HIGH", "NEG"],
            "listing_days": [100] * 3,
            "maturity_days": [500] * 3,
            "mean_amount_20d": [50_000_000.0] * 3,
            "close": [100.0] * 3,
            "cb_over_rate": [20.0] * 3,
            "cb_value": [100.0] * 3,
            "momentum_20d": [0.05, 0.10, -0.10],
            "volume": [100_000.0] * 3,
            "amount": [50_000_000.0] * 3,
        }
    )
    schedule = pl.DataFrame(
        {"signal_date": [signal_date], "entry_date": [entry_date]}
    )

    candidates = study.build_candidates(panel, schedule)

    assert candidates["symbol"].to_list() == ["HIGH", "LOW"]


def test_gate_rejects_unresolved_positions_in_any_capacity_account() -> None:
    def account(unresolved: int) -> dict:
        return {
            "metrics": {
                "annualized": 0.60,
                "max_drawdown": -0.20,
                "positive_years": 3,
            },
            "execution": {
                "buy": {"execution_rate": 0.95},
                "sell": {"execution_rate": 0.95},
            },
            "integrity": {
                "ending_unresolved_positions": unresolved,
                "max_cash_reconciliation_error": 0.0,
            },
        }

    accounts = {name: account(0) for name in study.ACCOUNT_SIZES}
    benchmark = {"annualized": 0.20}
    assert study.evaluate_gate(accounts, benchmark)["passed"] is True

    accounts["cny_1m"] = account(1)
    assert study.evaluate_gate(accounts, benchmark)["passed"] is False
