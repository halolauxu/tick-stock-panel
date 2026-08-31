from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_etf_share_flow_reversal_development.py"
    )
    spec = importlib.util.spec_from_file_location(
        "p0_etf_share_flow_reversal", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_prepare_panel_lags_share_information_and_neutralizes_split() -> None:
    days = [date(2013, 1, 1) + timedelta(days=index) for index in range(30)]
    daily = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "date": days,
            "open": [1.0] * len(days),
            "high": [1.0] * len(days),
            "low": [1.0] * len(days),
            "close": [1.0] * len(days),
            "volume": [10_000.0] * len(days),
            "amount": [30_000_000.0] * len(days),
            "source": ["test"] * len(days),
        }
    )
    adjustments = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "trade_date": days,
            "adj_factor": [1.0] * 10 + [2.0] * 20,
        }
    )
    shares = pl.DataFrame(
        {
            "symbol": ["510001.SH"] * len(days),
            "date": days,
            "shares_10k": [100.0] * 10 + [200.0] * 20,
        }
    )
    master = pl.DataFrame(
        {
            "symbol": ["510001.SH"],
            "list_date": [date(2010, 1, 1)],
            "delist_date": [None],
        },
        schema_overrides={"delist_date": pl.Date},
    )

    result = study.prepare_panel(daily, adjustments, master, shares)

    assert result["split_adjusted_shares"][9] == 100.0
    assert result["split_adjusted_shares"][10] == 100.0
    assert result["share_flow_5d"][15] == 0.0


def test_candidates_require_bottom_decile_and_five_percent_redemption() -> None:
    symbols = [f"{510000 + index:06d}.SH" for index in range(20)]
    signal_date = date(2020, 1, 3)
    panel = pl.DataFrame(
        {
            "symbol": symbols,
            "date": [signal_date] * len(symbols),
            "listing_days": [500] * len(symbols),
            "mean_amount_20d": [30_000_000.0] * len(symbols),
            "share_flow_5d": [-0.20, -0.10] + [-0.01] * 18,
            "amount": [30_000_000.0] * len(symbols),
        }
    )
    schedule = pl.DataFrame(
        {
            "signal_date": [signal_date],
            "entry_date": [date(2020, 1, 6)],
        }
    )

    result = study.build_candidates(panel, schedule)

    assert result["symbol"].to_list() == symbols[:2]
    assert result["cap_rank"].to_list() == [1, 2]


def test_gate_requires_all_four_capital_levels() -> None:
    def result(annualized: float) -> dict:
        return {
            "metrics": {
                "annualized": annualized,
                "max_drawdown": -0.20,
                "positive_years": 6,
            },
            "execution": {
                "buy": {"execution_rate": 0.95},
                "sell": {"execution_rate": 0.95},
            },
            "integrity": {"max_cash_reconciliation_error": 0.0},
            "account": {"ending_positions": 0},
        }

    accounts = {
        "cny_200k": result(0.60),
        "cny_300k": result(0.60),
        "cny_500k": result(0.60),
        "cny_1000k": result(0.60),
    }

    assert study.evaluate_gate(accounts, {"annualized": 0.20}, 100)[
        "passed"
    ] is True
    accounts["cny_1000k"] = result(0.49)
    assert study.evaluate_gate(accounts, {"annualized": 0.20}, 100)[
        "passed"
    ] is False
