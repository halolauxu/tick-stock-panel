from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_northbound_accumulation_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_northbound_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _panel() -> pl.DataFrame:
    rows = []
    dates = [
        date(2020, 1, 3),
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 10),
        date(2020, 1, 13),
        date(2020, 1, 14),
    ]
    for symbol, shares, second_holding in (
        ("600000.SH", 1_000_000.0, 20_000.0),
        ("000001.SZ", 2_000_000.0, 30_000.0),
    ):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "_global_index": index,
                    "total_shares": shares,
                    "raw_close": 10.0,
                    "mean_amount_20d": 100_000_000.0,
                    "second_holding": second_holding,
                }
            )
    return pl.DataFrame(rows)


def test_build_candidates_uses_adjacent_snapshot_and_two_day_lag() -> None:
    panel = _panel()
    holdings = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 3),
                date(2020, 1, 3),
                date(2020, 1, 10),
                date(2020, 1, 10),
            ],
            "symbol": ["600000.SH", "000001.SZ"] * 2,
            "holding_shares": [10_000.0, 20_000.0, 20_000.0, 30_000.0],
        }
    )

    candidates, action_dates, audit = study.build_candidates(holdings, panel)

    assert candidates.height == 2
    assert candidates["entry_date"].unique().to_list() == [date(2020, 1, 14)]
    assert action_dates == [date(2020, 1, 7), date(2020, 1, 14)]
    assert audit["active_signal_weeks"] == 1
    deltas = dict(candidates.select("symbol", "holding_ratio_delta_pct").iter_rows())
    assert round(deltas["600000.SH"], 8) == 1.0
    assert round(deltas["000001.SZ"], 8) == 0.5


def test_build_candidates_rejects_non_adjacent_stock_snapshot() -> None:
    panel = _panel()
    holdings = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 3),
                date(2020, 1, 10),
                date(2020, 1, 10),
            ],
            "symbol": ["600000.SH", "600000.SH", "000001.SZ"],
            "holding_shares": [10_000.0, 20_000.0, 30_000.0],
        }
    )

    candidates, _, _ = study.build_candidates(holdings, panel)

    assert candidates.get_column("symbol").to_list() == ["600000.SH"]


def test_gate_requires_strict_return_execution_and_integrity() -> None:
    account_result = {
        "metrics": {
            "annualized": 0.60,
            "max_drawdown": -0.20,
            "positive_full_years": 2,
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

    passed = study.evaluate_gate(account_result, {"annualized": 0.20})
    failed = study.evaluate_gate(account_result, {"annualized": 0.45})

    assert passed["verdict"] == "PROMOTE_TO_VALIDATION"
    assert failed["verdict"] == "TERMINATE"
    assert failed["checks"]["annualized_excess_at_least_20pp"] is False
