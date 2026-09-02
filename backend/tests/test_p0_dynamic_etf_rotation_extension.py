from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_dynamic_etf_rotation_extension.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_etf_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_monthly_schedule_uses_only_completed_month_end() -> None:
    days = [date(2026, 1, 29), date(2026, 1, 30), date(2026, 2, 2)]
    panel = pl.DataFrame({"date": days, "symbol": ["510001.SH"] * len(days)})

    result = study.monthly_schedule(panel)

    assert result.row(0, named=True) == {
        "month": "2026-01",
        "signal_date": date(2026, 1, 30),
        "entry_date": date(2026, 2, 2),
    }


def test_overlay_gate_requires_target_every_year_and_drawdown() -> None:
    yearly = [{"year": year, "return": 0.10, "weeks": 52} for year in range(2014, 2026)] + [
        {"year": 2026, "return": 0.31, "weeks": 34}
    ]
    control = {"metrics": {"max_drawdown": -0.45}}
    variants = {"candidate": {"metrics": {"max_drawdown": -0.40, "yearly": yearly}}}

    assert study.evaluate_overlay_gate(control, variants)["passed"] is True
    variants["candidate"]["metrics"]["yearly"][3]["return"] = 0.0
    assert study.evaluate_overlay_gate(control, variants)["passed"] is False


def test_monthly_gate_needs_both_account_sizes() -> None:
    def account(return_2026: float) -> dict:
        return {
            "metrics": {
                "yearly": [{"year": 2026, "account_return": return_2026}],
                "max_drawdown": -0.20,
            },
            "integrity": {"max_cash_reconciliation_error": 0.0},
        }

    accounts = {"cny_200000": account(0.30), "cny_1000000": account(0.26)}
    assert study.evaluate_monthly_gate(accounts)["passed"] is True
    accounts["cny_1000000"] = account(0.24)
    assert study.evaluate_monthly_gate(accounts)["passed"] is False


def test_weekly_panel_preserves_listing_age() -> None:
    day = date(2026, 1, 2)
    panel = pl.DataFrame(
        {
            "symbol": ["510001.SH"],
            "date": [day],
            "open": [1.0],
            "close": [1.1],
            "volume": [1_000.0],
            "amount": [100_000_000.0],
            "momentum_120d": [0.2],
            "mean_amount_20d": [90_000_000.0],
            "listing_days": [(day - (day - timedelta(days=200))).days],
        }
    )

    result = study.weekly_etf_panel(panel)

    assert result.row(0, named=True)["listing_days"] == 200
    assert result.row(0, named=True)["adjusted_open"] == 1.0


def test_weekly_rank_does_not_use_future_exit_to_replace_the_winner() -> None:
    signal = date(2026, 1, 2)
    entry = date(2026, 1, 5)
    exit_date = date(2026, 1, 12)
    rows = []
    for symbol, momentum in (("510001.SH", 0.30), ("510002.SH", 0.20)):
        for day in (signal, entry, exit_date):
            if symbol == "510001.SH" and day == exit_date:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "adjusted_open": 1.0,
                    "adjusted_close": 1.0,
                    "volume": 1_000.0,
                    "amount": 100_000_000.0,
                    "momentum_120d": momentum,
                    "mean_amount_20d": 100_000_000.0,
                    "listing_days": 200,
                }
            )
    panel = pl.DataFrame(rows)
    schedule = pl.DataFrame({"date": [signal], "entry_date": [entry], "exit_date": [exit_date]})

    result = study.build_weekly_best_etf(panel, schedule)

    assert result.height == 1
    assert result.row(0, named=True)["symbol"] == "510001.SH"
    assert result.row(0, named=True)["entry_executable"] is True
    assert result.row(0, named=True)["exit_executable"] is False
    assert result.row(0, named=True)["etf_return"] == -1.0
