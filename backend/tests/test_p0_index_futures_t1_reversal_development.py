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
        / "run_p0_index_futures_t1_reversal_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_index_futures_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_build_signals_uses_only_prior_turnover_window() -> None:
    start = date(2015, 1, 1)
    rows = []
    for offset in range(61):
        rows.append(
            {
                "instrument": "000300.SH",
                "date": start + timedelta(days=offset),
                "close": 99.0 if offset == 60 else 100.0,
                "pre_close": 100.0,
                "amount": 200.0 if offset == 60 else 100.0,
            }
        )

    result = study.build_signals(pl.DataFrame(rows))

    assert result.height == 1
    assert result["signal_date"][0] == start + timedelta(days=60)
    assert result["prior_amount_q75"][0] == 100.0
    assert result["prior_amount_median"][0] == 100.0


def test_build_signals_chooses_largest_same_day_score() -> None:
    start = date(2015, 1, 1)
    rows = []
    for instrument, final_close in (("000300.SH", 99.0), ("000905.SH", 97.0)):
        for offset in range(61):
            rows.append(
                {
                    "instrument": instrument,
                    "date": start + timedelta(days=offset),
                    "close": final_close if offset == 60 else 100.0,
                    "pre_close": 100.0,
                    "amount": 200.0 if offset == 60 else 100.0,
                }
            )

    result = study.build_signals(pl.DataFrame(rows))

    assert result.height == 1
    assert result["index_code"][0] == "000905.SH"
    assert result["future_series"][0] == "IC.CFX"


def test_simulate_account_applies_multiplier_cost_margin_and_ledger() -> None:
    events = pl.DataFrame(
        [
            {
                "signal_date": date(2020, 1, 2),
                "execution_date": date(2020, 1, 3),
                "future_series": "IF.CFX",
                "future_open": 3000.0,
                "future_high": 3040.0,
                "future_low": 2990.0,
                "future_close": 3030.0,
                "future_volume": 100_000.0,
                "contract": "IF2001.CFX",
                "contract_multiplier": 300.0,
            }
        ]
    )

    result = study.simulate_account(events, 200_000.0)

    assert result["trades"] == 1
    assert result["records"][0]["contracts"] == 1
    assert result["records"][0]["gross_pnl"] == 9000.0
    assert result["records"][0]["cost"] == 450.0
    assert result["ending_equity"] == 208_550.0
    assert result["ledger_error"] == pytest.approx(0.0)


def test_simulate_account_rejects_locked_or_stale_market() -> None:
    events = pl.DataFrame(
        [
            {
                "signal_date": date(2020, 1, 2),
                "execution_date": date(2020, 1, 3),
                "future_series": "IF.CFX",
                "future_open": 3000.0,
                "future_high": 3000.0,
                "future_low": 3000.0,
                "future_close": 3000.0,
                "future_volume": 100_000.0,
                "contract": "IF2001.CFX",
                "contract_multiplier": 300.0,
            }
        ]
    )

    result = study.simulate_account(events, 200_000.0)

    assert result["trades"] == 0
    assert result["records"][0]["reason"] == "NO_EXECUTABLE_INTRADAY_MARKET"


def test_gate_requires_every_frozen_account_condition() -> None:
    account = {
        "annualized_return": 0.50,
        "max_drawdown": -0.25,
        "trades": 60,
        "positive_years": 4,
        "execution_rate": 0.90,
        "ledger_error": 0.0,
    }

    assert all(study.evaluate_gate(account).values())
