from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_apl_beta_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_apl", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_compute_apl_betas_recovers_partial_regression_coefficient() -> None:
    rows = []
    start = date(2020, 1, 2)
    for offset in range(20):
        market = (offset - 10) / 100.0
        limit_fraction = ((offset * 7) % 11) / 100.0
        rows.append(
            {
                "symbol": "A.SZ",
                "date": start + timedelta(days=offset),
                "month": "2020-01",
                "daily_return": 0.01 + 2.0 * market - 3.0 * limit_fraction,
                "market_return": market,
                "limit_hit_fraction": limit_fraction,
            }
        )

    result = study.compute_apl_betas(pl.DataFrame(rows), minimum_days=15)

    assert result.to_dicts() == [
        {
            "symbol": "A.SZ",
            "month": "2020-01",
            "observations": 20,
            "apl_beta": pytest.approx(3.0),
        }
    ]


def test_daily_annualized_uses_252_trading_days() -> None:
    daily_return = 0.001

    result = study._daily_annualized([daily_return] * 252)

    assert result == pytest.approx((1.0 + daily_return) ** 252 - 1.0)


def test_include_initial_cash_return_reconciles_full_equity_path() -> None:
    daily = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3)],
            "equity": [110.0, 121.0],
            "daily_return": [None, 0.1],
        }
    )

    result = study._include_initial_cash_return(daily, 100.0)

    returns = result.get_column("daily_return").to_list()
    assert returns == pytest.approx([0.1, 0.1])
    assert study.baseline._compound(returns) == pytest.approx(0.21)


def test_build_candidates_uses_microcap_decile_then_lowest_apl_beta() -> None:
    signal_date, entry_date = date(2020, 1, 31), date(2020, 2, 3)
    observations = pl.DataFrame(
        {
            "signal_date": [signal_date] * 20,
            "entry_date": [entry_date] * 20,
            "next_rebalance_date": [date(2020, 3, 2)] * 20,
            "symbol": [f"{index:06d}.SZ" for index in range(20)],
            "apl_beta": [float(20 - index) for index in range(20)],
            "market_cap": [float(index + 1) for index in range(20)],
            "amount": [100_000_000.0] * 20,
        }
    )

    candidate = study.build_candidates(observations, strategy="low_apl", top_n=1)
    control = study.build_candidates(observations, strategy="microcap", top_n=1)

    assert candidate.get_column("symbol").to_list() == ["000001.SZ"]
    assert control.get_column("symbol").to_list() == ["000000.SZ"]


def _quote(
    day: date,
    symbol: str,
    *,
    raw_open: float = 10.0,
    limit_down: float = 9.0,
    excluded: bool = False,
) -> dict:
    return {
        "date": day,
        "symbol": symbol,
        "open": raw_open,
        "raw_open": raw_open,
        "close": raw_open,
        "raw_close": raw_open,
        "volume": 1_000_000.0,
        "amount": 10_000_000.0,
        "limit_up_price": 11.0,
        "limit_down_price": limit_down,
        "is_excluded_name": excluded,
    }


def test_monthly_account_retries_blocked_exit_daily_and_can_sell_st() -> None:
    d0, d1, d2, d3 = (
        date(2020, 1, 31),
        date(2020, 2, 3),
        date(2020, 3, 2),
        date(2020, 3, 3),
    )
    candidates = pl.DataFrame(
        {
            "signal_date": [d0],
            "entry_date": [d1],
            "symbol": ["A.SZ"],
            "cap_rank": [1],
            "signal_amount": [100_000_000.0],
        }
    )
    quotes = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d2, "A.SZ", raw_open=9.0, limit_down=9.0, excluded=True),
            _quote(d3, "A.SZ", raw_open=10.5, excluded=True),
        ]
    )

    result = study.simulate_monthly_account(
        candidates,
        quotes,
        [d1, d2, d3],
        rebalance_dates=[d1, d2],
        initial_cash=200_000.0,
        target_positions=10,
    )

    assert [trade["side"] for trade in result["trades"]] == ["BUY", "SELL"]
    assert result["trades"][1]["date"] == d3
    assert result["completed_positions"][0]["exit_delay_days"] == 1
    assert result["ending_positions"] == []
    assert result["max_cash_reconciliation_error"] == pytest.approx(0.0)


def test_monthly_account_never_backfills_after_terminal_exit_failure() -> None:
    d0, d1, d2, d3, d4 = (
        date(2020, 1, 31),
        date(2020, 2, 3),
        date(2020, 3, 2),
        date(2020, 3, 3),
        date(2020, 3, 4),
    )
    candidates = pl.DataFrame(
        {
            "signal_date": [d0],
            "entry_date": [d1],
            "symbol": ["A.SZ"],
            "cap_rank": [1],
            "signal_amount": [100_000_000.0],
        }
    )
    quotes = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d2, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d4, "A.SZ", raw_open=10.5),
        ]
    )

    result = study.simulate_monthly_account(
        candidates,
        quotes,
        [d1, d2, d3, d4],
        rebalance_dates=[d1, d2],
        initial_cash=200_000.0,
        target_positions=10,
        max_exit_delay=1,
    )

    assert [trade["side"] for trade in result["trades"]] == ["BUY"]
    sell_orders = [row for row in result["orders"] if row["side"] == "SELL"]
    assert [row["date"] for row in sell_orders] == [d2, d3]
    assert result["ending_positions"][0]["terminal_exit_failure"] == "limit_down"
