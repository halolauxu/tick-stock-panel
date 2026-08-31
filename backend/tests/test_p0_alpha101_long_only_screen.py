from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_alpha101_long_only_screen.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_alpha101_screen", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


screen = _load_module()


def test_select_candidates_is_causal_eligible_and_symbol_deterministic() -> None:
    dates = [date(2013, 12, 31), date(2014, 1, 2)]
    symbols = np.array(["C.SZ", "A.SZ", "B.SZ", "D.SZ"])
    values = np.array(
        [[99.0, 99.0, 99.0, 99.0], [2.0, 3.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    eligible = np.array(
        [[True, True, True, True], [True, True, True, False]],
        dtype=bool,
    )
    amount = np.full(values.shape, 100_000_000.0, dtype=np.float64)

    result = screen.select_candidates(
        1,
        values,
        eligible,
        amount,
        dates,
        symbols,
        development_start=date(2014, 1, 1),
        top_n=2,
    )

    assert result.select("signal_date", "symbol", "rank").to_dicts() == [
        {"signal_date": date(2014, 1, 2), "symbol": "A.SZ", "rank": 1},
        {"signal_date": date(2014, 1, 2), "symbol": "B.SZ", "rank": 2},
    ]


def test_attach_execution_dates_uses_next_two_market_days() -> None:
    dates = [
        date(2014, 1, 2),
        date(2014, 1, 3),
        date(2014, 1, 6),
    ]
    candidates = pl.DataFrame(
        {
            "alpha_id": [1],
            "signal_date": [dates[0]],
            "symbol": ["A.SZ"],
            "rank": [1],
            "alpha_value": [1.0],
            "signal_amount": [100_000_000.0],
        }
    )
    prices = np.array([[10.0], [10.0], [11.0]], dtype=np.float32)
    context = screen.formulas.Alpha101Context.from_arrays(
        open=prices,
        high=prices * 1.01,
        low=prices * 0.99,
        close=prices,
        volume=np.full_like(prices, 1_000_000.0),
        amount=np.full_like(prices, 1_000_000_000.0),
    )

    result = screen.attach_execution_dates_and_benchmark(
        candidates, context, np.ones(prices.shape, dtype=bool), dates
    )

    assert result.select(
        "entry_date", "planned_exit_date", "benchmark_return"
    ).to_dicts() == [
        {
            "entry_date": dates[1],
            "planned_exit_date": dates[2],
            "benchmark_return": pytest.approx(0.1),
        }
    ]


def _quote(
    day: date,
    symbol: str,
    *,
    raw_open: float = 10.0,
    limit_up: float = 11.0,
    limit_down: float = 9.0,
    volume: float = 1_000_000.0,
) -> dict:
    return {
        "date": day,
        "symbol": symbol,
        "open": raw_open,
        "raw_open": raw_open,
        "close": raw_open,
        "raw_close": raw_open,
        "volume": volume,
        "amount": raw_open * volume,
        "limit_up_price": limit_up,
        "limit_down_price": limit_down,
        "is_excluded_name": False,
    }


def test_daily_executor_uses_t1_and_delays_locked_exit_without_fake_fill() -> None:
    d0 = date(2014, 1, 2)
    d1 = date(2014, 1, 3)
    d2 = date(2014, 1, 6)
    d3 = date(2014, 1, 7)
    candidates = pl.DataFrame(
        {
            "alpha_id": [1],
            "signal_date": [d0],
            "entry_date": [d1],
            "planned_exit_date": [d2],
            "symbol": ["A.SZ"],
            "rank": [1],
            "alpha_value": [1.0],
            "signal_amount": [100_000_000.0],
            "benchmark_return": [0.01],
        }
    )
    quotes = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d2, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "A.SZ", raw_open=11.0),
        ]
    )

    result = screen.simulate_daily_account(
        candidates,
        quotes,
        [d1, d2, d3],
        initial_cash=200_000.0,
        target_positions=10,
        max_exit_delay=20,
    )

    assert [trade["side"] for trade in result["trades"]] == ["BUY", "SELL"]
    assert result["trades"][1]["date"] == d3
    assert result["completed_trades"][0]["exit_delay_days"] == 1
    assert result["completed_trades"][0]["net_return"] > 0.09
    assert any(order["reason"] == "limit_down" for order in result["orders"])
    assert result["unresolved_exits"] == 0
    assert result["max_cash_reconciliation_error"] == pytest.approx(0.0)


def test_daily_executor_marks_already_held_and_never_resets_exit_clock() -> None:
    d0, d1, d2, d3 = (
        date(2014, 1, 2),
        date(2014, 1, 3),
        date(2014, 1, 6),
        date(2014, 1, 7),
    )
    candidates = pl.DataFrame(
        {
            "alpha_id": [1, 1],
            "signal_date": [d0, d1],
            "entry_date": [d1, d2],
            "planned_exit_date": [d2, d3],
            "symbol": ["A.SZ", "A.SZ"],
            "rank": [1, 1],
            "alpha_value": [1.0, 2.0],
            "signal_amount": [100_000_000.0, 100_000_000.0],
            "benchmark_return": [0.0, 0.0],
        }
    )
    quotes = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d2, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "A.SZ", raw_open=11.0),
        ]
    )

    result = screen.simulate_daily_account(
        candidates, quotes, [d1, d2, d3], initial_cash=200_000.0
    )

    assert any(order["reason"] == "ALREADY_HELD" for order in result["orders"])
    assert result["completed_trades"][0]["planned_exit_date"] == d2
    assert result["completed_trades"][0]["exit_date"] == d3


def test_daily_executor_never_fills_after_frozen_exit_delay_limit() -> None:
    d0, d1, d2, d3, d4 = (
        date(2014, 1, 2),
        date(2014, 1, 3),
        date(2014, 1, 6),
        date(2014, 1, 7),
        date(2014, 1, 8),
    )
    candidates = pl.DataFrame(
        {
            "alpha_id": [1],
            "signal_date": [d0],
            "entry_date": [d1],
            "planned_exit_date": [d2],
            "symbol": ["A.SZ"],
            "rank": [1],
            "alpha_value": [1.0],
            "signal_amount": [100_000_000.0],
            "benchmark_return": [0.0],
        }
    )
    quotes = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d2, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d4, "A.SZ", raw_open=12.0),
        ]
    )

    result = screen.simulate_daily_account(
        candidates,
        quotes,
        [d1, d2, d3, d4],
        initial_cash=200_000.0,
        max_exit_delay=1,
    )

    assert [trade["side"] for trade in result["trades"]] == ["BUY"]
    assert result["completed_trades"] == []
    assert result["unresolved_exits"] == 1


def test_holm_bonferroni_adjusts_all_31_formulas() -> None:
    p_values = {alpha_id: 0.5 for alpha_id in screen.ALPHA_IDS}
    p_values[1] = 0.001
    p_values[2] = 0.002

    result = screen.holm_bonferroni(p_values, family_alpha=0.05)

    assert result[1]["rank"] == 1
    assert result[1]["threshold"] == pytest.approx(0.05 / 31)
    assert result[1]["rejected"] is True
    assert result[2]["rejected"] is False
    assert all(result[alpha_id]["rejected"] is False for alpha_id in screen.ALPHA_IDS[1:])
