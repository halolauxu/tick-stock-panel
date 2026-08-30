from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_microcap_account.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_microcap_account", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


account = _load_module()


def _candidates(rows: list[tuple[date, date, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": signal_date,
                "entry_date": entry_date,
                "symbol": symbol,
                "cap_rank": rank,
                "signal_amount": 10_000_000.0,
            }
            for signal_date, entry_date, symbol, rank in rows
        ]
    )


def _quote(
    entry_date: date,
    symbol: str,
    *,
    raw_open: float = 10.0,
    adjusted_open: float | None = None,
    close: float | None = None,
    limit_up: float = 11.0,
    limit_down: float = 9.0,
    exact: bool = True,
    excluded_name: bool = False,
) -> dict:
    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "quote_date": entry_date,
        "open": adjusted_open or raw_open,
        "raw_open": raw_open,
        "close": close or adjusted_open or raw_open,
        "raw_close": close or raw_open,
        "entry_volume": 1_000_000.0,
        "entry_amount": 10_000_000.0,
        "limit_up_price": limit_up,
        "limit_down_price": limit_down,
        "exact_quote": exact,
        "is_excluded_name": excluded_name,
    }


def test_affordable_shares_respects_lot_and_minimum_commission() -> None:
    shares = account.affordable_shares(10.0, 15_000.0, 15_000.0)

    assert shares == 1400
    gross = shares * 10.0
    assert gross * 1.0005 + account.commission(gross) <= 15_000.0
    assert account.commission(gross) == 5.0


def test_simulator_backfills_blocked_buy_then_sells_before_next_buy() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d0, d1, "B.SZ", 2),
            (d2, d3, "C.SZ", 1),
            (d2, d3, "B.SZ", 2),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ", raw_open=11.0, limit_up=11.0),
            _quote(d1, "B.SZ"),
            _quote(d1, "C.SZ"),
            _quote(d3, "A.SZ"),
            _quote(d3, "B.SZ", raw_open=10.5, close=10.5),
            _quote(d3, "C.SZ", raw_open=5.0, close=5.0),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
    )

    assert [row["symbol"] for row in result["trades"]] == [
        "B.SZ",
        "B.SZ",
        "C.SZ",
    ]
    assert result["orders"][0]["reason"] == "limit_up"
    assert result["trades"][1]["side"] == "SELL"
    assert result["trades"][2]["side"] == "BUY"
    assert result["ending_positions"][0]["symbol"] == "C.SZ"
    assert result["max_cash_reconciliation_error"] == pytest.approx(0.0)


def test_blocked_exit_keeps_position_and_prevents_replacement() -> None:
    d0, d1 = date(2024, 1, 5), date(2024, 1, 8)
    d2, d3 = date(2024, 1, 12), date(2024, 1, 15)
    candidates = _candidates(
        [
            (d0, d1, "A.SZ", 1),
            (d2, d3, "B.SZ", 1),
        ]
    )
    execution = pl.DataFrame(
        [
            _quote(d1, "A.SZ"),
            _quote(d1, "B.SZ"),
            _quote(d3, "A.SZ", raw_open=9.0, limit_down=9.0),
            _quote(d3, "B.SZ"),
        ]
    )

    result = account.simulate_account(
        candidates,
        execution,
        initial_cash=30_000.0,
        target_positions=1,
    )

    assert result["ending_positions"][0]["symbol"] == "A.SZ"
    rejected_sell = [
        row
        for row in result["orders"]
        if row["side"] == "SELL" and row["status"] == "REJECTED"
    ]
    assert rejected_sell[0]["reason"] == "limit_down"
    assert not any(
        row["side"] == "BUY" and row["symbol"] == "B.SZ"
        for row in result["orders"]
    )


def test_daily_equity_uses_stale_mark_without_future_backfill() -> None:
    d1, d2, d3 = date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)
    simulation = {
        "intervals": [
            {
                "position_id": 1,
                "symbol": "A.SZ",
                "units": 100.0,
                "start_date": d1,
                "end_date": None,
            }
        ],
        "snapshots": [{"date": d1, "cash": 1_000.0}],
        "ending_positions": [{"symbol": "A.SZ"}],
    }
    quotes = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "date": [d1, d3],
            "close": [10.0, 12.0],
        }
    )

    daily, integrity = account.build_daily_equity(
        simulation,
        quotes,
        [d1, d2, d3],
        initial_cash=2_000.0,
    )

    assert daily["equity"].to_list() == [2_000.0, 2_000.0, 2_200.0]
    assert integrity["stale_position_days"] == 1
    assert integrity["longest_stale_trading_days"] == 1
    assert integrity["ending_unresolved_positions"] == 0


def test_main_board_st_uses_five_percent_limit_and_cannot_be_new_buy() -> None:
    days = [date(2024, 1, 8), date(2024, 1, 9)]
    source = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": days,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "raw_close": [10.0, 10.0],
            "volume": [1_000.0, 1_000.0],
            "amount": [1_000_000.0, 1_000_000.0],
            "name": ["ST样本", "ST样本"],
        }
    )

    quotes = account.prepare_quote_panel(source)

    assert quotes["limit_up_price"][1] == pytest.approx(10.5)
    assert quotes["limit_down_price"][1] == pytest.approx(9.5)
    candidate = {"signal_amount": 1_000_000.0}
    quote = _quote(days[1], "000001.SZ", excluded_name=True)
    assert account._buy_rejection(candidate, quote, 10_000.0) == (
        "became_risk_warning"
    )
    position = {"units": 1_000.0}
    assert account._sell_rejection(position, quote) is None
