from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_t0_etf_overnight_intraday_reversal_development.py"
)
SPEC = importlib.util.spec_from_file_location("p0_t0_etf_overnight_reversal", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _minutes(days: list[date], opening_overrides: dict[tuple[date, str], float] | None = None) -> pl.DataFrame:
    opening_overrides = opening_overrides or {}
    rows = []
    for day_index, day in enumerate(days):
        for symbol_index, symbol in enumerate(study.SYMBOLS):
            base = 1.0 + symbol_index * 0.01 + day_index * 0.001
            stamps = [datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30) + timedelta(minutes=i) for i in range(121)]
            stamps += [datetime.combine(day, datetime.min.time()).replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
            for stamp in stamps:
                price = opening_overrides.get((day, symbol), base) if stamp.time().hour == 9 and stamp.time().minute == 30 else base
                rows.append(
                    {
                        "symbol": symbol,
                        "datetime": stamp,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 10_000_000.0,
                        "amount": 100_000.0,
                    }
                )
    return pl.DataFrame(rows)


def test_signal_uses_open_gap_and_selects_three_lowest() -> None:
    first = date(2025, 8, 1)
    second = date(2025, 8, 4)
    frame = _minutes([first, second])
    prior = {
        row["symbol"]: row["close"]
        for row in frame.filter(pl.col("datetime") == datetime(2025, 8, 1, 15, 0)).iter_rows(named=True)
    }
    losers = study.SYMBOLS[:3]
    overrides = {(second, symbol): prior[symbol] * (0.90 + index * 0.01) for index, symbol in enumerate(losers)}
    frame = _minutes([first, second], overrides)
    signals = study.build_signals(frame)
    assert [row["symbol"] for row in signals] == list(losers)
    assert [row["rank"] for row in signals] == [1, 2, 3]


def test_future_prices_do_not_change_signal_membership() -> None:
    first = date(2025, 8, 1)
    second = date(2025, 8, 4)
    frame = _minutes([first, second])
    baseline = [(row["symbol"], row["rank"]) for row in study.build_signals(frame)]
    changed = frame.with_columns(
        pl.when(
            (pl.col("datetime").dt.date() == second)
            & (pl.col("datetime").dt.time() >= datetime(2025, 8, 4, 9, 31).time())
        )
        .then(pl.col("close") * 3)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert [(row["symbol"], row["rank"]) for row in study.build_signals(changed)] == baseline


def test_account_closes_intraday_and_balances() -> None:
    first = date(2025, 8, 1)
    second = date(2025, 8, 4)
    frame = _minutes([first, second])
    signals = study.build_signals(frame)
    result = study.simulate_account(frame, signals, 200_000.0)
    assert result["planned_legs"] == 3
    assert result["intraday_trades"] == 3
    assert result["open_positions"] == 0
    assert result["ending_market_value"] == 0
    assert result["total_cost"] > 0
    assert abs(result["ledger_error"]) <= 0.01


def test_gate_requires_return_excess_and_trade_count() -> None:
    account = {
        "annualized_return": 0.51,
        "max_drawdown": -0.20,
        "intraday_trades": 200,
        "positive_months": 4,
        "entry_execution_rate": 0.95,
        "carry_days": 0,
        "open_positions": 0,
        "ending_market_value": 0,
        "ledger_error": 0.0,
    }
    diagnostics = {"mean_excess_return": 0.0011}
    assert all(study.evaluate_gate(account, diagnostics).values())
    diagnostics["mean_excess_return"] = 0.0009
    assert not study.evaluate_gate(account, diagnostics)["mean_excess_at_least_10bps"]
