from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research import run_p0_ipo_first_open_wshape_development as study  # noqa: E402


def test_missing_pit_name_is_unknown_not_a_fabricated_risk_warning(
    tmp_path: Path,
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "list_date": [date(2014, 6, 16)],
            "delist_date": pl.Series([None], dtype=pl.Date),
        }
    ).write_parquet(research / "historical_stock_universe_all_a.parquet")
    pl.DataFrame(
        schema={
            "symbol": pl.String,
            "name": pl.String,
            "start_date": pl.Date,
            "end_date": pl.Date,
        }
    ).write_parquet(research / "historical_stock_names_all_a.parquet")
    source = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [date(2014, 6, 16)],
            "open": [10.0],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1_000.0],
            "raw_close": [10.0],
            "consecutive_limit_ups": pl.Series([0], dtype=pl.UInt32),
        }
    )
    row = study.attach_point_in_time_security(source, tmp_path).row(0, named=True)
    assert row["name_status_known"] is False
    assert row["excluded_name"] is False


def _panel_rows(
    symbol: str,
    *,
    first_open_return: float,
    first_open_at_limit: bool = False,
    start: date = date(2014, 6, 16),
) -> list[dict]:
    closes = [10.0, 11.0, 12.1, 13.31]
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "symbol": symbol,
                "date": start + timedelta(days=index),
                "open": close,
                "close": close,
                "raw_close": close,
                "volume": 1_000_000.0,
                "amount": 20_000_000.0,
                "consecutive_limit_ups": index,
                "list_date": start,
                "excluded_name": False,
                "name_status_known": True,
            }
        )
    prior = closes[-1]
    limit_up = math.floor((prior * 1.10 + 0.005) * 100) / 100
    raw_open = limit_up if first_open_at_limit else limit_up - 0.01
    rows.append(
        {
            "symbol": symbol,
            "date": start + timedelta(days=4),
            "open": raw_open,
            "close": raw_open * (1.0 + first_open_return),
            "raw_close": raw_open * (1.0 + first_open_return),
            "volume": 1_000_000.0,
            "amount": 20_000_000.0,
            "consecutive_limit_ups": 0,
            "list_date": start,
            "excluded_name": False,
            "name_status_known": True,
        }
    )
    return rows


def test_first_open_signal_is_causal_unique_and_excludes_open_at_limit() -> None:
    raw = pl.DataFrame(
        _panel_rows("000001.SZ", first_open_return=0.06)
        + _panel_rows("600001.SH", first_open_return=-0.08, first_open_at_limit=True),
        infer_schema_length=None,
    )
    panel = study.prepare_panel(raw)
    events = study.build_first_open_events(panel)
    assert events["symbol"].to_list() == ["000001.SZ"]
    row = events.row(0, named=True)
    assert row["prior_consecutive_limit_ups"] == 3
    assert row["entry_index"] == row["trade_index"] + 1
    assert row["planned_exit_index"] == row["trade_index"] + 6


def test_ranked_arms_and_controls_differ_only_by_threshold() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signal_date": [date(2015, 1, 1)] * 3,
            "entry_index": [10, 10, 10],
            "intraday_return": [0.08, 0.01, -0.07],
        }
    )
    strength = study.rank_candidates(events, "OPEN_STRENGTH", control=False)
    strength_control = study.rank_candidates(events, "OPEN_STRENGTH", control=True)
    selloff = study.rank_candidates(events, "OPEN_SELLOFF", control=False)
    selloff_control = study.rank_candidates(events, "OPEN_SELLOFF", control=True)
    assert strength["symbol"].to_list() == ["A"]
    assert strength_control["symbol"].to_list() == ["A", "B", "C"]
    assert selloff["symbol"].to_list() == ["C"]
    assert selloff_control["symbol"].to_list() == ["C", "B", "A"]


def test_development_builder_does_not_admit_validation_signals() -> None:
    rows = _panel_rows("000001.SZ", first_open_return=0.06)
    validation_rows = _panel_rows("600001.SH", first_open_return=-0.08)
    rows.extend(
        {
            **row,
            "date": row["date"].replace(year=2017),
            "list_date": row["list_date"].replace(year=2017),
        }
        for row in validation_rows
    )
    events = study.build_first_open_events(
        study.prepare_panel(pl.DataFrame(rows, infer_schema_length=None))
    )
    assert events["symbol"].to_list() == ["000001.SZ"]


def test_december_2016_ipo_is_not_dropped_from_development() -> None:
    raw = pl.DataFrame(
        _panel_rows(
            "600001.SH",
            first_open_return=0.06,
            start=date(2016, 12, 19),
        ),
        infer_schema_length=None,
    )
    events = study.build_first_open_events(study.prepare_panel(raw))
    assert events["symbol"].to_list() == ["600001.SH"]


def _quote(
    symbol: str,
    day: date,
    index: int,
    price: float,
    *,
    volume: float = 1_000_000.0,
    limit_down: float = 0.0,
) -> dict:
    return {
        "symbol": symbol,
        "date": day,
        "trade_index": index,
        "open": price,
        "raw_open": price,
        "close": price,
        "entry_volume": volume,
        "entry_amount": 100_000_000.0,
        "limit_up_price": price * 2,
        "limit_down_price": limit_down,
        "is_excluded_name": False,
        "exact_quote": True,
    }


def test_account_enters_t_plus_1_and_exits_after_five_intervals() -> None:
    days = [date(2015, 1, 1) + timedelta(days=i) for i in range(8)]
    ranked = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "signal_date": [days[0]],
            "entry_index": [1],
            "intraday_return": [0.08],
            "rank": [0],
        }
    )
    lookup = {("000001.SZ", i): _quote("000001.SZ", day, i, 10.0 + i) for i, day in enumerate(days)}
    summary, orders = study.simulate_account(
        ranked,
        lookup,
        days,
        initial_capital=200_000.0,
        start_index=0,
        end_index=7,
    )
    fills = orders.filter(pl.col("status") == "FILLED").sort("trade_index")
    assert fills["side"].to_list() == ["BUY", "SELL"]
    assert fills["trade_index"].to_list() == [1, 6]
    assert summary["completed_sells"] == 1
    assert summary["sell_execution_rate"] == 1.0
    assert summary["maximum_cash_ledger_error"] <= 0.01


def test_sell_delay_becomes_unresolved_without_made_up_fill() -> None:
    days = [date(2015, 1, 1) + timedelta(days=i) for i in range(12)]
    ranked = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "signal_date": [days[0]],
            "entry_index": [1],
            "intraday_return": [-0.08],
            "rank": [0],
        }
    )
    lookup = {
        ("000001.SZ", i): _quote("000001.SZ", day, i, 10.0) for i, day in enumerate(days) if i < 6
    }
    summary, orders = study.simulate_account(
        ranked,
        lookup,
        days,
        initial_capital=200_000.0,
        start_index=0,
        end_index=11,
    )
    assert summary["completed_sells"] == 0
    assert summary["unresolved_exits"] == 1
    assert summary["sell_execution_rate"] == 0.0
    assert orders.filter(pl.col("status") == "UNRESOLVED").height == 1


def test_gate_requires_returns_execution_trade_quality_and_integrity() -> None:
    candidate = {
        "annualized_return": 0.70,
        "max_drawdown": -0.20,
        "positive_years": 2,
        "completed_sells": 50,
        "buy_execution_rate": 0.95,
        "sell_execution_rate": 1.0,
        "unresolved_exits": 0,
        "mean_net_trade_return": 0.03,
        "signal_day_cluster_t": 2.5,
        "largest_positive_year_share": 0.55,
        "maximum_cash_ledger_error": 0.0,
    }
    control = {"annualized_return": 0.40}
    benchmark = {"annualized_return": 0.20}
    assert study.evaluate_gate(candidate, control, benchmark)["passed"]
    candidate["mean_net_trade_return"] = 0.01
    decision = study.evaluate_gate(candidate, control, benchmark)
    assert not decision["passed"]
    assert "mean_net_trade_return_at_least_2pct" in decision["failed_checks"]


def test_implementation_commit_can_be_frozen(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_GIT_COMMIT", "abc123")
    assert study.implementation_git_commit() == "abc123"
