from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "run_p0_emotion_limit_up_study.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_emotion_limit_up", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p0 = _load_module()


def test_state_thresholds_use_development_period_only() -> None:
    daily = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2021, 1, 4)],
            "limit_up_rate": [0.01, 0.02, 1.0],
            "ge2_rate": [0.002, 0.004, 1.0],
            "promotion_rate": [0.1, 0.2, 1.0],
            "winner_return": [-0.01, 0.01, 1.0],
            "broken_rate": [0.2, 0.4, 1.0],
            "up_ratio": [0.3, 0.5, 1.0],
        }
    )

    thresholds = p0.fit_state_thresholds(daily)

    assert thresholds["breadth_q90"] < 0.02
    assert thresholds["ge2_q90"] < 0.004
    assert thresholds["winner_q60"] < 0.01


def _synthetic_panel(*, entry_open_limit: bool, exit_open_limit_down: bool) -> pl.DataFrame:
    dates = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    raw_opens = [10.0, 11.0 if entry_open_limit else 10.0, 9.0, 11.0]
    limit_ups = [11.0, 11.0, 11.0, 12.0]
    limit_downs = [9.0, 9.0, 9.0, 9.0]
    return pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 4,
            "date": dates,
            "_global_index": [0, 1, 2, 3],
            "open": raw_opens,
            "raw_open": raw_opens,
            "close": [10.0, 10.0, 9.0, 11.0],
            "volume": [100.0] * 4,
            "amount": [20_000_000.0] * 4,
            "limit_up_price": limit_ups,
            "limit_down_price": limit_downs,
            "open_limit_down": [False, False, exit_open_limit_down, False],
        }
    )


def _synthetic_event(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.head(1).with_columns(
        pl.lit("示例").alias("name"),
        pl.col("date").alias("signal_date"),
        pl.lit("known_stress").alias("period"),
        pl.lit("ferment").alias("state"),
        pl.lit("first_board").alias("event_type"),
        pl.col("amount").alias("signal_day_amount"),
    )


def test_open_limit_up_is_not_manufactured_as_entry() -> None:
    panel = _synthetic_panel(entry_open_limit=True, exit_open_limit_down=False)
    outcomes = p0.attach_event_outcomes(_synthetic_event(panel), panel)

    assert outcomes["entry_status"].item() == "open_limit_up"
    assert outcomes["tradable_h1"].item() is False
    assert outcomes["net_return_h1"].item() is None


def test_t_plus_one_exit_defers_open_limit_down() -> None:
    panel = _synthetic_panel(entry_open_limit=False, exit_open_limit_down=True)
    outcomes = p0.attach_event_outcomes(_synthetic_event(panel), panel)

    assert outcomes["entry_date"].item() == date(2024, 1, 3)
    assert outcomes["exit_date_h1"].item() == date(2024, 1, 5)
    assert outcomes["exit_offset_h1"].item() == 3
    assert outcomes["tradable_h1"].item() is True
    expected = 11.0 * (1 - 0.0002 - 0.0005 - 0.0005) / (10.0 * (1 + 0.0002 + 0.0005)) - 1
    assert outcomes["net_return_h1"].item() == pytest.approx(expected)


def test_summary_clusters_events_by_signal_day() -> None:
    long = pl.DataFrame(
        {
            "period": ["validation"] * 3,
            "state": ["ferment"] * 3,
            "event_type": ["first_board"] * 3,
            "hold_days": [1] * 3,
            "signal_date": [date(2022, 1, 4), date(2022, 1, 4), date(2022, 1, 5)],
            "tradable": [True] * 3,
            "net_return": [0.01, 0.03, 0.04],
            "entry_gap": [0.0] * 3,
            "entry_status": ["ok"] * 3,
        }
    )

    summary = p0.summarize_outcomes(long)

    assert summary["event_mean_net"].item() == pytest.approx(0.08 / 3)
    assert summary["cluster_day_count"].item() == 2
    assert summary["cluster_mean_net"].item() == pytest.approx(0.03)


def test_concentration_audit_detects_single_year_dependency() -> None:
    long = pl.DataFrame(
        {
            "period": ["validation"] * 4,
            "state": ["ferment"] * 4,
            "event_type": ["first_board"] * 4,
            "hold_days": [1] * 4,
            "signal_date": [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2022, 1, 4),
                date(2023, 1, 4),
            ],
            "tradable": [True] * 4,
            "net_return": [0.10, 0.10, 0.01, -0.01],
            "industry": ["电子", "医药", "电子", "医药"],
        }
    )

    audit = p0.concentration_audits(long).row(0, named=True)

    assert audit["profitable_year_count"] == 2
    assert audit["top_year_positive_share"] > 0.9
    assert audit["industry_coverage"] == 1.0


def test_small_study_runs_end_to_end(tmp_path: Path) -> None:
    blocks = [
        [date(2020, 1, day) for day in (2, 3, 6, 7)],
        [date(2022, 1, day) for day in (4, 5, 6, 7)],
        [date(2024, 1, day) for day in (2, 3, 4, 5)],
    ]
    dates = [day for block in blocks for day in block]
    block_prices = [10.0, 11.0, 11.2, 11.3]
    prices = block_prices * len(blocks)
    highs = [10.0, 11.0, 11.3, 11.4] * len(blocks)
    frame = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * len(dates),
            "date": dates,
            "open": prices,
            "high": highs,
            "low": [value - 0.1 for value in prices],
            "close": prices,
            "volume": [100_000.0] * len(dates),
            "amount": [20_000_000.0] * len(dates),
            "raw_close": prices,
            "raw_high": highs,
            "raw_low": [value - 0.1 for value in prices],
        }
    )
    for day in dates:
        path = tmp_path / "kline_daily_enriched" / f"date={day.isoformat()}"
        path.mkdir(parents=True)
        frame.filter(pl.col("date") == day).write_parquet(path / "part.parquet")
    research = tmp_path / "research"
    research.mkdir()
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "list_date": [date(1999, 1, 1)],
            "delist_date": [None],
        }
    ).write_parquet(research / "historical_stock_universe.parquet")
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "name": ["浦发银行"],
            "start_date": [date(1999, 1, 1)],
            "end_date": [None],
        }
    ).write_parquet(research / "historical_stock_names.parquet")
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "l1_name": ["银行"],
            "in_date": [date(1999, 1, 1)],
            "out_date": [None],
        }
    ).write_parquet(research / "sw_l1_membership.parquet")
    output = research / "result.json"

    payload = p0.run(tmp_path, output)

    assert output.is_file()
    assert payload["data"]["first_date"] == date(2020, 1, 2)
    assert payload["data"]["last_date"] == date(2024, 1, 5)
    assert payload["data"]["event_rows"] == 3
    assert payload["decision"]["verdict"] == "DOWNGRADE"
