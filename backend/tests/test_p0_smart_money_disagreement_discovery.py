from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_smart_money_disagreement_discovery.py"
    )
    spec = importlib.util.spec_from_file_location("p0_smart_money", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _row(day: date, symbol: str, large: float, small: float) -> dict:
    return {
        "trade_date": day,
        "symbol": symbol,
        "buy_sm_cny": max(small, 0.0),
        "sell_sm_cny": max(-small, 0.0),
        "buy_md_cny": 0.0,
        "sell_md_cny": 0.0,
        "buy_lg_cny": max(large, 0.0),
        "sell_lg_cny": max(-large, 0.0),
        "buy_elg_cny": 0.0,
        "sell_elg_cny": 0.0,
        "net_mf_cny": large + small,
    }


def test_weekly_events_use_last_actual_day_and_keep_frozen_directions() -> None:
    rows = [
        _row(date(2020, 1, 9), "600001.SH", 20.0, -10.0),
        _row(date(2020, 1, 10), "600001.SH", 30.0, -20.0),
        _row(date(2020, 1, 10), "000001.SZ", -30.0, 20.0),
        _row(date(2020, 1, 10), "300001.SZ", 100.0, -100.0),
    ]
    moneyflow = pl.DataFrame(rows)
    panel = pl.DataFrame(
        {
            "symbol": [row["symbol"] for row in rows],
            "trade_date": [row["trade_date"] for row in rows],
            "event_daily_amount": [100_000_000.0] * len(rows),
        }
    )

    events = study.build_weekly_events(moneyflow, panel)

    assert set(events.get_column("ann_date").to_list()) == {date(2020, 1, 10)}
    smart = events.filter(pl.col("category") == study.CANDIDATE)
    retail = events.filter(pl.col("category") == study.RETAIL_CONTROL)
    assert smart.get_column("symbol").to_list() == ["600001.SH"]
    assert retail.get_column("symbol").to_list() == ["000001.SZ"]
    assert "300001.SZ" not in events.get_column("symbol").to_list()


def test_weekly_ranking_is_deterministic_and_capped() -> None:
    day = date(2020, 1, 10)
    rows = [
        _row(day, f"600{i:03d}.SH", float(i + 1), -float(i + 1))
        for i in range(12)
    ]
    moneyflow = pl.DataFrame(rows)
    panel = pl.DataFrame(
        {
            "symbol": [row["symbol"] for row in rows],
            "trade_date": [day] * len(rows),
            "event_daily_amount": [100_000_000.0] * len(rows),
        }
    )

    events = study.build_weekly_events(moneyflow, panel)
    smart = events.filter(pl.col("category") == study.CANDIDATE)

    assert smart.height == study.TARGETS_PER_WEEK
    assert smart.sort("signal_rank").get_column("symbol").to_list()[0] == "600011.SH"


def test_gate_requires_effect_above_both_controls() -> None:
    def summary(excess: float) -> dict:
        return {
            "tradable_events": 2_000,
            "announcement_days": 300,
            "tradable_rate": 0.99,
            "unresolved_exits": 0,
            "mean_net_return": 0.006,
            "mean_excess_return": excess,
            "excess_daily_cluster_t": 3.0,
            "positive_excess_years": 6,
        }

    passing = {
        study.CANDIDATE: summary(0.004),
        study.LARGE_CONTROL: summary(0.0025),
        study.RETAIL_CONTROL: summary(-0.001),
    }
    failed = {
        **passing,
        study.LARGE_CONTROL: summary(0.0035),
    }

    assert study.evaluate(passing)["passed"] is True
    assert study.evaluate(failed)["passed"] is False
    assert (
        study.evaluate(failed)["checks"]["beats_large_inflow_control_by_10bp"]
        is False
    )
