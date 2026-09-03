from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_compression_volume_breakout_discovery as study  # noqa: E402


def test_volume_ratio_uses_only_prior_ten_bars() -> None:
    rows = []
    start = date(2020, 1, 1)
    for offset in range(20):
        rows.append(
            {
                "symbol": "600000.SH",
                "date": start + timedelta(days=offset),
                "close": 10.0 + offset * 0.01,
                "high": 10.2 + offset * 0.01,
                "low": 9.8 + offset * 0.01,
                "volume": 200.0 if offset == 19 else 100.0,
            }
        )

    result = study.attach_features(pl.DataFrame(rows))

    assert result["vol_ratio_10d"][-1] == 2.0


def test_ranked_events_select_high_and_low_score_arms() -> None:
    rows = []
    signal_date = date(2020, 1, 3)
    for index in range(30):
        rows.append(
            {
                "symbol": f"600{index:03d}.SH",
                "date": signal_date,
                "raw_close": 10.0,
                "amount": 1e8,
                "boll_width": 0.01 + index * 0.01,
                "vol_ratio_10d": 3.0 - index * 0.05,
                "close_position": 1.0 - index * 0.02,
            }
        )

    result = study.build_ranked_events(pl.DataFrame(rows))

    assert result.filter(pl.col("category") == study.CANDIDATE).height > 0
    assert result.filter(pl.col("category") == study.LOW_SCORE_CONTROL).height > 0
    best = result.sort("composite_score", descending=True).row(0, named=True)
    assert best["symbol"] == "600000.SH"
    assert best["category"] == study.CANDIDATE


def test_evaluate_requires_directional_control_spread() -> None:
    candidate = {
        "universe_eligible_events": 5_100,
        "tradable_events": 5_050,
        "announcement_days": 350,
        "tradable_rate": 0.99,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 10,
        "mean_net_return": 0.012,
        "mean_excess_return": 0.015,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.30,
    }
    summaries = {
        study.CANDIDATE: candidate,
        study.LOW_SCORE_CONTROL: {"mean_excess_return": 0.004},
    }

    decision = study.evaluate(summaries)

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_ACCOUNT_CONTRACT"
