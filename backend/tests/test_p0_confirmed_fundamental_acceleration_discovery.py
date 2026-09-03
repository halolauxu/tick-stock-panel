from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_confirmed_fundamental_acceleration_discovery as study  # noqa: E402


def test_reaction_panel_requires_adjacent_market_days() -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "600001.SH", "600001.SH"],
            "date": [
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 2),
                date(2020, 1, 6),
            ],
            "close": [10.0, 10.4, 20.0, 20.4],
        }
    )

    result = study.build_reaction_panel(panel)

    assert result.height == 1
    assert result["symbol"][0] == "600000.SH"
    assert abs(result["reaction_return"][0] - 0.04) < 1e-12


def test_classification_uses_frozen_reaction_bands() -> None:
    reactions = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH", "600002.SH"],
            "ann_date": [date(2020, 4, 1)] * 3,
            "reaction_date": [date(2020, 4, 2)] * 3,
            "reaction_return": [0.03, -0.02, 0.08],
            "category": ["mother"] * 3,
        }
    )

    result = study.classify_reactions(reactions)
    categories = dict(result.select("symbol", "category").iter_rows())

    assert categories == {
        "600000.SH": study.CANDIDATE,
        "600001.SH": study.NEGATIVE_CONTROL,
        "600002.SH": study.OVERREACTION_CONTROL,
    }
    assert result["ann_date"].to_list() == [date(2020, 4, 2)] * 3


def test_evaluate_requires_both_directional_controls() -> None:
    candidate = {
        "universe_eligible_events": 800,
        "tradable_events": 780,
        "announcement_days": 300,
        "tradable_rate": 0.975,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 4,
        "mean_net_return": 0.018,
        "mean_excess_return": 0.015,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.30,
    }
    primary = {
        study.CANDIDATE: candidate,
        study.NEGATIVE_CONTROL: {"mean_excess_return": 0.005},
        study.OVERREACTION_CONTROL: {"mean_excess_return": 0.009},
    }

    decision = study.evaluate(primary)

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_ACCOUNT_CONTRACT"
