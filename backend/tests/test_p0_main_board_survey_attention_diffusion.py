from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_main_board_survey_attention_diffusion as study  # noqa: E402


def test_classification_is_main_board_prior_only_and_has_separate_cooldowns() -> None:
    frame = pl.DataFrame(
        {
            "event_id": ["a", "b", "c", "d", "e", "f", "g"],
            "symbol": [
                "600000.SH",
                "600000.SH",
                "600000.SH",
                "600000.SH",
                "000001.SZ",
                "000001.SZ",
                "300001.SZ",
            ],
            "notice_date": [
                date(2013, 6, 1),
                date(2014, 1, 10),
                date(2014, 2, 1),
                date(2014, 4, 15),
                date(2013, 7, 1),
                date(2014, 1, 20),
                date(2014, 1, 20),
            ],
            "institution_count": [4, 10, 20, 30, 10, 12, 30],
        }
    )

    result = study.classify_attention_events(frame)
    classified = dict(result.select("event_id", "category").iter_rows())

    assert classified == {
        "b": study.CANDIDATE,
        "d": study.CANDIDATE,
        "f": study.CONTROL,
    }
    assert "g" not in classified


def test_evaluate_requires_absolute_relative_and_control_edge() -> None:
    candidate = {
        "tradable_events": 1_000,
        "announcement_days": 500,
        "tradable_rate": 0.90,
        "benchmark_coverage": 0.99,
        "entry_capacity_feasible_rate": 0.95,
        "unresolved_exits": 10,
        "mean_net_return": 0.015,
        "mean_excess_return": 0.010,
        "excess_daily_cluster_t": 3.0,
        "positive_excess_years": 5,
        "max_year_positive_excess_share": 0.40,
    }
    results = {
        study.CANDIDATE: candidate,
        study.CONTROL: {"mean_excess_return": 0.0075},
    }

    assert study.evaluate(results)["passed"] is True
    results[study.CONTROL]["mean_excess_return"] = 0.0076
    assert study.evaluate(results)["passed"] is False


def test_primary_holding_period_is_ten_days() -> None:
    assert study.HOLD_TRADING_DAYS == 10
    assert study.MAX_EXIT_DELAY == 20
