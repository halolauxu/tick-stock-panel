from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_industry_confirmed_forecast_drift_discovery as study  # noqa: E402


def test_classification_uses_leave_one_out_industry_breadth() -> None:
    rows: list[dict[str, object]] = []
    day = date(2020, 4, 15)
    for index in range(8):
        rows.append(
            {
                "symbol": f"600{index:03d}.SH",
                "ann_date": day,
                "l1_code": "A",
                "category": "growth_0_50" if index == 0 else (
                    "growth_ge_100" if index < 7 else "negative_control"
                ),
            }
        )
    for index in range(10):
        rows.append(
            {
                "symbol": f"000{index:03d}.SZ",
                "ann_date": day,
                "l1_code": "B",
                "category": "growth_50_100" if index == 0 else "negative_control",
            }
        )

    classified = study.classify_events(pl.DataFrame(rows))
    categories = dict(
        classified.select("symbol", "category").iter_rows()
    )

    assert categories["600000.SH"] == study.CANDIDATE
    assert categories["000000.SZ"] == study.NEGATIVE_CONTROL
    candidate = classified.filter(pl.col("symbol") == "600000.SH").row(
        0, named=True
    )
    assert candidate["industry_peer_count"] == 7
    assert candidate["industry_peer_positive_share"] == 6 / 7


def test_attach_industry_respects_membership_end_date() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH"],
            "ann_date": [date(2020, 1, 10), date(2020, 1, 10)],
        }
    )
    membership = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH"],
            "l1_code": ["A", "B"],
            "l1_name": ["甲", "乙"],
            "in_date": [date(2019, 1, 1), date(2019, 1, 1)],
            "out_date": [None, date(2019, 12, 31)],
        },
        schema_overrides={"out_date": pl.Date},
    )

    mapped, audit = study.attach_industry(events, membership)

    assert mapped.get_column("symbol").to_list() == ["600000.SH"]
    assert audit["mapping_rate"] == 0.5


def test_evaluate_requires_candidate_to_beat_both_controls() -> None:
    candidate = {
        "universe_eligible_events": 450,
        "tradable_events": 440,
        "announcement_days": 70,
        "tradable_rate": 0.97,
        "benchmark_coverage": 1.0,
        "entry_capacity_feasible_rate": 1.0,
        "unresolved_exits": 2,
        "mean_net_return": 0.018,
        "mean_excess_return": 0.020,
        "excess_daily_cluster_t": 3.2,
        "positive_excess_years": 6,
        "max_year_positive_excess_share": 0.30,
    }
    primary = {
        study.CANDIDATE: candidate,
        study.NEUTRAL_CONTROL: {"mean_excess_return": 0.012},
        study.NEGATIVE_CONTROL: {"mean_excess_return": 0.005},
    }

    decision = study.evaluate(primary)

    assert decision["passed"] is True
    assert decision["verdict"] == "PROMOTE_TO_ACCOUNT_CONTRACT"

