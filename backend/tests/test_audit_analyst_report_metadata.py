from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_analyst_report_metadata.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_analyst_report_metadata", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _report(
    report_id: str,
    day: date,
    org: str,
    target: float,
    rating: str,
) -> dict:
    return {
        "report_id": report_id,
        "publish_date": day,
        "symbol": "000001.SZ",
        "org_code": org,
        "target_price_high": target,
        "target_price_low": target,
        "current_rating": rating,
    }


def test_prepare_revisions_compares_only_prior_same_broker_history() -> None:
    reports = pl.DataFrame(
        [
            _report("a", date(2020, 1, 1), "A", 10.0, "增持"),
            _report("b", date(2020, 2, 1), "A", 12.0, "买入"),
            _report("c", date(2020, 9, 1), "A", 20.0, "买入"),
        ]
    )

    revisions = audit.prepare_revisions(reports)

    assert revisions["target_up_10pct"].to_list() == [False, True, False]
    assert revisions["rating_upgrade"].to_list() == [False, True, False]
    assert revisions["comparison_available"].to_list() == [False, True, False]


def test_breadth_signal_requires_distinct_brokers_and_applies_cooldown() -> None:
    revisions = pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 4,
            "publish_date": [
                date(2020, 1, 1),
                date(2020, 1, 5),
                date(2020, 1, 10),
                date(2020, 4, 10),
            ],
            "org_code": ["A", "B", "C", "D"],
            "target_up_10pct": [True] * 4,
        }
    )

    signals = audit.breadth_signals(
        revisions, "target_up_10pct", minimum_brokers=2
    )

    assert signals == [
        {
            "symbol": "000001.SZ",
            "signal_date": date(2020, 1, 5),
            "broker_count": 2,
            "event_type": "target_up_10pct",
        }
    ]
