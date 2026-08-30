"""Audit analyst-report history and potential revision sample sizes without prices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START_YEAR = 2017
END_YEAR = 2020
COMPARISON_DAYS = 180
BREADTH_DAYS = 20
COOLDOWN_DAYS = 90
TARGET_REVISION_MINIMUM = 0.10
RATING_RANK = {
    "卖出": 0,
    "减持": 1,
    "中性": 2,
    "持有": 2,
    "增持": 3,
    "买入": 4,
}


def load_reports(data_dir: Path) -> pl.DataFrame:
    paths = [
        data_dir / "event_data" / "analyst_reports" / f"year={year}" / "part.parquet"
        for year in range(START_YEAR, END_YEAR + 1)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("all 2017-2020 analyst-report partitions are required")
    return pl.read_parquet(paths).sort(
        ["symbol", "org_code", "publish_date", "report_id"]
    )


def _target_midpoint(high: Any, low: Any) -> float | None:
    values = []
    for value in (high, low):
        if value is not None and float(value) > 0:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def prepare_revisions(reports: pl.DataFrame) -> pl.DataFrame:
    canonical = reports.unique(
        subset=["symbol", "org_code", "publish_date"],
        keep="last",
        maintain_order=True,
    ).sort(["symbol", "org_code", "publish_date", "report_id"])
    rows = []
    for group in canonical.partition_by(["symbol", "org_code"], maintain_order=True):
        previous: dict[str, Any] | None = None
        for row in group.iter_rows(named=True):
            midpoint = _target_midpoint(
                row.get("target_price_high"), row.get("target_price_low")
            )
            target_revision = None
            rating_upgrade = False
            comparison_available = False
            if previous is not None:
                gap = (row["publish_date"] - previous["publish_date"]).days
                comparison_available = 0 < gap <= COMPARISON_DAYS
                if comparison_available:
                    prior_midpoint = previous["target_midpoint"]
                    if midpoint is not None and prior_midpoint is not None and prior_midpoint > 0:
                        target_revision = midpoint / prior_midpoint - 1.0
                    current_rank = RATING_RANK.get(str(row.get("current_rating") or ""))
                    prior_rank = RATING_RANK.get(
                        str(previous.get("current_rating") or "")
                    )
                    rating_upgrade = bool(
                        current_rank is not None
                        and prior_rank is not None
                        and current_rank > prior_rank
                    )
            rows.append(
                {
                    **row,
                    "target_midpoint": midpoint,
                    "comparison_available": comparison_available,
                    "target_revision": target_revision,
                    "target_up_10pct": bool(
                        target_revision is not None
                        and target_revision >= TARGET_REVISION_MINIMUM
                    ),
                    "rating_upgrade": rating_upgrade,
                }
            )
            previous = {**row, "target_midpoint": midpoint}
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def breadth_signals(
    revisions: pl.DataFrame,
    event_column: str,
    minimum_brokers: int,
) -> list[dict[str, Any]]:
    scoped = revisions.filter(pl.col(event_column)).sort(
        ["symbol", "publish_date", "org_code"]
    )
    events_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped.iter_rows(named=True):
        events_by_symbol[row["symbol"]].append(row)
    signals = []
    for symbol, events in events_by_symbol.items():
        window: list[dict[str, Any]] = []
        last_signal: date | None = None
        for event in events:
            event_date = event["publish_date"]
            window = [
                row
                for row in window
                if 0 <= (event_date - row["publish_date"]).days <= BREADTH_DAYS
            ]
            window.append(event)
            brokers = {str(row["org_code"]) for row in window if row["org_code"]}
            if len(brokers) < minimum_brokers:
                continue
            if last_signal is not None and (event_date - last_signal).days < COOLDOWN_DAYS:
                continue
            signals.append(
                {
                    "symbol": symbol,
                    "signal_date": event_date,
                    "broker_count": len(brokers),
                    "event_type": event_column,
                }
            )
            last_signal = event_date
    return signals


def _signal_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "signals": len(signals),
        "signal_days": len({row["signal_date"] for row in signals}),
        "symbols": len({row["symbol"] for row in signals}),
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    reports = load_reports(data_dir)
    revisions = prepare_revisions(reports)
    yearly = (
        revisions.with_columns(pl.col("publish_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("reports"),
            pl.col("report_id").n_unique().alias("unique_reports"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("org_code").n_unique().alias("brokers"),
            pl.col("target_midpoint").is_not_null().sum().alias("target_covered"),
            pl.col("comparison_available").sum().alias("history_comparable"),
            pl.col("target_up_10pct").sum().alias("target_up_events"),
            pl.col("rating_upgrade").sum().alias("rating_upgrade_events"),
        )
        .sort("year")
    )
    candidates = {}
    for event_column in ("target_up_10pct", "rating_upgrade"):
        candidates[event_column] = {
            "raw_events": revisions.filter(pl.col(event_column)).height,
            "breadth_2": _signal_summary(
                breadth_signals(revisions, event_column, minimum_brokers=2)
            ),
            "breadth_3": _signal_summary(
                breadth_signals(revisions, event_column, minimum_brokers=3)
            ),
        }
    payload = {
        "schema_version": "p0-analyst-report-metadata-audit-v1",
        "period": {"start_year": START_YEAR, "end_year": END_YEAR},
        "outcome_fields_read": False,
        "assumptions": {
            "same_broker_comparison_days": COMPARISON_DAYS,
            "target_revision_minimum": TARGET_REVISION_MINIMUM,
            "breadth_calendar_days": BREADTH_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
        },
        "data": {
            "raw_reports": reports.height,
            "canonical_broker_stock_days": revisions.height,
            "unique_reports": reports.get_column("report_id").n_unique(),
            "symbols": reports.get_column("symbol").n_unique(),
            "brokers": reports.get_column("org_code").n_unique(),
            "yearly": yearly.to_dicts(),
        },
        "candidate_sample_sizes": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_analyst_report_metadata_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
