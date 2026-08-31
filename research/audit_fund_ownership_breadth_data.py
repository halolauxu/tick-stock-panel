"""Audit public-fund ownership metadata and candidate counts without prices."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START_YEAR = 2017
START_QUARTER = 1
END_YEAR = 2026
END_QUARTER = 2
DEVELOPMENT_END_YEAR = 2020
MIN_AVERAGE_HOLDING_CNY = 1_000_000.0
MIN_COUNT_INCREASE = 10
MIN_SAME_DEPTH_MAX_COVERAGE_RATIO = 0.25


def required_periods() -> list[tuple[int, int]]:
    output = []
    for year in range(START_YEAR, END_YEAR + 1):
        for quarter in range(1, 5):
            current = year * 10 + quarter
            if START_YEAR * 10 + START_QUARTER <= current <= END_YEAR * 10 + END_QUARTER:
                output.append((year, quarter))
    return output


def load_events(data_dir: Path) -> tuple[pl.DataFrame, int]:
    root = data_dir / "event_data" / "fund_ownership_breadth"
    frames = []
    missing = []
    for year, quarter in required_periods():
        path = root / f"year={year}" / f"quarter={quarter}" / "part.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue
        frames.append(
            pl.read_parquet(path).with_columns(
                pl.lit(year).alias("partition_year"),
                pl.lit(quarter).alias("partition_quarter"),
            )
        )
    if missing:
        raise ValueError(
            f"all 38 fund ownership partitions are required; missing={len(missing)}"
        )
    return pl.concat(frames, how="vertical_relaxed"), len(frames)


def quarter_quality(events: pl.DataFrame) -> pl.DataFrame:
    quarterly = (
        events.with_columns(
            (
                pl.col("period_end").dt.year() * 4
                + ((pl.col("period_end").dt.month() - 1) // 3)
            ).alias("period_ordinal")
        )
        .group_by("period_end", "period_ordinal")
        .agg(
            pl.len().alias("rows"),
            pl.col("fund_coverage_count").max().alias("maximum_coverage"),
            pl.col("market_fund_count").n_unique().alias("market_count_versions"),
            pl.col("market_fund_count").first().alias("market_fund_count"),
        )
        .sort("period_end")
    )
    prior = quarterly.select(
        (pl.col("period_ordinal") + 2).alias("period_ordinal"),
        pl.col("maximum_coverage").alias("same_depth_prior_maximum_coverage"),
    )
    return (
        quarterly.join(prior, on="period_ordinal", how="left")
        .with_columns(
            (
                pl.col("maximum_coverage")
                / pl.col("same_depth_prior_maximum_coverage")
            ).alias("same_depth_maximum_coverage_ratio")
        )
        .with_columns(
            (
                pl.col("same_depth_prior_maximum_coverage").is_null()
                | (
                    pl.col("same_depth_maximum_coverage_ratio")
                    >= MIN_SAME_DEPTH_MAX_COVERAGE_RATIO
                )
            ).alias("complete")
        )
        .sort("period_end")
    )


def same_depth_changes(events: pl.DataFrame, quality: pl.DataFrame) -> pl.DataFrame:
    complete_periods = (
        quality.filter(pl.col("complete")).get_column("period_end").to_list()
    )
    current = events.filter(pl.col("period_end").is_in(complete_periods)).with_columns(
        (
            pl.col("period_end").dt.year() * 4
            + ((pl.col("period_end").dt.month() - 1) // 3)
        ).alias("period_ordinal")
    )
    previous = current.select(
        "symbol",
        (pl.col("period_ordinal") + 2).alias("period_ordinal"),
        pl.col("fund_coverage_count").alias("previous_fund_coverage_count"),
        pl.col("coverage_share").alias("previous_coverage_share"),
    )
    return (
        current.join(previous, on=["symbol", "period_ordinal"], how="left")
        .with_columns(
            (
                pl.col("fund_coverage_count")
                - pl.col("previous_fund_coverage_count")
            ).alias("fund_count_increase"),
            (
                pl.col("coverage_share") / pl.col("previous_coverage_share") - 1.0
            ).alias("coverage_share_growth"),
        )
        .sort(["period_end", "symbol"])
    )


def _candidate_summary(frame: pl.DataFrame, condition: pl.Expr) -> dict[str, Any]:
    candidates = frame.filter(
        condition
        & (
            pl.col("average_market_value_per_fund_cny")
            >= MIN_AVERAGE_HOLDING_CNY
        )
    )
    if candidates.is_empty():
        return {"events": 0, "quarters": 0, "symbols": 0, "yearly": []}
    yearly = (
        candidates.with_columns(pl.col("period_end").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("events"),
            pl.col("period_end").n_unique().alias("quarters"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("year")
        .to_dicts()
    )
    return {
        "events": candidates.height,
        "quarters": candidates.get_column("period_end").n_unique(),
        "symbols": candidates.get_column("symbol").n_unique(),
        "yearly": yearly,
    }


def audit(events: pl.DataFrame, partition_count: int) -> dict[str, Any]:
    duplicate_rows = events.height - events.unique(
        subset=["period_end", "symbol"]
    ).height
    invalid_symbols = events.filter(
        ~pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
    ).height
    invalid_counts = events.filter(
        (pl.col("fund_coverage_count") <= 0)
        | (pl.col("market_fund_count") <= 0)
        | (pl.col("fund_coverage_count") > pl.col("market_fund_count"))
        | (pl.col("total_shares") <= 0)
        | (pl.col("total_market_value_cny") <= 0)
    ).height
    partition_mismatch = events.filter(
        (pl.col("period_end").dt.year() != pl.col("partition_year"))
        | (
            ((pl.col("period_end").dt.month() - 1) // 3 + 1)
            != pl.col("partition_quarter")
        )
    ).height
    quality = quarter_quality(events)
    incomplete_periods = quality.filter(~pl.col("complete")).get_column(
        "period_end"
    ).to_list()
    development_quality = quality.filter(
        pl.col("period_end").dt.year() <= DEVELOPMENT_END_YEAR
    )
    development_complete = bool(
        development_quality.height == 16
        and development_quality.get_column("complete").all()
    )
    changes = same_depth_changes(events, quality)
    candidates = {
        "share_growth_50pct_count_plus_10": _candidate_summary(
            changes,
            (pl.col("previous_coverage_share") > 0)
            & (pl.col("fund_count_increase") >= MIN_COUNT_INCREASE)
            & (pl.col("coverage_share_growth") >= 0.50),
        ),
        "share_growth_100pct_count_plus_10": _candidate_summary(
            changes,
            (pl.col("previous_coverage_share") > 0)
            & (pl.col("fund_count_increase") >= MIN_COUNT_INCREASE)
            & (pl.col("coverage_share_growth") >= 1.00),
        ),
        "new_coverage_at_least_10": _candidate_summary(
            changes,
            pl.col("previous_coverage_share").is_null()
            & (pl.col("fund_coverage_count") >= 10),
        ),
    }
    base_valid = bool(
        partition_count == len(required_periods())
        and duplicate_rows == 0
        and invalid_symbols == 0
        and invalid_counts == 0
        and partition_mismatch == 0
    )
    latest_only_partial = incomplete_periods == [date(2026, 6, 30)]
    development_sample = candidates["share_growth_50pct_count_plus_10"]["yearly"]
    development_events = sum(
        row["events"] for row in development_sample if row["year"] <= 2020
    )
    development_metadata_passed = bool(
        base_valid
        and development_complete
        and latest_only_partial
        and development_events >= 100
    )
    return {
        "schema_version": "p0-fund-ownership-breadth-metadata-audit-v1",
        "status": (
            "LATEST_PARTIAL_DEVELOPMENT_USABLE"
            if development_metadata_passed
            else "DATA_GAP"
        ),
        "outcome_fields_read": False,
        "assumptions": {
            "same_disclosure_depth_lag_quarters": 2,
            "minimum_average_holding_cny": MIN_AVERAGE_HOLDING_CNY,
            "minimum_count_increase": MIN_COUNT_INCREASE,
            "minimum_same_depth_maximum_coverage_ratio": MIN_SAME_DEPTH_MAX_COVERAGE_RATIO,
        },
        "data": {
            "partitions": partition_count,
            "required_partitions": len(required_periods()),
            "rows": events.height,
            "symbols": events.get_column("symbol").n_unique(),
            "duplicate_rows": duplicate_rows,
            "invalid_symbols": invalid_symbols,
            "invalid_counts": invalid_counts,
            "partition_mismatch": partition_mismatch,
            "incomplete_periods": incomplete_periods,
            "quarterly_quality": quality.to_dicts(),
        },
        "candidate_sample_sizes": candidates,
        "decision": {
            "development_metadata_passed": development_metadata_passed,
            "latest_partial_excluded": latest_only_partial,
            "validation_or_stress_accepted": False,
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    events, partition_count = load_events(data_dir)
    payload = audit(events, partition_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
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
        default=Path("/app/data/research/p0_fund_ownership_breadth_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
