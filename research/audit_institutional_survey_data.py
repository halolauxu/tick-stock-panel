"""Audit institutional-survey metadata before opening price outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.run_p0_institutional_survey_attention_development import (  # noqa: E402
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    WARMUP_START,
    select_attention_spikes,
)

END_YEAR = DEVELOPMENT_END.year
KEY = ["event_id"]
OUTCOME_FIELDS = {
    "open",
    "close",
    "return",
    "net_return",
    "excess_return",
    "future_return",
    "forward_return",
}
MIN_SPIKES = 500
MIN_SIGNAL_DAYS = 300


def expected_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "event_data" / "institutional_survey"
    return [
        root / f"year={year}" / f"month={month:02d}" / "part.parquet"
        for year in range(WARMUP_START.year, END_YEAR + 1)
        for month in range(1, 13)
    ]


def audit(data_dir: Path) -> dict[str, Any]:
    planned = expected_paths(data_dir)
    present = [path for path in planned if path.is_file()]
    missing = [str(path) for path in planned if not path.is_file()]
    if missing:
        return {
            "status": "DATA_INCOMPLETE",
            "future_returns_read": False,
            "price_data_read": False,
            "planned_partitions": len(planned),
            "present_partitions": len(present),
            "missing_partitions": missing,
        }
    frame = pl.read_parquet(planned, hive_partitioning=False)
    duplicates = frame.height - frame.unique(KEY).height
    invalid = frame.filter(
        pl.col("event_id").is_null()
        | ~pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
        | pl.col("notice_date").is_null()
        | ~pl.col("notice_date").is_between(
            WARMUP_START, DEVELOPMENT_END, closed="both"
        )
        | (pl.col("institution_count") <= 0)
        | (pl.col("survey_session_count") <= 0)
        | (pl.col("institution_detail_rows") <= 0)
    ).height
    spikes = select_attention_spikes(frame)
    spike_days = spikes["ann_date"].n_unique() if spikes.height else 0
    yearly = (
        spikes.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("attention_spikes"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("ann_date").n_unique().alias("signal_days"),
        )
        .sort("year")
        .to_dicts()
        if spikes.height
        else []
    )
    checks = {
        "all_months_present": True,
        "event_id_unique": duplicates == 0,
        "symbols_dates_and_counts_valid": invalid == 0,
        "outcome_fields_absent": not (OUTCOME_FIELDS & set(frame.columns)),
        "attention_spikes_at_least_500": spikes.height >= MIN_SPIKES,
        "attention_spike_days_at_least_300": spike_days >= MIN_SIGNAL_DAYS,
    }
    integrity_checks = [
        "all_months_present",
        "event_id_unique",
        "symbols_dates_and_counts_valid",
        "outcome_fields_absent",
    ]
    if not all(checks[name] for name in integrity_checks):
        status = "DATA_GAP"
    elif checks["attention_spikes_at_least_500"] and checks[
        "attention_spike_days_at_least_300"
    ]:
        status = "SAMPLE_SUFFICIENT"
    else:
        status = "SAMPLE_SPARSE"
    return {
        "status": status,
        "future_returns_read": False,
        "price_data_read": False,
        "period": {
            "warmup_start": WARMUP_START,
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
        },
        "planned_partitions": len(planned),
        "present_partitions": len(present),
        "missing_partitions": [],
        "rows": frame.height,
        "symbols": frame["symbol"].n_unique() if frame.height else 0,
        "notice_days": frame["notice_date"].n_unique() if frame.height else 0,
        "duplicate_event_ids": duplicates,
        "invalid_rows": invalid,
        "attention_spikes": spikes.height,
        "attention_spike_symbols": (
            spikes["symbol"].n_unique() if spikes.height else 0
        ),
        "attention_spike_days": spike_days,
        "yearly": yearly,
        "checks": checks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "p0-institutional-survey-data-audit-v1",
        "contract_frozen": "2026-08-31",
        **audit(data_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": digest},
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
        default=Path(
            "/app/data/research/p0_institutional_survey_data_audit.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
