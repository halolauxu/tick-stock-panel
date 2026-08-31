"""Collect index membership metadata before opening constituent returns."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2013, 1, 1)
END = date(2020, 12, 31)
INDEX_CODES = ("000300.SH", "000905.SH")
FIELDS = ("index_code", "con_code", "trade_date", "weight")
OUTCOME_FIELDS = {
    "open",
    "close",
    "return",
    "future_return",
    "forward_return",
    "net_return",
}
MIN_MONTH_COVERAGE = 0.95
MIN_REGULAR_CYCLES_PER_INDEX = 12
MIN_ADDITIONS = 500


def month_ranges(start: date = START, end: date = END) -> list[tuple[date, date]]:
    ranges = []
    for year in range(start.year, end.year + 1):
        first_month = start.month if year == start.year else 1
        last_month = end.month if year == end.year else 12
        for month in range(first_month, last_month + 1):
            last_day = calendar.monthrange(year, month)[1]
            ranges.append((date(year, month, 1), date(year, month, last_day)))
    return ranges


def normalize_weights(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "index_code": pl.Utf8,
                "symbol": pl.Utf8,
                "snapshot_date": pl.Date,
                "weight_pct": pl.Float64,
            }
        )
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "con_code": "symbol",
                "trade_date": "snapshot_date",
                "weight": "weight_pct",
            }
        )
        .with_columns(
            pl.col("index_code").cast(pl.Utf8).str.strip_chars(),
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("snapshot_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
            pl.col("weight_pct").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("index_code").is_in(INDEX_CODES)
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("snapshot_date").is_between(START, END, closed="both")
            & pl.col("weight_pct").is_not_null()
            & (pl.col("weight_pct") > 0)
        )
        .unique(["index_code", "symbol", "snapshot_date"], keep="last")
        .sort(["index_code", "snapshot_date", "symbol"])
    )


def derive_regular_additions(weights: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "index_code": pl.Utf8,
        "cycle_month": pl.Date,
        "previous_snapshot_date": pl.Date,
        "current_snapshot_date": pl.Date,
        "symbol": pl.Utf8,
        "weight_pct": pl.Float64,
    }
    if weights.is_empty():
        return pl.DataFrame(schema=schema)
    output: list[dict[str, Any]] = []
    snapshots: dict[tuple[str, int, int], tuple[date, dict[str, float]]] = {}
    for key, group in weights.partition_by(
        ["index_code", "snapshot_date"], as_dict=True
    ).items():
        index_code, snapshot_date = key
        snapshots[(index_code, snapshot_date.year, snapshot_date.month)] = (
            snapshot_date,
            dict(zip(group["symbol"].to_list(), group["weight_pct"].to_list(), strict=True)),
        )
    for index_code in INDEX_CODES:
        for year in range(START.year, END.year + 1):
            for month, previous_month in ((6, 5), (12, 11)):
                previous = snapshots.get((index_code, year, previous_month))
                current = snapshots.get((index_code, year, month))
                if previous is None or current is None:
                    continue
                previous_date, previous_members = previous
                current_date, current_members = current
                for symbol in sorted(set(current_members) - set(previous_members)):
                    output.append(
                        {
                            "index_code": index_code,
                            "cycle_month": date(year, month, 1),
                            "previous_snapshot_date": previous_date,
                            "current_snapshot_date": current_date,
                            "symbol": symbol,
                            "weight_pct": current_members[symbol],
                        }
                    )
    return pl.DataFrame(output, schema=schema).sort(
        ["cycle_month", "index_code", "symbol"]
    )


def audit(weights: pl.DataFrame, additions: pl.DataFrame) -> dict[str, Any]:
    expected_months = len(month_ranges())
    snapshots = (
        weights.with_columns(
            pl.col("snapshot_date").dt.strftime("%Y-%m").alias("month")
        )
        .group_by("index_code")
        .agg(
            pl.col("month").n_unique().alias("months"),
            pl.col("snapshot_date").n_unique().alias("snapshot_dates"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.len().alias("rows"),
        )
        .sort("index_code")
    )
    snapshot_rows = {row["index_code"]: row for row in snapshots.to_dicts()}
    cycle_counts = (
        additions.group_by("index_code")
        .agg(
            pl.col("cycle_month").n_unique().alias("cycles"),
            pl.len().alias("additions"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("index_code")
    )
    cycle_rows = {row["index_code"]: row for row in cycle_counts.to_dicts()}
    checks = {
        "both_indices_present": set(snapshot_rows) == set(INDEX_CODES),
        "membership_keys_unique": weights.unique(
            ["index_code", "symbol", "snapshot_date"]
        ).height
        == weights.height,
        "weights_positive": weights.filter(pl.col("weight_pct") <= 0).height == 0,
        "outcome_fields_absent": not (OUTCOME_FIELDS & set(weights.columns)),
        "each_index_month_coverage_at_least_95pct": all(
            snapshot_rows.get(code, {}).get("months", 0) / expected_months
            >= MIN_MONTH_COVERAGE
            for code in INDEX_CODES
        ),
        "each_index_has_at_least_12_regular_cycles": all(
            cycle_rows.get(code, {}).get("cycles", 0)
            >= MIN_REGULAR_CYCLES_PER_INDEX
            for code in INDEX_CODES
        ),
        "at_least_500_additions": additions.height >= MIN_ADDITIONS,
    }
    integrity = (
        "both_indices_present",
        "membership_keys_unique",
        "weights_positive",
        "outcome_fields_absent",
    )
    if not all(checks[name] for name in integrity):
        status = "DATA_GAP"
    elif all(checks.values()):
        status = "SAMPLE_SUFFICIENT_PENDING_OFFICIAL_NOTICE_MATCH"
    else:
        status = "SAMPLE_SPARSE"
    return {
        "status": status,
        "price_data_read": False,
        "future_returns_read": False,
        "period": {"start": START, "end": END},
        "expected_months_per_index": expected_months,
        "weights": snapshots.to_dicts(),
        "regular_cycles": cycle_counts.to_dicts(),
        "addition_rows": additions.height,
        "addition_symbols": additions["symbol"].n_unique(),
        "checks": checks,
    }


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    frames: list[pl.DataFrame] = []
    try:
        total = len(INDEX_CODES) * len(month_ranges())
        progress = 0
        for index_code in INDEX_CODES:
            for month_start, month_end in month_ranges():
                frame = normalize_weights(
                    client.query(
                        "index_weight",
                        {
                            "index_code": index_code,
                            "start_date": month_start.strftime("%Y%m%d"),
                            "end_date": month_end.strftime("%Y%m%d"),
                        },
                        FIELDS,
                    )
                )
                if not frame.is_empty():
                    frames.append(frame)
                progress += 1
                if progress == 1 or progress % 12 == 0 or progress == total:
                    print(
                        f"index_weight_progress={progress}/{total} "
                        f"returned_months={len(frames)}",
                        flush=True,
                    )
    finally:
        client.close()
    weights = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(["index_code", "symbol", "snapshot_date"], keep="last")
        .sort(["index_code", "snapshot_date", "symbol"])
        if frames
        else normalize_weights([])
    )
    additions = derive_regular_additions(weights)
    root = data_dir / "research" / "index_inclusion"
    _atomic_parquet(weights, root / "monthly_weights.parquet")
    _atomic_parquet(additions, root / "regular_additions.parquet")
    payload = {
        "schema_version": "p0-index-inclusion-data-v1",
        "contract_frozen": "2026-08-31",
        **audit(weights, additions),
        "artifacts": {
            "monthly_weights": str(root / "monthly_weights.parquet"),
            "regular_additions": str(root / "regular_additions.parquet"),
        },
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
        default=Path("/app/data/research/p0_index_inclusion_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
