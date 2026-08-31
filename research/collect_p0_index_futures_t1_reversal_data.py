"""Collect frozen index and CFFEX futures data without evaluating returns."""
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
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2015, 1, 1)
END = date(2026, 8, 28)
SERIES_TO_INDEX = {
    "IF.CFX": "000300.SH",
    "IH.CFX": "000016.SH",
    "IC.CFX": "000905.SH",
}
CONTRACT_MULTIPLIERS = {"IF": 300.0, "IH": 300.0, "IC": 200.0}
FUTURE_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "vol",
    "amount",
    "oi",
)
INDEX_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
)
MAPPING_FIELDS = ("ts_code", "trade_date", "mapping_ts_code")
MASTER_FIELDS = (
    "ts_code",
    "symbol",
    "exchange",
    "name",
    "fut_code",
    "per_unit",
    "list_date",
    "delist_date",
)


def year_ranges(start: date = START, end: date = END) -> list[tuple[date, date]]:
    output = []
    for year in range(start.year, end.year + 1):
        left = max(start, date(year, 1, 1))
        right = min(end, date(year, 12, 31))
        output.append((left, right))
    return output


def normalize_market(rows: list[dict[str, Any]], *, future: bool) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    value_columns = ["open", "high", "low", "close", "vol", "amount"]
    if future:
        value_columns.extend(["settle", "oi"])
    else:
        value_columns.append("pre_close")
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "instrument",
                "trade_date": "date",
                "vol": "volume",
            }
        )
        .with_columns(
            pl.col("instrument").cast(pl.Utf8).str.strip_chars(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in value_columns
                if column != "vol"
            ],
            pl.col("volume").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique(["instrument", "date"], keep="last")
        .sort(["instrument", "date"])
    )


def normalize_mapping(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "instrument",
                "trade_date": "date",
                "mapping_ts_code": "contract",
            }
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique(["instrument", "date"], keep="last")
        .sort(["instrument", "date"])
    )


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "contract"})
        .with_columns(
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
            pl.col("per_unit").cast(pl.Float64, strict=False),
            pl.col("fut_code")
            .replace_strict(CONTRACT_MULTIPLIERS, default=None)
            .cast(pl.Float64)
            .alias("contract_multiplier"),
            *[
                pl.col(column)
                .cast(pl.Utf8)
                .str.to_date("%Y%m%d", strict=False)
                for column in ("list_date", "delist_date")
            ],
        )
        .unique("contract", keep="last")
        .sort("contract")
    )


def audit(
    futures: pl.DataFrame,
    indices: pl.DataFrame,
    mapping: pl.DataFrame,
    master: pl.DataFrame,
) -> dict[str, Any]:
    future_counts = futures.group_by("instrument").len().sort("instrument")
    index_counts = indices.group_by("instrument").len().sort("instrument")
    active_invalid = futures.filter(
        (pl.col("volume") > 0)
        & (
            (pl.col("open") <= 0)
            | (pl.col("high") <= 0)
            | (pl.col("low") <= 0)
            | (pl.col("close") <= 0)
            | (pl.col("settle") <= 0)
        )
    ).height
    index_invalid = indices.filter(
        (pl.col("volume") > 0)
        & (
            (pl.col("open") <= 0)
            | (pl.col("high") <= 0)
            | (pl.col("low") <= 0)
            | (pl.col("close") <= 0)
            | (pl.col("pre_close") <= 0)
        )
    ).height
    mapped_contracts = set(mapping["contract"].drop_nulls().to_list())
    master_rows = master.filter(pl.col("contract").is_in(mapped_contracts))
    missing_contracts = sorted(mapped_contracts - set(master_rows["contract"].to_list()))
    mapping_join = futures.join(mapping, on=["instrument", "date"], how="left")
    mapping_coverage = float(mapping_join["contract"].is_not_null().mean())
    checks = {
        "all_three_futures_present": set(future_counts["instrument"].to_list())
        == set(SERIES_TO_INDEX),
        "all_three_indices_present": set(index_counts["instrument"].to_list())
        == set(SERIES_TO_INDEX.values()),
        "each_future_at_least_2700_days": future_counts.filter(pl.col("len") < 2700).is_empty(),
        "each_index_at_least_2700_days": index_counts.filter(pl.col("len") < 2700).is_empty(),
        "future_keys_unique": futures.unique(["instrument", "date"]).height == futures.height,
        "index_keys_unique": indices.unique(["instrument", "date"]).height == indices.height,
        "mapping_keys_unique": mapping.unique(["instrument", "date"]).height == mapping.height,
        "mapping_coverage_at_least_99pct": mapping_coverage >= 0.99,
        "active_future_prices_valid": active_invalid == 0,
        "active_index_prices_valid": index_invalid == 0,
        "mapped_contracts_in_master": not missing_contracts,
        "mapped_contract_multipliers_valid": master_rows.filter(
            pl.col("contract_multiplier").is_null()
            | (pl.col("contract_multiplier") <= 0)
        ).is_empty(),
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "returns_evaluated": False,
        "development_metrics_computed": False,
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "counts": {
            "futures": future_counts.to_dicts(),
            "indices": index_counts.to_dicts(),
            "mapping_rows": mapping.height,
            "mapped_contracts": len(mapped_contracts),
            "mapping_coverage": mapping_coverage,
            "active_invalid_future_rows": active_invalid,
            "active_invalid_index_rows": index_invalid,
        },
        "checks": checks,
        "missing_contracts": missing_contracts,
    }


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    future_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    try:
        master_rows = client.query(
            "fut_basic", {"exchange": "CFFEX", "fut_type": "1"}, MASTER_FIELDS
        )
        total = len(year_ranges()) * len(SERIES_TO_INDEX)
        progress = 0
        for series, index_code in SERIES_TO_INDEX.items():
            for left, right in year_ranges():
                params = {
                    "start_date": left.strftime("%Y%m%d"),
                    "end_date": right.strftime("%Y%m%d"),
                }
                future_rows.extend(
                    client.query("fut_daily", {**params, "ts_code": series}, FUTURE_FIELDS)
                )
                mapping_rows.extend(
                    client.query("fut_mapping", {**params, "ts_code": series}, MAPPING_FIELDS)
                )
                index_rows.extend(
                    client.query("index_daily", {**params, "ts_code": index_code}, INDEX_FIELDS)
                )
                progress += 1
                print(f"index_futures_data_progress={progress}/{total}", flush=True)
    finally:
        client.close()
    futures = normalize_market(future_rows, future=True)
    indices = normalize_market(index_rows, future=False)
    mapping = normalize_mapping(mapping_rows)
    master = normalize_master(master_rows)
    root = data_dir / "research" / "index_futures_t1_reversal"
    _atomic_parquet(futures, root / "continuous_futures.parquet")
    _atomic_parquet(indices, root / "underlying_indices.parquet")
    _atomic_parquet(mapping, root / "main_mapping.parquet")
    _atomic_parquet(master, root / "contracts.parquet")
    payload = {
        "schema_version": "p0-index-futures-t1-reversal-data-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": START, "end": END},
        **audit(futures, indices, mapping, master),
        "artifacts": {
            "futures": str(root / "continuous_futures.parquet"),
            "indices": str(root / "underlying_indices.parquet"),
            "mapping": str(root / "main_mapping.parquet"),
            "contracts": str(root / "contracts.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({**payload, "sha256": digest}, ensure_ascii=False, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_index_futures_t1_reversal_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
