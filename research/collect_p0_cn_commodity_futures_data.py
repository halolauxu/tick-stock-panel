"""Collect and audit frozen Chinese commodity-futures research data."""
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

START = date(2014, 1, 1)
END = date(2020, 12, 31)
SERIES = (
    "CU.SHF",
    "AL.SHF",
    "ZN.SHF",
    "AU.SHF",
    "AG.SHF",
    "RB.SHF",
    "HC.SHF",
    "M.DCE",
    "Y.DCE",
    "P.DCE",
    "C.DCE",
    "I.DCE",
    "J.DCE",
    "JM.DCE",
    "CF.ZCE",
    "SR.ZCE",
    "TA.ZCE",
    "MA.ZCE",
    "RM.ZCE",
    "OI.ZCE",
)
EXCHANGES = ("SHFE", "DCE", "CZCE")
MASTER_FIELDS = (
    "ts_code",
    "symbol",
    "exchange",
    "name",
    "fut_code",
    "per_unit",
    "trade_unit",
    "quote_unit",
    "list_date",
    "delist_date",
    "d_month",
    "last_ddate",
)
DAILY_FIELDS = (
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
MAPPING_FIELDS = ("ts_code", "trade_date", "mapping_ts_code")


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "contract"})
        .with_columns(
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
            pl.col("fut_code").cast(pl.Utf8).str.to_uppercase(),
            pl.col("per_unit").cast(pl.Float64, strict=False),
            *[
                pl.col(column)
                .cast(pl.Utf8)
                .str.to_date("%Y%m%d", strict=False)
                .alias(column)
                for column in ("list_date", "delist_date", "last_ddate")
            ],
        )
        .filter(
            pl.col("list_date").is_not_null()
            & (pl.col("list_date") <= pl.lit(END))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("delist_date") >= pl.lit(START))
            )
        )
        .unique(subset=["contract"], keep="last")
        .sort("contract")
    )


def normalize_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {"ts_code": "series", "trade_date": "date", "vol": "volume"}
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "settle",
                    "volume",
                    "amount",
                    "oi",
                )
            ],
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique(subset=["series", "date"], keep="last")
        .sort(["series", "date"])
    )


def normalize_mapping(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "series",
                "trade_date": "date",
                "mapping_ts_code": "contract",
            }
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique(subset=["series", "date"], keep="last")
        .sort(["series", "date"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        master_rows: list[dict[str, Any]] = []
        for exchange in EXCHANGES:
            master_rows.extend(
                client.query(
                    "fut_basic",
                    {"exchange": exchange, "fut_type": "1"},
                    MASTER_FIELDS,
                )
            )
        master = normalize_master(master_rows)
        daily_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        for index, series in enumerate(SERIES, start=1):
            params = {
                "ts_code": series,
                "start_date": START.strftime("%Y%m%d"),
                "end_date": END.strftime("%Y%m%d"),
            }
            daily_rows.extend(client.query("fut_daily", params, DAILY_FIELDS))
            mapping_rows.extend(
                client.query("fut_mapping", params, MAPPING_FIELDS)
            )
            print(
                f"futures_progress={index}/{len(SERIES)} series={series} "
                f"daily_rows={len(daily_rows)} mapping_rows={len(mapping_rows)}",
                flush=True,
            )
        daily = normalize_daily(daily_rows)
        mapping = normalize_mapping(mapping_rows)
    finally:
        client.close()

    mapped_contracts = set(mapping["contract"].drop_nulls().to_list())
    master = master.filter(pl.col("contract").is_in(mapped_contracts))
    joined = daily.join(mapping, on=["series", "date"], how="left")
    counts = (
        joined.group_by("series")
        .agg(
            pl.len().alias("daily_rows"),
            pl.col("contract").is_not_null().sum().alias("mapped_rows"),
        )
        .with_columns(
            (pl.col("mapped_rows") / pl.col("daily_rows")).alias(
                "mapping_coverage"
            )
        )
        .sort("series")
    )
    invalid_active = daily.filter(
        (pl.col("volume") > 0)
        & (
            (pl.col("open") <= 0)
            | (pl.col("high") <= 0)
            | (pl.col("low") <= 0)
            | (pl.col("close") <= 0)
            | (pl.col("settle") <= 0)
        )
    ).height
    negative_rows = daily.filter(
        (pl.col("volume") < 0) | (pl.col("oi") < 0)
    ).height
    master_contracts = set(master["contract"].to_list())
    missing_contracts = sorted(mapped_contracts - master_contracts)
    invalid_units = master.filter(
        pl.col("contract").is_in(mapped_contracts)
        & (pl.col("per_unit").is_null() | (pl.col("per_unit") <= 0))
    ).height
    checks = {
        "daily_unique": daily.unique(["series", "date"]).height == daily.height,
        "mapping_unique": mapping.unique(["series", "date"]).height
        == mapping.height,
        "all_series_present": counts.height == len(SERIES),
        "each_series_at_least_1500_rows": counts.filter(
            pl.col("daily_rows") < 1500
        ).is_empty(),
        "mapping_coverage_at_least_99pct": counts.filter(
            pl.col("mapping_coverage") < 0.99
        ).is_empty(),
        "active_prices_valid": invalid_active == 0,
        "volume_and_open_interest_nonnegative": negative_rows == 0,
        "mapped_contract_master_complete": not missing_contracts,
        "mapped_contract_units_valid": invalid_units == 0,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    root = data_dir / "research" / "cn_commodity_futures"
    _atomic_parquet(master, root / "contracts.parquet")
    _atomic_parquet(daily, root / "continuous_daily.parquet")
    _atomic_parquet(mapping, root / "main_mapping.parquet")
    payload = {
        "schema_version": "p0-cn-commodity-futures-data-v2",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": START,
            "end": END,
            "validation_returns_read": False,
        },
        "status": status,
        "counts": {
            "frozen_series": len(SERIES),
            "contract_rows": master.height,
            "daily_rows": daily.height,
            "mapping_rows": mapping.height,
            "mapped_contracts": len(mapped_contracts),
            "invalid_active_rows": invalid_active,
            "negative_market_rows": negative_rows,
            "invalid_unit_contracts": invalid_units,
            "by_series": counts.to_dicts(),
        },
        "checks": checks,
        "missing_mapped_contracts": missing_contracts,
        "artifacts": {
            "contracts": str(root / "contracts.parquet"),
            "continuous_daily": str(root / "continuous_daily.parquet"),
            "main_mapping": str(root / "main_mapping.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    payload["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_cn_commodity_futures_data_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
