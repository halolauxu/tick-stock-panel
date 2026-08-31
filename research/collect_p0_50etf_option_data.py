"""Collect resumable 2015-2020 50ETF option data without evaluating returns."""
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

START = date(2015, 2, 9)
END = date(2020, 12, 31)
UNDERLYING = "510050.SH"
OPTION_CODE = "OP510050.SH"
MASTER_FIELDS = (
    "ts_code",
    "symbol",
    "exchange",
    "name",
    "per_unit",
    "opt_code",
    "opt_type",
    "call_put",
    "exercise_price",
    "opt_multiplier",
    "maturity_date",
    "list_date",
    "delist_date",
    "min_price_chg",
)
OPTION_FIELDS = (
    "ts_code",
    "trade_date",
    "exchange",
    "pre_settle",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "vol",
    "amount",
    "oi",
)
FUND_FIELDS = (
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


def normalize_master(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "contract"})
        .with_columns(
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
            pl.col("exercise_price").cast(pl.Float64, strict=False),
            pl.col("opt_multiplier").cast(pl.Float64, strict=False),
            pl.col("min_price_chg").cast(pl.Float64, strict=False),
            *[
                pl.col(column)
                .cast(pl.Utf8)
                .str.to_date("%Y%m%d", strict=False)
                for column in ("maturity_date", "list_date", "delist_date")
            ],
        )
        .filter(
            (pl.col("opt_code") == OPTION_CODE)
            & (pl.col("list_date") <= END)
            & (pl.col("delist_date") >= START)
        )
        .unique("contract", keep="last")
        .sort(["maturity_date", "exercise_price", "call_put", "contract"])
    )


def normalize_fund(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ("open", "high", "low", "close", "pre_close", "volume", "amount")
            ],
        )
        .filter(pl.col("date").is_between(START, END, closed="both"))
        .unique("date", keep="last")
        .sort("date")
    )


def normalize_options(
    rows: list[dict[str, Any]], allowed_contracts: set[str]
) -> pl.DataFrame:
    schema = {
        "contract": pl.Utf8,
        "date": pl.Date,
        "exchange": pl.Utf8,
        "pre_settle": pl.Float64,
        "pre_close": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "settle": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
        "open_interest": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "contract",
                "trade_date": "date",
                "vol": "volume",
                "oi": "open_interest",
            }
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in (
                    "pre_settle",
                    "pre_close",
                    "open",
                    "high",
                    "low",
                    "close",
                    "settle",
                    "volume",
                    "amount",
                    "open_interest",
                )
            ],
        )
        .filter(
            pl.col("contract").is_in(allowed_contracts)
            & pl.col("date").is_between(START, END, closed="both")
        )
        .select(*schema)
        .unique(["contract", "date"], keep="last")
        .sort(["date", "contract"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def audit(master: pl.DataFrame, fund: pl.DataFrame, options: pl.DataFrame) -> dict[str, Any]:
    standard = master.filter(pl.col("opt_multiplier") == 10_000.0)
    joined = options.join(master.select("contract", "call_put"), on="contract", how="left")
    dates = fund["date"].to_list()
    option_dates = set(options["date"].to_list())
    maturity_counts = standard.group_by("maturity_date").agg(
        pl.col("contract").n_unique().alias("contracts"),
        pl.col("call_put").n_unique().alias("sides"),
    )
    checks = {
        "underlying_has_at_least_1400_days": fund.height >= 1400,
        "at_least_60_monthly_maturities": maturity_counts.height >= 60,
        "each_maturity_has_calls_and_puts": maturity_counts.filter(pl.col("sides") < 2).is_empty(),
        "option_keys_unique": options.unique(["contract", "date"]).height == options.height,
        "master_keys_unique": master.unique("contract").height == master.height,
        "all_option_contracts_in_master": joined.filter(pl.col("call_put").is_null()).is_empty(),
        "active_prices_nonnegative": options.filter(
            (pl.col("volume") > 0)
            & ((pl.col("open") < 0) | (pl.col("close") < 0) | (pl.col("settle") < 0))
        ).is_empty(),
        "option_date_coverage_at_least_98pct": (
            len(option_dates.intersection(dates)) / len(dates) >= 0.98 if dates else False
        ),
        "standard_contract_terms_valid": standard.filter(
            (pl.col("exercise_price") <= 0)
            | (pl.col("min_price_chg") <= 0)
            | pl.col("maturity_date").is_null()
        ).is_empty(),
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "returns_evaluated": False,
        "strategy_metrics_computed": False,
        "counts": {
            "master_contracts": master.height,
            "standard_contracts": standard.height,
            "maturities": maturity_counts.height,
            "underlying_days": fund.height,
            "option_rows": options.height,
            "option_days": len(option_dates),
        },
        "checks": checks,
    }


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    root = data_dir / "research" / "50etf_option_vrp"
    try:
        master = normalize_master(
            client.query(
                "opt_basic", {"exchange": "SSE", "opt_code": OPTION_CODE}, MASTER_FIELDS
            )
        )
        fund = normalize_fund(
            client.query(
                "fund_daily",
                {
                    "ts_code": UNDERLYING,
                    "start_date": START.strftime("%Y%m%d"),
                    "end_date": END.strftime("%Y%m%d"),
                },
                FUND_FIELDS,
            )
        )
        _atomic_parquet(master, root / "contracts.parquet")
        _atomic_parquet(fund, root / "underlying.parquet")
        allowed = set(master["contract"].to_list())
        partitions = root / "daily"
        for index, trade_date in enumerate(fund["date"].to_list(), start=1):
            target = partitions / f"date={trade_date.isoformat()}" / "part.parquet"
            if not target.exists():
                frame = normalize_options(
                    client.query(
                        "opt_daily",
                        {"exchange": "SSE", "trade_date": trade_date.strftime("%Y%m%d")},
                        OPTION_FIELDS,
                    ),
                    allowed,
                )
                _atomic_parquet(frame, target)
            if index == 1 or index % 50 == 0 or index == fund.height:
                print(f"option_daily_progress={index}/{fund.height}", flush=True)
    finally:
        client.close()
    files = sorted((root / "daily").glob("date=*/part.parquet"))
    options = pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")
    payload = {
        "schema_version": "p0-50etf-option-data-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": START, "end": END},
        **audit(master, fund, options),
        "artifacts": {
            "contracts": str(root / "contracts.parquet"),
            "underlying": str(root / "underlying.parquet"),
            "daily_root": str(root / "daily"),
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
        default=Path("/app/data/research/p0_50etf_option_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
