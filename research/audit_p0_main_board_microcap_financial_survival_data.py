"""Audit PIT annual-statement coverage for the micro-cap survival filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

START_REPORT_YEAR = 2013
END_REPORT_YEAR = 2025
MAX_MISSING_RATE = 0.05
MIN_SYMBOLS_PER_YEAR = 1_000
MAIN_BOARD_PATTERN = (
    r"^(?:(?:000|001|002|003)\d{3}\.SZ|"
    r"(?:600|601|603|605)\d{3}\.SH)$"
)
FORBIDDEN_FIELDS = {"open", "high", "low", "close", "return", "future_return"}


def load_annual_statement(
    data_dir: Path,
    dataset: str,
    columns: tuple[str, ...],
    announce_alias: str,
) -> pl.DataFrame:
    path = data_dir / "financials" / dataset / "part.parquet"
    frame = pl.read_parquet(path)
    needed = {"symbol", "period_end", "announce_date", *columns}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset} missing columns: {sorted(missing)}")
    return (
        frame.select("symbol", "period_end", "announce_date", *columns)
        .with_columns(
            pl.col("period_end").cast(pl.Date, strict=False),
            pl.col("announce_date").cast(pl.Date, strict=False).alias(announce_alias),
        )
        .drop("announce_date")
        .filter(
            pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
            & (pl.col("period_end").dt.month() == 12)
            & pl.col("period_end")
            .dt.year()
            .is_between(START_REPORT_YEAR, END_REPORT_YEAR, closed="both")
            & (pl.col(announce_alias) > pl.col("period_end"))
        )
        .sort(["symbol", "period_end", announce_alias])
        .unique(["symbol", "period_end"], keep="first")
    )


def build_snapshots(data_dir: Path) -> pl.DataFrame:
    income = load_annual_statement(
        data_dir,
        "income",
        ("net_income_attributable",),
        "income_announce_date",
    )
    cash = load_annual_statement(
        data_dir,
        "cash_flow",
        ("net_operating_cash_flow",),
        "cash_announce_date",
    )
    balance = load_annual_statement(
        data_dir,
        "balance_sheet",
        ("total_assets", "total_liabilities", "total_equity", "goodwill"),
        "balance_announce_date",
    )
    return (
        income.join(cash, on=["symbol", "period_end"], how="inner")
        .join(balance, on=["symbol", "period_end"], how="inner")
        .with_columns(
            pl.max_horizontal(
                "income_announce_date",
                "cash_announce_date",
                "balance_announce_date",
            ).alias("financial_available_date")
        )
        .with_columns(
            (pl.col("total_liabilities") / pl.col("total_assets")).alias("debt_ratio"),
            (pl.col("goodwill").fill_null(0.0) / pl.col("total_assets")).alias(
                "goodwill_ratio"
            ),
        )
        .sort(["symbol", "financial_available_date", "period_end"])
    )


def audit(snapshots: pl.DataFrame) -> dict[str, Any]:
    critical = (
        "net_income_attributable",
        "net_operating_cash_flow",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "debt_ratio",
        "goodwill_ratio",
    )
    total_cells = snapshots.height * len(critical)
    null_cells = sum(snapshots[column].null_count() for column in critical)
    yearly = (
        snapshots.with_columns(pl.col("period_end").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("symbol").n_unique().alias("symbols"), pl.len().alias("rows"))
        .sort("year")
    )
    yearly_map = {row["year"]: row["symbols"] for row in yearly.to_dicts()}
    checks = {
        "price_and_return_fields_absent": not (
            FORBIDDEN_FIELDS & set(snapshots.columns)
        ),
        "unique_symbol_period": snapshots.unique(["symbol", "period_end"]).height
        == snapshots.height,
        "availability_after_period_end": snapshots.filter(
            pl.col("financial_available_date") <= pl.col("period_end")
        ).height
        == 0,
        "critical_missing_rate_at_most_5pct": (
            null_cells / total_cells if total_cells else 1.0
        )
        <= MAX_MISSING_RATE,
        "every_year_has_at_least_1000_symbols": all(
            yearly_map.get(year, 0) >= MIN_SYMBOLS_PER_YEAR
            for year in range(START_REPORT_YEAR, END_REPORT_YEAR + 1)
        ),
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "price_data_read": False,
        "future_returns_read": False,
        "rows": snapshots.height,
        "symbols": snapshots["symbol"].n_unique(),
        "period": {
            "first_report": snapshots["period_end"].min(),
            "last_report": snapshots["period_end"].max(),
            "last_available_date": snapshots["financial_available_date"].max(),
        },
        "critical_missing_rate": null_cells / total_cells if total_cells else None,
        "yearly_coverage": yearly.to_dicts(),
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    snapshots = build_snapshots(data_dir)
    snapshot_path = (
        data_dir
        / "research"
        / "main_board_microcap_financial_survival"
        / "annual_snapshots.parquet"
    )
    _atomic_parquet(snapshots, snapshot_path)
    payload = {
        "schema_version": "p0-main-board-microcap-financial-survival-data-v1",
        "contract_frozen": "2026-09-03",
        **audit(snapshots),
        "artifact": str(snapshot_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **payload,
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
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
            "/app/data/research/p0_main_board_microcap_financial_survival_data.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
