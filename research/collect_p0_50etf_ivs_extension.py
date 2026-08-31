"""Extend the audited 50ETF option dataset through 2024 without reading returns."""

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
sys.path.insert(0, str(ROOT / "research"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402
from collect_p0_50etf_option_data import (  # noqa: E402
    FUND_FIELDS,
    MASTER_FIELDS,
    OPTION_CODE,
    OPTION_FIELDS,
    UNDERLYING,
    _atomic_parquet,
    normalize_fund,
    normalize_master,
    normalize_options,
)

START = date(2015, 2, 9)
END = date(2024, 12, 31)
SHIBOR_FIELDS = ("date", "on", "1w", "2w", "1m", "3m", "6m", "9m", "1y")


def normalize_shibor(
    rows: list[dict[str, Any]], *, start: date = START, end: date = END
) -> pl.DataFrame:
    schema = {
        "date": pl.Date,
        "on": pl.Float64,
        "1w": pl.Float64,
        "2w": pl.Float64,
        "1m": pl.Float64,
        "3m": pl.Float64,
        "6m": pl.Float64,
        "9m": pl.Float64,
        "1y": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in schema
                if column != "date"
            ],
        )
        .filter(pl.col("date").is_between(start, end, closed="both"))
        .select(*schema)
        .unique("date", keep="last")
        .sort("date")
    )


def _year_chunks(client: TushareClient, api_name: str, fields: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for year in range(START.year, END.year + 1):
        lower = max(START, date(year, 1, 1))
        upper = min(END, date(year, 12, 31))
        params = {
            "start_date": lower.strftime("%Y%m%d"),
            "end_date": upper.strftime("%Y%m%d"),
        }
        if api_name == "fund_daily":
            params["ts_code"] = UNDERLYING
        rows.extend(client.query(api_name, params, fields))
    return rows


def audit_extension(
    master: pl.DataFrame,
    fund: pl.DataFrame,
    options: pl.DataFrame,
    shibor: pl.DataFrame,
) -> dict[str, Any]:
    option_dates = set(options["date"].to_list())
    fund_dates = set(fund["date"].to_list())
    shibor_dates = set(shibor["date"].to_list())
    standard = master.filter(
        (pl.col("exercise_price") > 0)
        & (pl.col("opt_multiplier") > 0)
        & (pl.col("min_price_chg") > 0)
    )
    checks = {
        "underlying_covers_2015_through_2024": (
            fund.height >= 2_350
            and fund["date"].min() == START
            and fund["date"].max() == END
        ),
        "option_date_coverage_at_least_98pct": (
            len(option_dates & fund_dates) / len(fund_dates) >= 0.98 if fund_dates else False
        ),
        "shibor_has_at_least_2200_days": len(shibor_dates) >= 2_200,
        "option_keys_unique": options.unique(["contract", "date"]).height == options.height,
        "master_keys_unique": master.unique("contract").height == master.height,
        "contract_terms_positive": standard.height == master.height,
        "active_prices_nonnegative": options.filter(
            (pl.col("volume") > 0)
            & ((pl.col("open") < 0) | (pl.col("close") < 0) | (pl.col("settle") < 0))
        ).is_empty(),
    }
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "returns_evaluated": False,
        "strategy_metrics_computed": False,
        "counts": {
            "master_contracts": master.height,
            "underlying_days": fund.height,
            "option_rows": options.height,
            "option_days": len(option_dates),
            "shibor_days": len(shibor_dates),
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
            ),
            start=START,
            end=END,
        )
        fund = normalize_fund(
            _year_chunks(client, "fund_daily", FUND_FIELDS), start=START, end=END
        )
        shibor = normalize_shibor(_year_chunks(client, "shibor", SHIBOR_FIELDS))
        _atomic_parquet(master, root / "contracts.parquet")
        _atomic_parquet(fund, root / "underlying.parquet")
        _atomic_parquet(shibor, root / "shibor.parquet")

        allowed = set(master["contract"].to_list())
        partitions = root / "daily"
        dates = fund["date"].to_list()
        for index, trade_date in enumerate(dates, start=1):
            target = partitions / f"date={trade_date.isoformat()}" / "part.parquet"
            if not target.exists():
                frame = normalize_options(
                    client.query(
                        "opt_daily",
                        {"exchange": "SSE", "trade_date": trade_date.strftime("%Y%m%d")},
                        OPTION_FIELDS,
                    ),
                    allowed,
                    start=START,
                    end=END,
                )
                _atomic_parquet(frame, target)
            if index == 1 or index % 50 == 0 or index == len(dates):
                print(f"option_daily_progress={index}/{len(dates)}", flush=True)
    finally:
        client.close()

    files = sorted((root / "daily").glob("date=*/part.parquet"))
    options = pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")
    payload = {
        "schema_version": "p0-50etf-ivs-extension-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": START, "end": END},
        "risk_free_proxy": "SHIBOR nearest tenor, causal as-of join",
        **audit_extension(master, fund, options, shibor),
        "artifacts": {
            "contracts": str(root / "contracts.parquet"),
            "underlying": str(root / "underlying.parquet"),
            "shibor": str(root / "shibor.parquet"),
            "daily_root": str(root / "daily"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({**payload, "sha256": digest}, ensure_ascii=False, indent=2, default=str))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_50etf_ivs_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
