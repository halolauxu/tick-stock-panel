"""Collect and audit point-in-time futures price limits and margin rates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

import collect_p0_cn_commodity_futures_data as v2  # noqa: E402

LIMIT_FIELDS = (
    "trade_date",
    "ts_code",
    "name",
    "up_limit",
    "down_limit",
    "m_ratio",
    "cont",
    "exchange",
)


def normalize_limits(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "trade_date": "date",
                "ts_code": "contract",
                "m_ratio": "margin_rate_pct",
            }
        )
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("contract").cast(pl.Utf8).str.strip_chars(),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in ("up_limit", "down_limit", "margin_rate_pct")
            ],
        )
        .filter(pl.col("date").is_between(v2.START, v2.END, closed="both"))
        .unique(subset=["contract", "date"], keep="last")
        .sort(["contract", "date"])
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "cn_commodity_futures"
    daily = pl.read_parquet(root / "contract_daily.parquet")
    mapped_contracts = set(daily["contract"].unique().to_list())
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    rows: list[dict[str, Any]] = []
    try:
        for series_index, series in enumerate(v2.SERIES, start=1):
            product = series.split(".", 1)[0]
            for year in range(v2.START.year, v2.END.year + 1):
                rows.extend(
                    client.query(
                        "ft_limit",
                        {
                            "cont": product,
                            "start_date": f"{year}0101",
                            "end_date": f"{year}1231",
                        },
                        LIMIT_FIELDS,
                    )
                )
            print(
                f"futures_limit_progress={series_index}/{len(v2.SERIES)} "
                f"series={series} rows={len(rows)}",
                flush=True,
            )
    finally:
        client.close()

    limits = normalize_limits(rows).filter(
        pl.col("contract").is_in(mapped_contracts)
    )
    active = daily.filter(pl.col("volume") > 0)
    joined = active.join(
        limits.select(
            "contract", "date", "up_limit", "down_limit", "margin_rate_pct"
        ),
        on=["contract", "date"],
        how="left",
    )
    covered = joined.filter(pl.col("up_limit").is_not_null()).height
    coverage = covered / active.height
    invalid_limits = limits.filter(
        pl.col("up_limit").is_null()
        | pl.col("down_limit").is_null()
        | pl.col("margin_rate_pct").is_null()
        | (pl.col("up_limit") <= pl.col("down_limit"))
        | (pl.col("down_limit") <= 0)
        | (pl.col("margin_rate_pct") <= 0)
    ).height
    outside_limits = joined.filter(
        pl.col("up_limit").is_not_null()
        & (
            (pl.col("open") > pl.col("up_limit") * 1.000001)
            | (pl.col("open") < pl.col("down_limit") * 0.999999)
        )
    ).height
    checks = {
        "limits_unique": limits.unique(["contract", "date"]).height
        == limits.height,
        "active_quote_coverage_at_least_99_5pct": coverage >= 0.995,
        "limit_and_margin_values_valid": invalid_limits == 0,
        "active_opens_within_published_limits": outside_limits == 0,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    _atomic_parquet(limits, root / "contract_limits.parquet")
    payload = {
        "schema_version": "p0-cn-commodity-futures-limit-data-v4",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": v2.START,
            "end": v2.END,
            "validation_returns_read": False,
        },
        "status": status,
        "counts": {
            "limit_rows": limits.height,
            "limit_contracts": limits["contract"].n_unique(),
            "active_contract_daily_rows": active.height,
            "covered_active_rows": covered,
            "coverage": coverage,
            "invalid_limit_rows": invalid_limits,
            "opens_outside_limits": outside_limits,
        },
        "checks": checks,
        "artifact": str(root / "contract_limits.parquet"),
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
            "/app/data/research/p0_cn_commodity_futures_limit_data_v4_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
