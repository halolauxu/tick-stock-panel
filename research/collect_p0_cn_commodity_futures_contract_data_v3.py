"""Collect exact mapped futures contracts for executable roll simulation."""
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


def normalize_contract_daily(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = v2.normalize_daily(rows)
    if normalized.is_empty():
        return normalized
    return normalized.rename({"series": "contract"})


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "cn_commodity_futures"
    master = pl.read_parquet(root / "contracts.parquet")
    mapping = pl.read_parquet(root / "main_mapping.parquet")
    contracts = sorted(set(mapping["contract"].drop_nulls().to_list()))
    master_by_contract = {
        row["contract"]: row for row in master.to_dicts()
    }
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        rows: list[dict[str, Any]] = []
        for index, contract in enumerate(contracts, start=1):
            metadata = master_by_contract[contract]
            start = max(v2.START, metadata["list_date"])
            end = min(v2.END, metadata.get("delist_date") or v2.END)
            rows.extend(
                client.query(
                    "fut_daily",
                    {
                        "ts_code": contract,
                        "start_date": start.strftime("%Y%m%d"),
                        "end_date": end.strftime("%Y%m%d"),
                    },
                    v2.DAILY_FIELDS,
                )
            )
            if index == 1 or index % 50 == 0 or index == len(contracts):
                print(
                    f"contract_daily_progress={index}/{len(contracts)} "
                    f"rows={len(rows)}",
                    flush=True,
                )
        daily = normalize_contract_daily(rows)
    finally:
        client.close()

    mapped = mapping.join(daily, on=["contract", "date"], how="left")
    mapped_quote_rows = mapped.filter(pl.col("settle").is_not_null()).height
    mapping_coverage = mapped_quote_rows / mapping.height
    rolls = (
        mapping.sort(["series", "date"])
        .with_columns(
            pl.col("contract").shift(1).over("series").alias("previous_contract")
        )
        .filter(
            pl.col("previous_contract").is_not_null()
            & (pl.col("contract") != pl.col("previous_contract"))
        )
    )
    previous_quotes = daily.select(
        pl.col("contract").alias("previous_contract"),
        "date",
        pl.col("settle").alias("previous_settle"),
    )
    current_quotes = daily.select(
        "contract",
        "date",
        pl.col("open").alias("current_open"),
    )
    roll_quotes = rolls.join(
        previous_quotes, on=["previous_contract", "date"], how="left"
    ).join(current_quotes, on=["contract", "date"], how="left")
    executable_rolls = roll_quotes.filter(
        pl.col("previous_settle").is_not_null()
        & pl.col("current_open").is_not_null()
    ).height
    roll_coverage = executable_rolls / rolls.height if rolls.height else 0.0
    active = daily.filter(pl.col("volume") > 0)
    invalid_active = active.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("settle") <= 0)
    ).height
    negative_rows = daily.filter(
        (pl.col("volume") < 0) | (pl.col("oi") < 0)
    ).height
    checks = {
        "contract_daily_unique": daily.unique(["contract", "date"]).height
        == daily.height,
        "mapped_quote_coverage_at_least_99_9pct": mapping_coverage >= 0.999,
        "roll_quote_coverage_at_least_99pct": roll_coverage >= 0.99,
        "active_prices_valid": invalid_active == 0,
        "volume_and_open_interest_nonnegative": negative_rows == 0,
    }
    status = "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP"
    _atomic_parquet(daily, root / "contract_daily.parquet")
    payload = {
        "schema_version": "p0-cn-commodity-futures-contract-data-v3",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": v2.START,
            "end": v2.END,
            "validation_returns_read": False,
        },
        "status": status,
        "counts": {
            "requested_contracts": len(contracts),
            "contract_daily_rows": daily.height,
            "contract_daily_symbols": daily["contract"].n_unique(),
            "mapping_rows": mapping.height,
            "mapped_quote_rows": mapped_quote_rows,
            "mapping_coverage": mapping_coverage,
            "roll_events": rolls.height,
            "executable_roll_events": executable_rolls,
            "roll_coverage": roll_coverage,
            "invalid_active_rows": invalid_active,
            "negative_market_rows": negative_rows,
        },
        "checks": checks,
        "artifact": str(root / "contract_daily.parquet"),
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
            "/app/data/research/p0_cn_commodity_futures_contract_data_v3_audit.json"
        ),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
