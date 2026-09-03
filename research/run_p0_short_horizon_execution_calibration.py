"""Run the frozen R4-01 opening execution calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

SCHEMA_VERSION = "p0-short-horizon-execution-calibration-v1"
START = date(2025, 8, 27)
END = date(2026, 8, 28)
INPUTS = {
    "bare_microcap": (
        "p0_main_board_microcap_account_v1.json",
        "80276b1e187f29b3896e376d92ed57bfe1da838be8526c108aad3e8a99db950d",
    ),
    "forecast_overlay_v1": (
        "p0_risk_admitted_idiosyncratic_forecast_overlay_v1.json",
        "6c70333c3c07543a9240a86ae3166fd75f4afaf13a418167e2ef394e89964145",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date_paths(root: Path) -> list[Path]:
    paths = []
    for path in root.glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if START <= value <= END:
            paths.append(path)
    return sorted(paths)


def _nested(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for key in path:
        value = value[key]
    return value


def load_orders(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = {
        "bare_microcap": ("accounts", "200000", "periods", "known_stress", "orders"),
        "forecast_overlay_v1": ("results", "known_stress", "orders"),
    }
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for account, (name, expected_hash) in INPUTS.items():
        path = data_dir / "research" / name
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"input hash mismatch: {name}")
        document = json.loads(path.read_text(encoding="utf-8"))
        source_orders = _nested(document, paths[account])
        accepted = 0
        for index, order in enumerate(source_orders):
            if order.get("status") != "FILLED":
                continue
            order_date = date.fromisoformat(order["date"])
            if not START <= order_date <= END:
                continue
            shares = int(order.get("raw_shares") or 0)
            gross = float(order.get("gross") or 0)
            if shares <= 0 or gross <= 0:
                continue
            rows.append(
                {
                    "account": account,
                    "order_index": index,
                    "date": order_date,
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "raw_shares": shares,
                    "gross": gross,
                }
            )
            accepted += 1
        audits[account] = {
            "path": str(path),
            "sha256": actual_hash,
            "source_order_rows": len(source_orders),
            "overlap_filled_orders": accepted,
        }
    return pl.DataFrame(rows).sort(["account", "date", "order_index"]), audits


def load_auction(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = _date_paths(data_dir / "tushare_supplemental" / "auction")
    if not paths:
        raise ValueError("opening-auction partitions are required")
    frame = (
        pl.read_parquet(paths, hive_partitioning=False)
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter((pl.col("session") == "open") & pl.col("date").is_between(START, END))
        .select(
            "symbol",
            "date",
            pl.col("open").alias("auction_open"),
            pl.col("amount").alias("auction_amount"),
        )
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["date", "symbol"])
    )
    return frame, {
        "partition_count": len(paths),
        "rows": frame.height,
        "trading_days": frame.get_column("date").n_unique(),
        "start": frame.get_column("date").min(),
        "end": frame.get_column("date").max(),
    }


def load_minute_0931(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = _date_paths(data_dir / "kline_minute")
    if not paths:
        raise ValueError("minute partitions are required")
    frame = (
        pl.scan_parquet(paths, hive_partitioning=False)
        .filter((pl.col("datetime").dt.hour() == 9) & (pl.col("datetime").dt.minute() == 31))
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .select(
            "symbol",
            "date",
            pl.col("open").alias("minute_0931_open"),
            pl.col("amount").alias("minute_0931_amount"),
        )
        .unique(subset=["symbol", "date"], keep="last")
        .collect(engine="streaming")
        .sort(["date", "symbol"])
    )
    return frame, {
        "partition_count": len(paths),
        "rows": frame.height,
        "trading_days": frame.get_column("date").n_unique(),
        "start": frame.get_column("date").min(),
        "end": frame.get_column("date").max(),
    }


def match_orders(orders: pl.DataFrame, auction: pl.DataFrame, minute: pl.DataFrame) -> pl.DataFrame:
    side_sign = pl.when(pl.col("side") == "BUY").then(1.0).otherwise(-1.0)
    return (
        orders.join(auction, on=["symbol", "date"], how="left")
        .join(minute, on=["symbol", "date"], how="left")
        .with_columns((pl.col("gross") / pl.col("raw_shares")).alias("daily_fill_price"))
        .with_columns(
            ((pl.col("auction_open") / pl.col("daily_fill_price") - 1.0).abs() * 10_000).alias(
                "auction_abs_bps"
            ),
            (
                (pl.col("auction_open") / pl.col("daily_fill_price") - 1.0) * side_sign * 10_000
            ).alias("auction_adverse_bps"),
            ((pl.col("minute_0931_open") / pl.col("daily_fill_price") - 1.0).abs() * 10_000).alias(
                "minute_0931_abs_bps"
            ),
            (
                (pl.col("minute_0931_open") / pl.col("daily_fill_price") - 1.0) * side_sign * 10_000
            ).alias("minute_0931_adverse_bps"),
            (pl.col("gross") <= pl.col("auction_amount") * 0.01).alias("auction_capacity_ok"),
            (pl.col("gross") <= pl.col("minute_0931_amount") * 0.05).alias(
                "minute_0931_capacity_ok"
            ),
        )
        .sort(["account", "date", "order_index"])
    )


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_group(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.height

    def present(column: str) -> pl.DataFrame:
        return frame.filter(pl.col(column).is_not_null())

    auction = present("auction_open")
    minute = present("minute_0931_open")

    def metric(data: pl.DataFrame, column: str, kind: str) -> Any:
        if data.is_empty():
            return None
        expression = {
            "median": pl.col(column).median(),
            "p95": pl.col(column).quantile(0.95, interpolation="nearest"),
            "max": pl.col(column).max(),
            "mean": pl.col(column).mean(),
        }[kind]
        return _finite(data.select(expression).item())

    return {
        "orders": rows,
        "trading_days": frame.get_column("date").n_unique(),
        "symbols": frame.get_column("symbol").n_unique(),
        "auction_coverage": auction.height / rows if rows else None,
        "minute_0931_coverage": minute.height / rows if rows else None,
        "auction_abs_bps_median": metric(auction, "auction_abs_bps", "median"),
        "auction_abs_bps_p95": metric(auction, "auction_abs_bps", "p95"),
        "auction_abs_bps_max": metric(auction, "auction_abs_bps", "max"),
        "auction_adverse_bps_mean": metric(auction, "auction_adverse_bps", "mean"),
        "minute_0931_abs_bps_median": metric(minute, "minute_0931_abs_bps", "median"),
        "minute_0931_abs_bps_p95": metric(minute, "minute_0931_abs_bps", "p95"),
        "minute_0931_abs_bps_max": metric(minute, "minute_0931_abs_bps", "max"),
        "minute_0931_adverse_bps_mean": metric(minute, "minute_0931_adverse_bps", "mean"),
        "auction_capacity_pass_rate": (
            auction.get_column("auction_capacity_ok").sum() / auction.height
            if auction.height
            else None
        ),
        "minute_0931_capacity_pass_rate": (
            minute.get_column("minute_0931_capacity_ok").sum() / minute.height
            if minute.height
            else None
        ),
    }


def summarize(matched: pl.DataFrame) -> dict[str, Any]:
    accounts: dict[str, Any] = {}
    for account in matched.get_column("account").unique().sort().to_list():
        subset = matched.filter(pl.col("account") == account)
        accounts[account] = {
            "all": summarize_group(subset),
            "buy": summarize_group(subset.filter(pl.col("side") == "BUY")),
            "sell": summarize_group(subset.filter(pl.col("side") == "SELL")),
        }
    return {"accounts": accounts, "combined": summarize_group(matched)}


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    account_all = [value["all"] for value in summary["accounts"].values()]
    sufficient = all(
        row["auction_coverage"] >= 0.95 and row["minute_0931_coverage"] >= 0.95
        for row in account_all
    )
    price_valid = sufficient and all(
        row["auction_coverage"] >= 0.99
        and row["auction_abs_bps_median"] <= 1.0
        and row["auction_abs_bps_p95"] <= 5.0
        for row in account_all
    )
    capacity_failures = {
        account: {
            "auction_failed_orders": round(
                values["all"]["orders"]
                * values["all"]["auction_coverage"]
                * (1.0 - values["all"]["auction_capacity_pass_rate"])
            ),
            "minute_0931_failed_orders": round(
                values["all"]["orders"]
                * values["all"]["minute_0931_coverage"]
                * (1.0 - values["all"]["minute_0931_capacity_pass_rate"])
            ),
        }
        for account, values in summary["accounts"].items()
    }
    return {
        "verdict": "CALIBRATION_COMPLETE" if sufficient else "DATA_INSUFFICIENT",
        "daily_open_price": (
            "DAILY_OPEN_PRICE_VALIDATED" if price_valid else "DAILY_OPEN_PRICE_NOT_VALIDATED"
        ),
        "capacity_failures": capacity_failures,
        "alpha_admission_changed": False,
        "r4_02_dependency_satisfied": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    orders, order_audit = load_orders(data_dir)
    auction, auction_audit = load_auction(data_dir)
    minute, minute_audit = load_minute_0931(data_dir)
    matched = match_orders(orders, auction, minute)
    summary = summarize(matched)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_frozen": "2026-09-04",
        "contract_commits": ["cd49c44", "d4970aa"],
        "period": {"start": START, "end": END},
        "inputs": {
            "orders": order_audit,
            "auction": auction_audit,
            "minute_0931": minute_audit,
        },
        "summary": summary,
        "matched_orders": matched.to_dicts(),
        "decision": decide(summary),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "period": payload["period"],
                "inputs": payload["inputs"],
                "summary": summary,
                "decision": payload["decision"],
                "output": str(output),
                "sha256": _sha256(output),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
