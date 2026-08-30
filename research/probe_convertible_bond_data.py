"""Probe configured convertible-bond data capabilities without exposing secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import (  # noqa: E402
    MINUTE_FIELDS,
    TushareClient,
    TushareError,
)
from app.plugins.tushare.provider import get_api_key  # noqa: E402

CB_BASIC_FIELDS = (
    "ts_code",
    "bond_short_name",
    "cb_type",
    "stk_code",
    "issue_size",
    "remain_size",
    "list_date",
    "delist_date",
    "exchange",
    "conv_start_date",
    "conv_end_date",
    "first_conv_price",
    "conv_price",
    "issue_rating",
    "newest_rating",
)

CB_DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "pct_chg",
    "vol",
    "amount",
    "bond_value",
    "bond_over_rate",
    "cb_value",
    "cb_over_rate",
)

CB_PRICE_CHANGE_FIELDS = (
    "ts_code",
    "bond_short_name",
    "publish_date",
    "change_date",
    "convert_price_initial",
    "convertprice_bef",
    "convertprice_aft",
)

CB_CALL_FIELDS = (
    "ts_code",
    "call_type",
    "is_call",
    "ann_date",
    "call_date",
    "call_price",
    "call_reg_date",
)

PROBE_DATES = (
    "20150105",
    "20180102",
    "20200102",
    "20210104",
    "20220104",
    "20230103",
    "20240102",
    "20250102",
    "20260828",
)


def _missing_rates(rows: list[dict], fields: tuple[str, ...]) -> dict[str, float]:
    if not rows:
        return {field: 1.0 for field in fields}
    return {
        field: sum(row.get(field) in (None, "") for row in rows) / len(rows)
        for field in fields
    }


def summarize_keyed_rows(
    rows: list[dict],
    *,
    key: tuple[str, ...],
    critical_fields: tuple[str, ...],
) -> dict[str, Any]:
    keys = [tuple(row.get(column) for column in key) for row in rows]
    return {
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "missing_rates": _missing_rates(rows, critical_fields),
    }


def safe_probe(call: Callable[[], list[dict]]) -> dict[str, Any]:
    try:
        rows = call()
        return {"status": "AVAILABLE", "rows": rows}
    except TushareError as exc:
        return {"status": "UNAVAILABLE", "error": str(exc), "rows": []}
    except Exception as exc:  # keep the audit running across independent probes
        return {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "rows": [],
        }


def run(output: Path) -> dict[str, Any]:
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    try:
        basic_probe = safe_probe(
            lambda: client.query("cb_basic", {}, CB_BASIC_FIELDS)
        )
        basic_rows = basic_probe.pop("rows")
        basic = {
            **basic_probe,
            **summarize_keyed_rows(
                basic_rows,
                key=("ts_code",),
                critical_fields=(
                    "stk_code",
                    "issue_size",
                    "remain_size",
                    "list_date",
                    "delist_date",
                ),
            ),
            "first_list_date": min(
                (row["list_date"] for row in basic_rows if row.get("list_date")),
                default=None,
            ),
            "last_list_date": max(
                (row["list_date"] for row in basic_rows if row.get("list_date")),
                default=None,
            ),
            "active_as_of_20260828": sum(
                bool(row.get("list_date"))
                and row["list_date"].replace("-", "") <= "20260828"
                and (
                    not row.get("delist_date")
                    or row["delist_date"].replace("-", "") > "20260828"
                )
                for row in basic_rows
            ),
        }

        daily_probes: list[dict[str, Any]] = []
        latest_rows: list[dict] = []
        for trade_date in PROBE_DATES:
            probe = safe_probe(
                lambda trade_date=trade_date: client.query(
                    "cb_daily", {"trade_date": trade_date}, CB_DAILY_FIELDS
                )
            )
            rows = probe.pop("rows")
            if trade_date == PROBE_DATES[-1]:
                latest_rows = rows
            daily_probes.append(
                {
                    "trade_date": trade_date,
                    **probe,
                    **summarize_keyed_rows(
                        rows,
                        key=("ts_code", "trade_date"),
                        critical_fields=(
                            "open",
                            "high",
                            "low",
                            "close",
                            "vol",
                            "amount",
                            "bond_value",
                            "cb_value",
                            "cb_over_rate",
                        ),
                    ),
                }
            )

        sample_code = (
            sorted(
                (row for row in latest_rows if row.get("ts_code")),
                key=lambda row: float(row.get("amount") or 0.0),
                reverse=True,
            )[0]["ts_code"]
            if latest_rows
            else None
        )
        sample_daily_row = next(
            (row for row in latest_rows if row.get("ts_code") == sample_code),
            None,
        )
        minute_probe = (
            safe_probe(
                lambda: client.stock_minutes(
                    sample_code,
                    freq="1min",
                    start_time=datetime(2026, 8, 28, 9, 25),
                    end_time=datetime(2026, 8, 28, 15, 5),
                )
            )
            if sample_code
            else {
                "status": "NOT_TESTED",
                "error": "latest cb_daily sample is empty",
                "rows": [],
            }
        )
        minute_rows = minute_probe.pop("rows")
        minute = {
            **minute_probe,
            "sample_code": sample_code,
            **summarize_keyed_rows(
                minute_rows,
                key=("ts_code", "trade_time"),
                critical_fields=MINUTE_FIELDS,
            ),
            "first_time": min(
                (row["trade_time"] for row in minute_rows if row.get("trade_time")),
                default=None,
            ),
            "last_time": max(
                (row["trade_time"] for row in minute_rows if row.get("trade_time")),
                default=None,
            ),
            "unit_reconciliation": {
                "daily_volume_hands": (
                    sample_daily_row.get("vol") if sample_daily_row else None
                ),
                "minute_volume_raw_sum": sum(
                    float(row.get("vol") or 0.0) for row in minute_rows
                ),
                "minute_to_daily_volume_ratio": (
                    sum(float(row.get("vol") or 0.0) for row in minute_rows)
                    / float(sample_daily_row.get("vol") or 0.0)
                    if minute_rows
                    and sample_daily_row
                    and sample_daily_row.get("vol")
                    else None
                ),
                "daily_amount_cny": (
                    float(sample_daily_row.get("amount") or 0.0) * 10_000.0
                    if sample_daily_row
                    else None
                ),
                "minute_amount_raw_sum": sum(
                    float(row.get("amount") or 0.0) for row in minute_rows
                ),
                "minute_to_daily_amount_ratio": (
                    sum(float(row.get("amount") or 0.0) for row in minute_rows)
                    / (float(sample_daily_row.get("amount") or 0.0) * 10_000.0)
                    if minute_rows
                    and sample_daily_row
                    and sample_daily_row.get("amount")
                    else None
                ),
            },
        }

        price_probe = (
            safe_probe(
                lambda: client.query(
                    "cb_price_chg",
                    {"ts_code": sample_code},
                    CB_PRICE_CHANGE_FIELDS,
                )
            )
            if sample_code
            else {"status": "NOT_TESTED", "rows": []}
        )
        price_rows = price_probe.pop("rows")
        price_changes = {
            **price_probe,
            "sample_code": sample_code,
            **summarize_keyed_rows(
                price_rows,
                key=("ts_code", "publish_date", "change_date"),
                critical_fields=(
                    "publish_date",
                    "change_date",
                    "convertprice_bef",
                    "convertprice_aft",
                ),
            ),
        }

        call_probe = safe_probe(
            lambda: client.query("cb_call", {}, CB_CALL_FIELDS)
        )
        call_rows = call_probe.pop("rows")
        calls = {
            **call_probe,
            **summarize_keyed_rows(
                call_rows,
                key=("ts_code", "ann_date", "call_date", "is_call"),
                critical_fields=("ann_date", "call_date", "is_call"),
            ),
        }
    finally:
        client.close()

    local_data_dirs = []
    data_root = Path("/app/data")
    if data_root.is_dir():
        for path in data_root.iterdir():
            if any(token in path.name.lower() for token in ("bond", "convert", "cb", "kzz")):
                local_data_dirs.append(str(path))

    payload = {
        "schema_version": "p0-convertible-bond-data-audit-v1",
        "audit_date": date.today(),
        "secrets": "configured token used but never serialized",
        "local_storage": {
            "matching_top_level_paths": sorted(local_data_dirs),
            "has_persisted_convertible_bond_dataset": bool(local_data_dirs),
        },
        "remote": {
            "cb_basic": basic,
            "cb_daily_by_date": daily_probes,
            "cb_price_chg": price_changes,
            "cb_call": calls,
            "cb_minute_via_existing_stk_mins": minute,
        },
        "limitations": {
            "pro_bar": "SDK-only integrated endpoint; current project uses the HTTP client",
            "minute_conclusion": (
                "AVAILABLE only if the live cb-code stk_mins probe returned a complete session"
            ),
            "persistence": "remote availability does not mean data is stored locally",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "local_storage": payload["local_storage"],
        "cb_basic": basic,
        "cb_daily_by_date": daily_probes,
        "cb_price_chg": price_changes,
        "cb_call": calls,
        "cb_minute_via_existing_stk_mins": minute,
        "output": str(output),
        "sha256": sha256,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_convertible_bond_data_audit.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
