"""Probe point-in-time corporate and trading event data availability."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402


class ProbeSpec(NamedTuple):
    api_name: str
    fields: tuple[str, ...]
    date_field: str
    key_fields: tuple[str, ...]
    critical_fields: tuple[str, ...]
    requests: tuple[dict[str, str], ...]
    row_limit: int


PERIOD_REQUESTS = (
    {"start_date": "20190101", "end_date": "20190331"},
    {"start_date": "20240101", "end_date": "20240331"},
    {"start_date": "20260101", "end_date": "20260331"},
)
TRADE_DATE_REQUESTS = (
    {"trade_date": "20190131"},
    {"trade_date": "20240131"},
    {"trade_date": "20260130"},
)
ANN_DATE_REQUESTS = (
    {"ann_date": "20190131"},
    {"ann_date": "20240131"},
    {"ann_date": "20260130"},
)

SPECS = (
    ProbeSpec(
        api_name="forecast",
        fields=(
            "ts_code",
            "ann_date",
            "end_date",
            "type",
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            "last_parent_net",
            "first_ann_date",
        ),
        date_field="ann_date",
        key_fields=("ts_code", "ann_date", "end_date", "type"),
        critical_fields=("ts_code", "ann_date", "end_date", "type"),
        requests=ANN_DATE_REQUESTS,
        row_limit=3500,
    ),
    ProbeSpec(
        api_name="repurchase",
        fields=(
            "ts_code",
            "ann_date",
            "end_date",
            "proc",
            "exp_date",
            "vol",
            "amount",
            "high_limit",
            "low_limit",
        ),
        date_field="ann_date",
        key_fields=("ts_code", "ann_date", "proc", "end_date"),
        critical_fields=("ts_code", "ann_date", "proc"),
        requests=PERIOD_REQUESTS,
        row_limit=2000,
    ),
    ProbeSpec(
        api_name="stk_holdertrade",
        fields=(
            "ts_code",
            "ann_date",
            "holder_name",
            "holder_type",
            "in_de",
            "change_vol",
            "change_ratio",
            "after_share",
            "after_ratio",
            "avg_price",
            "total_share",
            "begin_date",
            "close_date",
        ),
        date_field="ann_date",
        key_fields=("ts_code", "ann_date", "holder_name", "in_de", "begin_date"),
        critical_fields=("ts_code", "ann_date", "holder_type", "in_de"),
        requests=PERIOD_REQUESTS,
        row_limit=3000,
    ),
    ProbeSpec(
        api_name="top_list",
        fields=(
            "trade_date",
            "ts_code",
            "name",
            "close",
            "pct_change",
            "turnover_rate",
            "amount",
            "l_sell",
            "l_buy",
            "l_amount",
            "net_amount",
            "net_rate",
            "amount_rate",
            "float_values",
            "reason",
        ),
        date_field="trade_date",
        key_fields=("trade_date", "ts_code", "reason"),
        critical_fields=("trade_date", "ts_code", "net_amount", "reason"),
        requests=TRADE_DATE_REQUESTS,
        row_limit=10000,
    ),
    ProbeSpec(
        api_name="block_trade",
        fields=(
            "ts_code",
            "trade_date",
            "price",
            "vol",
            "amount",
            "buyer",
            "seller",
        ),
        date_field="trade_date",
        key_fields=("ts_code", "trade_date", "price", "vol", "buyer", "seller"),
        critical_fields=("ts_code", "trade_date", "price", "amount"),
        requests=TRADE_DATE_REQUESTS,
        row_limit=1000,
    ),
    ProbeSpec(
        api_name="share_float",
        fields=(
            "ts_code",
            "ann_date",
            "float_date",
            "float_share",
            "float_ratio",
            "holder_name",
            "share_type",
        ),
        date_field="ann_date",
        key_fields=("ts_code", "ann_date", "float_date", "holder_name"),
        critical_fields=("ts_code", "ann_date", "float_date", "float_ratio"),
        requests=PERIOD_REQUESTS,
        row_limit=6000,
    ),
)


def _clean_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message[:500]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and not math.isfinite(value)
    )


def summarize_rows(spec: ProbeSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    symbols = {str(row.get("ts_code")) for row in rows if row.get("ts_code")}
    dates = sorted(
        str(row.get(spec.date_field))
        for row in rows
        if row.get(spec.date_field)
    )
    keys = {
        tuple(str(row.get(field)) for field in spec.key_fields) for row in rows
    }
    null_rates = {
        field: (
            sum(_missing(row.get(field)) for row in rows) / len(rows) if rows else None
        )
        for field in spec.critical_fields
    }
    return {
        "rows": len(rows),
        "symbols": len(symbols),
        "first_event_date": dates[0] if dates else None,
        "last_event_date": dates[-1] if dates else None,
        "duplicate_key_rows": len(rows) - len(keys),
        "critical_null_rates": null_rates,
        "hit_row_limit": len(rows) >= spec.row_limit,
    }


def probe_spec(client: Any, spec: ProbeSpec) -> dict[str, Any]:
    samples = []
    successful_requests = 0
    nonempty_requests = 0
    for params in spec.requests:
        try:
            rows = client.query(spec.api_name, params, spec.fields)
            summary = summarize_rows(spec, rows)
            successful_requests += 1
            nonempty_requests += int(bool(rows))
            samples.append({"params": params, "status": "ok", **summary})
        except Exception as error:
            samples.append(
                {
                    "params": params,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": _clean_error(error),
                }
            )
    critical_complete = nonempty_requests > 0 and all(
        sample.get("status") == "ok"
        and all(
            rate == 0.0
            for rate in sample.get("critical_null_rates", {}).values()
            if rate is not None
        )
        for sample in samples
        if sample.get("rows", 0) > 0
    )
    critical_usable = nonempty_requests > 0 and all(
        all(
            rate is None or rate <= 0.01
            for rate in sample.get("critical_null_rates", {}).values()
        )
        for sample in samples
        if sample.get("rows", 0) > 0
    )
    return {
        "api_name": spec.api_name,
        "successful_requests": successful_requests,
        "nonempty_requests": nonempty_requests,
        "available": successful_requests == len(spec.requests),
        "cross_period_ready": nonempty_requests >= 2,
        "critical_fields_complete_in_samples": critical_complete,
        "critical_fields_usable_in_samples": critical_usable,
        "samples": samples,
    }


def run(output: Path) -> dict[str, Any]:
    token = get_api_key()
    if not token:
        raise ValueError("configured Tushare token is unavailable")
    client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    try:
        probes = [probe_spec(client, spec) for spec in SPECS]
    finally:
        client.close()
    ready = [
        probe["api_name"]
        for probe in probes
        if probe["available"]
        and probe["cross_period_ready"]
        and probe["critical_fields_usable_in_samples"]
    ]
    payload = {
        "schema_version": "p0-corporate-event-data-audit-v1",
        "probes": probes,
        "decision": {
            "ready_for_bounded_collection": ready,
            "counts_toward_50pct_goal": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_corporate_event_data_audit.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
