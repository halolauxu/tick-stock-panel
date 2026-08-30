"""Probe high-turnover A-share market-microstructure data availability."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from research.probe_corporate_event_data import (  # noqa: E402
    ProbeSpec,
    probe_spec,
)

from app.plugins.tushare.client import TushareClient  # noqa: E402
from app.plugins.tushare.provider import get_api_key  # noqa: E402

FLOW_DATES = (
    {"trade_date": "20190131"},
    {"trade_date": "20240131"},
    {"trade_date": "20260130"},
)
AUCTION_DATES = (
    {"trade_date": "20240930"},
    {"trade_date": "20250829"},
    {"trade_date": "20260828"},
)

SPECS = (
    ProbeSpec(
        api_name="moneyflow",
        fields=(
            "ts_code",
            "trade_date",
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ),
        date_field="trade_date",
        key_fields=("ts_code", "trade_date"),
        critical_fields=("ts_code", "trade_date", "net_mf_amount"),
        requests=FLOW_DATES,
        row_limit=6000,
    ),
    ProbeSpec(
        api_name="margin_detail",
        fields=(
            "trade_date",
            "ts_code",
            "rzye",
            "rqye",
            "rzmre",
            "rqyl",
            "rzche",
            "rqchl",
            "rqmcl",
            "rzrqye",
        ),
        date_field="trade_date",
        key_fields=("ts_code", "trade_date"),
        critical_fields=("ts_code", "trade_date", "rzye", "rzmre"),
        requests=FLOW_DATES,
        row_limit=6000,
    ),
    ProbeSpec(
        api_name="stk_auction_o",
        fields=(
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "vol",
            "amount",
            "vwap",
        ),
        date_field="trade_date",
        key_fields=("ts_code", "trade_date"),
        critical_fields=(
            "ts_code",
            "trade_date",
            "close",
            "open",
            "vol",
            "amount",
        ),
        requests=AUCTION_DATES,
        row_limit=10000,
    ),
)


def run(output: Path, client: Any | None = None) -> dict[str, Any]:
    owned_client = client is None
    if client is None:
        token = get_api_key()
        if not token:
            raise ValueError("configured Tushare token is unavailable")
        client = TushareClient(token, timeout=30.0, min_interval_s=0.35)
    try:
        probes = [probe_spec(client, spec) for spec in SPECS]
    finally:
        if owned_client:
            client.close()
    ready = [
        probe["api_name"]
        for probe in probes
        if probe["available"]
        and probe["cross_period_ready"]
        and probe["critical_fields_usable_in_samples"]
    ]
    payload = {
        "schema_version": "p0-market-microstructure-data-audit-v1",
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
        default=Path(
            "/app/data/research/p0_market_microstructure_data_audit.json"
        ),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
