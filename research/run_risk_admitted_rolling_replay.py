"""Replay the frozen risk-admitted portfolio through a newer complete trading day.

This writes a separate rolling artifact.  It never mutates the immutable research
artifact whose digest admitted the forward account.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_microcap_idiosyncratic_forecast_unified_account as unified  # noqa: E402
import run_p0_risk_gated_idiosyncratic_forecast_overlay as gated  # noqa: E402

SCHEMA_VERSION = "risk-admitted-idiosyncratic-forecast-rolling-v1"
ROLLING_START = date(2024, 1, 1)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(
    data_dir: Path,
    *,
    end: date,
    contract_sha256: str,
    thresholds_path: Path,
    output: Path,
) -> dict[str, Any]:
    if end < ROLLING_START:
        raise ValueError("rolling replay end precedes its fixed start")
    admission_by_open, risk_audit = gated.build_event_gate(
        data_dir,
        ROLLING_START,
        end,
        thresholds_path,
    )
    result = unified.run_period(
        data_dir,
        ROLLING_START,
        end,
        event_admission_by_date=admission_by_open,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "generated_at": datetime.now().astimezone().isoformat(),
        "risk": risk_audit,
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "start": ROLLING_START.isoformat(),
                "end": end.isoformat(),
                "trading_days": result["metrics"]["trading_days"],
                "orders": len(result.get("orders") or []),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=RESEARCH / "p0_microcap_escape_thresholds.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.data_dir,
        end=args.end,
        contract_sha256=args.contract_sha256,
        thresholds_path=args.thresholds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
