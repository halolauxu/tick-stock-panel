"""Run frozen QDII ETF morning-overshoot reversal on reused development data."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

import run_p0_qdii_etf_intraday_momentum_development as engine  # noqa: E402

MIN_MEDIAN_RETURN = -0.001
MAX_LAGGARD_RETURN = -0.005
MIN_LAG = 0.002


def build_signals(minutes: pl.DataFrame) -> list[dict[str, Any]]:
    frame = minutes.with_columns(pl.col("datetime").dt.date().alias("date"))
    signals = []
    for trade_date in sorted(frame["date"].unique().to_list()):
        day = frame.filter(pl.col("date") == trade_date).drop("date")
        if day.height != len(engine.SYMBOLS) * engine.EXPECTED_BARS:
            continue
        rows = engine._day_rows(day)
        opening = rows.get(engine.time(9, 30), {})
        signal_bars = rows.get(engine.SIGNAL_TIME, {})
        if set(opening) != set(engine.SYMBOLS) or set(signal_bars) != set(engine.SYMBOLS):
            continue
        cumulative = (
            day.filter(pl.col("datetime").dt.time() <= engine.SIGNAL_TIME)
            .group_by("symbol")
            .agg(pl.col("amount").sum().alias("amount"))
        )
        amounts = dict(
            zip(cumulative["symbol"].to_list(), cumulative["amount"].to_list(), strict=True)
        )
        eligible = [
            symbol
            for symbol in engine.SYMBOLS
            if float(amounts.get(symbol, 0.0)) >= engine.MIN_CUMULATIVE_AMOUNT
        ]
        if len(eligible) < engine.MIN_ELIGIBLE_FUNDS:
            continue
        returns = {
            symbol: float(signal_bars[symbol]["close"]) / float(opening[symbol]["open"]) - 1.0
            for symbol in eligible
            if float(opening[symbol]["open"]) > 0
        }
        if len(returns) < engine.MIN_ELIGIBLE_FUNDS:
            continue
        median_return = statistics.median(returns.values())
        laggard = min(returns, key=lambda symbol: (returns[symbol], symbol))
        laggard_return = returns[laggard]
        lag = median_return - laggard_return
        if (
            median_return > MIN_MEDIAN_RETURN
            or laggard_return > MAX_LAGGARD_RETURN
            or lag < MIN_LAG
        ):
            continue
        signals.append(
            {
                "date": trade_date,
                "symbol": laggard,
                "eligible_funds": len(eligible),
                "median_return": median_return,
                "laggard_return": laggard_return,
                "lag": lag,
                "signal_close": float(signal_bars[laggard]["close"]),
                # Execution ledger field aliases; economics remain laggard/lag.
                "winner_return": laggard_return,
                "winner_lead": lag,
            }
        )
    return signals


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit = json.loads(
        (data_dir / "research" / "p0_qdii_etf_minute_data_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "DATA_QUALIFIED":
        raise RuntimeError("QDII ETF minute data has not passed the frozen audit")
    root = data_dir / "research" / "qdii_etf_intraday_momentum" / "phases" / "development"
    minutes = pl.read_parquet(str(root / "symbol=*" / "part.parquet"))
    if minutes.filter(pl.col("datetime").dt.date() > engine.DEV_END).height:
        raise RuntimeError("sealed validation or pressure rows entered development input")
    signals = build_signals(minutes)
    accounts = {
        str(int(capital)): engine.simulate_account(minutes, signals, capital)
        for capital in engine.INITIAL_CAPITALS
    }
    primary = accounts[str(int(engine.INITIAL_CAPITALS[0]))]
    gate = engine.evaluate_gate(primary)
    payload = {
        "schema_version": "p0-qdii-etf-overshoot-reversal-development-v1",
        "contract_frozen": "2026-08-31",
        "development_data_previously_used": True,
        "period": {"start": engine.DEV_START, "end": engine.DEV_END},
        "validation_metrics_computed": False,
        "pressure_metrics_computed": False,
        "signal_count": len(signals),
        "gate": gate,
        "decision": "CONTINUE_TO_VALIDATION" if all(gate.values()) else "TERMINATE_DEVELOPMENT",
        "accounts": accounts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    compact = {
        **{key: value for key, value in payload.items() if key != "accounts"},
        "accounts": {
            key: {field: value[field] for field in value if field != "records"}
            for key, value in accounts.items()
        },
        "output": str(output),
        "sha256": digest,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=_json_default))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_qdii_etf_overshoot_reversal_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
