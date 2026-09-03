"""Audit holding duration and re-entry behavior in frozen account results.

This is a read-only diagnostic.  It reconstructs position lifecycles from the
orders already embedded in frozen result JSON files; it does not read prices,
create candidates, run a backtest, or alter either source result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "p0-short-horizon-baseline-audit-v1"
MICROCAP_FAMILY = "main_board_microcap"


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_lifecycles(
    orders: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
    trading_dates: list[date],
    *,
    default_family: str = "unknown",
) -> dict[str, Any]:
    """Pair filled buys with sells or settlements on the account clock."""

    normalized_dates = sorted({_as_date(day) for day in trading_dates})
    date_index = {day: index for index, day in enumerate(normalized_dates)}
    events: list[dict[str, Any]] = []
    for index, order in enumerate(orders):
        if order.get("status") != "FILLED" or order.get("side") not in {
            "BUY",
            "SELL",
        }:
            continue
        events.append(
            {
                "kind": order["side"],
                "date": _as_date(order["date"]),
                "sequence": index,
                "payload": order,
            }
        )
    settlement_offset = len(events)
    for index, settlement in enumerate(settlements):
        events.append(
            {
                "kind": "SETTLEMENT",
                "date": _as_date(settlement["date"]),
                "sequence": settlement_offset + index,
                "payload": settlement,
            }
        )

    # The simulator sells before buying on the same session.  Preserve that
    # ordering so same-day replacement cannot look like overlapping positions.
    priority = {"SELL": 0, "SETTLEMENT": 1, "BUY": 2}
    events.sort(key=lambda row: (row["date"], priority[row["kind"]], row["sequence"]))

    open_positions: dict[str, dict[str, Any]] = {}
    last_exit: dict[str, date] = {}
    cycles: list[dict[str, Any]] = []
    issues: list[str] = []

    for event in events:
        event_date = event["date"]
        payload = event["payload"]
        symbol = str(payload["symbol"])
        if event_date not in date_index:
            issues.append(f"event_date_not_in_account_clock:{symbol}:{event_date}")
            continue

        if event["kind"] == "BUY":
            if symbol in open_positions:
                issues.append(f"buy_while_position_open:{symbol}:{event_date}")
                continue
            previous_exit = last_exit.get(symbol)
            open_positions[symbol] = {
                "symbol": symbol,
                "family": payload.get("family") or default_family,
                "entry_date": event_date,
                "buy_cash_delta": float(payload.get("cash_delta") or 0.0),
                "reentry_gap_sessions": (
                    date_index[event_date] - date_index[previous_exit]
                    if previous_exit in date_index
                    else None
                ),
            }
            continue

        position = open_positions.pop(symbol, None)
        if position is None:
            action = "sell" if event["kind"] == "SELL" else "settlement"
            issues.append(f"{action}_without_open_position:{symbol}:{event_date}")
            continue

        if event["kind"] == "SELL":
            exit_cash_delta = float(payload.get("cash_delta") or 0.0)
            exit_type = "SELL"
        else:
            exit_cash_delta = float(payload.get("recovery_value") or 0.0)
            exit_type = str(payload.get("status") or "SETTLEMENT")
        holding_sessions = (
            date_index[event_date] - date_index[position["entry_date"]] + 1
        )
        cycles.append(
            {
                "symbol": symbol,
                "family": position["family"],
                "entry_date": position["entry_date"].isoformat(),
                "exit_date": event_date.isoformat(),
                "exit_type": exit_type,
                "closed": True,
                "holding_sessions": holding_sessions,
                "observed_sessions": holding_sessions,
                "reentry_gap_sessions": position["reentry_gap_sessions"],
                "cash_pnl": position["buy_cash_delta"] + exit_cash_delta,
            }
        )
        last_exit[symbol] = event_date

    final_date = normalized_dates[-1] if normalized_dates else None
    for symbol, position in sorted(open_positions.items()):
        observed_sessions = (
            date_index[final_date] - date_index[position["entry_date"]] + 1
            if final_date is not None
            else None
        )
        cycles.append(
            {
                "symbol": symbol,
                "family": position["family"],
                "entry_date": position["entry_date"].isoformat(),
                "exit_date": None,
                "exit_type": None,
                "closed": False,
                "holding_sessions": None,
                "observed_sessions": observed_sessions,
                "reentry_gap_sessions": position["reentry_gap_sessions"],
                "cash_pnl": None,
            }
        )
    cycles.sort(key=lambda row: (row["entry_date"], row["symbol"]))
    return {"cycles": cycles, "issues": issues}


def summarize_lifecycles(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in cycles if row["closed"]]
    opened = [row for row in cycles if not row["closed"]]
    hold_counts = Counter()
    for row in closed:
        sessions = int(row["holding_sessions"])
        if sessions < 2:
            hold_counts["under_2"] += 1
        elif sessions <= 5:
            hold_counts["2_to_5"] += 1
        elif sessions <= 10:
            hold_counts["6_to_10"] += 1
        else:
            hold_counts["over_10"] += 1
    pnl_rows = [row for row in closed if row.get("cash_pnl") is not None]
    reentries = [
        int(row["reentry_gap_sessions"])
        for row in cycles
        if row.get("reentry_gap_sessions") is not None
    ]
    over_10 = [row for row in closed if int(row["holding_sessions"]) > 10]
    return {
        "cycles": len(cycles),
        "closed_cycles": len(closed),
        "open_cycles": len(opened),
        "open_over_10_cycles": sum(
            int(row.get("observed_sessions") or 0) > 10 for row in opened
        ),
        "cash_pnl": sum(float(row["cash_pnl"]) for row in pnl_rows),
        "winning_cycles": sum(float(row["cash_pnl"]) > 0 for row in pnl_rows),
        "win_rate": (
            sum(float(row["cash_pnl"]) > 0 for row in pnl_rows) / len(pnl_rows)
            if pnl_rows
            else None
        ),
        "holding_buckets": {
            key: hold_counts.get(key, 0)
            for key in ("under_2", "2_to_5", "6_to_10", "over_10")
        },
        "max_holding_sessions": max(
            (int(row["holding_sessions"]) for row in closed), default=None
        ),
        "over_10_cycles": len(over_10),
        "over_10_loss_cycles": sum(
            float(row.get("cash_pnl") or 0.0) < 0 for row in over_10
        ),
        "over_10_cash_pnl": sum(
            float(row.get("cash_pnl") or 0.0) for row in over_10
        ),
        "reentries": len(reentries),
        "reentries_within_10_sessions": sum(gap <= 10 for gap in reentries),
        "next_session_reentries": sum(gap == 1 for gap in reentries),
    }


def build_cycle_audit(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in cycles})
    entry_years = sorted(
        {str(_as_date(row["entry_date"]).year) for row in cycles}
    )
    exit_years = sorted(
        {
            str(_as_date(row["exit_date"]).year)
            for row in cycles
            if row.get("exit_date")
        }
    )
    return {
        "all": summarize_lifecycles(cycles),
        "by_family": {
            family: summarize_lifecycles(
                [row for row in cycles if row["family"] == family]
            )
            for family in families
        },
        "by_exit_year": {
            year: summarize_lifecycles(
                [
                    row
                    for row in cycles
                    if row.get("exit_date")
                    and str(_as_date(row["exit_date"]).year) == year
                ]
            )
            for year in exit_years
        },
        "by_entry_year": {
            year: summarize_lifecycles(
                [
                    row
                    for row in cycles
                    if str(_as_date(row["entry_date"]).year) == year
                ]
            )
            for year in entry_years
        },
        "by_family_exit_year": {
            family: {
                year: summarize_lifecycles(
                    [
                        row
                        for row in cycles
                        if row["family"] == family
                        and row.get("exit_date")
                        and str(_as_date(row["exit_date"]).year) == year
                    ]
                )
                for year in exit_years
                if any(
                    row["family"] == family
                    and row.get("exit_date")
                    and str(_as_date(row["exit_date"]).year) == year
                    for row in cycles
                )
            }
            for family in families
        },
        "by_family_entry_year": {
            family: {
                year: summarize_lifecycles(
                    [
                        row
                        for row in cycles
                        if row["family"] == family
                        and str(_as_date(row["entry_date"]).year) == year
                    ]
                )
                for year in entry_years
                if any(
                    row["family"] == family
                    and str(_as_date(row["entry_date"]).year) == year
                    for row in cycles
                )
            }
            for family in families
        },
    }


def _rejection_summary(orders: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side in ("BUY", "SELL"):
        scoped = [row for row in orders if row.get("side") == side]
        output[side.lower()] = {
            "orders": len(scoped),
            "filled": sum(row.get("status") == "FILLED" for row in scoped),
            "reasons": dict(
                sorted(
                    Counter(
                        str(row.get("reason"))
                        for row in scoped
                        if row.get("status") != "FILLED"
                    ).items()
                )
            ),
        }
    return output


def audit_period(
    payload: dict[str, Any], *, default_family: str = "unknown"
) -> dict[str, Any]:
    trading_dates = [_as_date(row["date"]) for row in payload["daily_equity"]]
    reconstructed = reconstruct_lifecycles(
        payload.get("orders", []),
        payload.get("settlements", []),
        trading_dates,
        default_family=default_family,
    )
    return {
        "trading_days": len(trading_dates),
        "cycle_audit": build_cycle_audit(reconstructed["cycles"]),
        "order_audit": _rejection_summary(payload.get("orders", [])),
        "issues": reconstructed["issues"],
        "cycles": reconstructed["cycles"],
    }


def run(core_result: Path, overlay_result: Path, output: Path) -> dict[str, Any]:
    core_payload = json.loads(core_result.read_text(encoding="utf-8"))
    overlay_payload = json.loads(overlay_result.read_text(encoding="utf-8"))
    core_periods = core_payload["accounts"]["200000"]["periods"]
    overlay_periods = overlay_payload["results"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_existing_result_diagnostic",
        "sources": {
            "core": {"sha256": _sha256(core_result)},
            "overlay": {"sha256": _sha256(overlay_result)},
        },
        "core": {
            period: audit_period(row, default_family=MICROCAP_FAMILY)
            for period, row in core_periods.items()
        },
        "overlay": {
            period: audit_period(row)
            for period, row in overlay_periods.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "sources": payload["sources"],
                "core": {
                    period: row["cycle_audit"]
                    for period, row in payload["core"].items()
                },
                "overlay": {
                    period: row["cycle_audit"]
                    for period, row in payload["overlay"].items()
                },
                "output": str(output),
                "sha256": _sha256(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-result", type=Path, required=True)
    parser.add_argument("--overlay-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.core_result, args.overlay_result, args.output)


if __name__ == "__main__":
    main()
