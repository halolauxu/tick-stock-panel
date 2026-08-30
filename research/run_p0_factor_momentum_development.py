"""Run the frozen development-only factor-momentum account study."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_academic_factor_development_screen as academic  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_state_aware_return_screen as state  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
TARGET_POSITIONS = 10
TRAILING_MONTHS = 6
MIN_COMPLETED_CONSTITUENTS = 5
ESTIMATED_ROUND_TRIP_COST = 0.0025
SLEEVE_IDS = (*academic.FACTOR_IDS, "revised_momentum", "microcap")


def build_microcap_candidates(monthly: pl.DataFrame) -> pl.DataFrame:
    return (
        monthly.filter(
            (pl.col("listing_days") >= state.MIN_LISTING_DAYS)
            & (pl.col("mean_amount_20d") >= state.MIN_AMOUNT_MICROCAP)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("market_cap_rank"),
        )
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(
            pl.col("market_cap_rank")
            <= (pl.col("universe_count") * 0.10).ceil()
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            (-pl.col("market_cap")).alias("factor_value"),
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def completed_sleeve_returns(
    candidates_by_sleeve: dict[str, pl.DataFrame],
    panel: pl.DataFrame,
    signal_dates: list[date],
) -> list[dict[str, Any]]:
    calendar = pl.DataFrame({"date": signal_dates}).sort("date").with_columns(
        pl.col("date").shift(-1).alias("completion_date")
    )
    entry_quotes = panel.select(
        "symbol",
        pl.col("date").alias("entry_date"),
        pl.col("open").alias("entry_open"),
        pl.col("raw_open").alias("entry_raw_open"),
        pl.col("limit_up_price").alias("entry_limit_up"),
        pl.col("volume").alias("entry_volume"),
        pl.col("amount").alias("entry_amount"),
    )
    mark_history = panel.select(
        "symbol",
        pl.col("date").alias("mark_date"),
        pl.col("close").alias("mark_close"),
    ).sort(["symbol", "mark_date"])
    rows: list[dict[str, Any]] = []
    for sleeve_id, candidates in candidates_by_sleeve.items():
        realized = (
            candidates.join(calendar, on="date", how="left")
            .drop_nulls("completion_date")
            .join(entry_quotes, on=["symbol", "entry_date"], how="left")
            .sort(["symbol", "completion_date"])
            .join_asof(
                mark_history,
                left_on="completion_date",
                right_on="mark_date",
                by="symbol",
                strategy="backward",
                check_sortedness=False,
            )
            .filter(
                (pl.col("entry_volume") > 0)
                & (pl.col("entry_amount") > 0)
                & (pl.col("entry_raw_open") < pl.col("entry_limit_up") - 0.005)
                & (pl.col("entry_open") > 0)
                & (pl.col("mark_close") > 0)
            )
            .with_columns(
                (
                    pl.col("mark_close") / pl.col("entry_open")
                    - 1.0
                    - ESTIMATED_ROUND_TRIP_COST
                ).alias("sleeve_return")
            )
            .group_by("completion_date")
            .agg(
                pl.col("sleeve_return").mean().alias("sleeve_return"),
                pl.len().alias("constituents"),
            )
            .filter(pl.col("constituents") >= MIN_COMPLETED_CONSTITUENTS)
            .sort("completion_date")
        )
        rows.extend(
            {"sleeve_id": sleeve_id, **row} for row in realized.to_dicts()
        )
    return sorted(rows, key=lambda row: (row["completion_date"], row["sleeve_id"]))


def select_factor_by_trailing_returns(
    completed_returns: list[dict[str, Any]], signal_dates: list[date]
) -> dict[date, dict[str, Any]]:
    history: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for row in completed_returns:
        history[row["sleeve_id"]].append(
            (row["completion_date"], float(row["sleeve_return"]))
        )
    selections: dict[date, dict[str, Any]] = {}
    for signal_date in signal_dates:
        ranked: list[tuple[float, str, list[float]]] = []
        for sleeve_id in SLEEVE_IDS:
            available = [
                value
                for completion, value in history.get(sleeve_id, [])
                if completion <= signal_date
            ][-TRAILING_MONTHS:]
            if len(available) != TRAILING_MONTHS:
                continue
            compounded = math.prod(1.0 + value for value in available) - 1.0
            ranked.append((compounded, sleeve_id, available))
        if not ranked:
            continue
        compounded, sleeve_id, values = max(
            ranked, key=lambda row: (row[0], row[1])
        )
        if compounded <= 0:
            continue
        selections[signal_date] = {
            "sleeve_id": sleeve_id,
            "trailing_compounded_return": compounded,
            "trailing_monthly_returns": values,
        }
    return selections


def select_candidates(
    candidates_by_sleeve: dict[str, pl.DataFrame],
    selections: dict[date, dict[str, Any]],
) -> pl.DataFrame:
    frames = []
    for signal_date, selection in selections.items():
        frame = candidates_by_sleeve[selection["sleeve_id"]].filter(
            pl.col("date") == signal_date
        )
        if frame.height:
            frames.append(frame)
    if not frames:
        raise ValueError("factor momentum produced no candidates")
    return pl.concat(frames, how="diagonal_relaxed").sort(
        ["entry_date", "cap_rank", "symbol"]
    )


def evaluate_gate(
    result: dict[str, Any], benchmark: dict[str, Any], active_fraction: float
) -> dict[str, Any]:
    decision = shared.evaluate_gate(result, benchmark)
    decision["checks"].pop("mean_cash_ratio_at_most_25pct")
    decision["checks"]["active_rebalance_fraction_at_least_50pct"] = (
        active_fraction >= 0.50
    )
    decision["passed"] = all(decision["checks"].values())
    decision["verdict"] = (
        "PROMOTE_TO_VALIDATION" if decision["passed"] else "TERMINATE"
    )
    return decision


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    raw_source = raw_all.filter(pl.col("date") >= DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = academic.attach_price_features(baseline.prepare_panel(pit))
    panel = state.attach_return_features(panel)
    panel = panel.join(state.build_market_state(panel), on="date", how="left")
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= DEVELOPMENT_START)
    )
    monthly, action_dates = academic.monthly_signal_panel(panel)
    monthly = academic.attach_annual_factors(
        monthly, academic.load_annual_factors(data_dir)
    )
    candidates_by_sleeve = {
        factor_id: academic.build_candidates(monthly, factor_id)
        for factor_id in academic.FACTOR_IDS
    }
    revised = state.build_candidates(monthly, "revised_momentum_monthly")
    candidates_by_sleeve["revised_momentum"] = revised.rename(
        {"signal_score": "factor_value"}
    )
    candidates_by_sleeve["microcap"] = build_microcap_candidates(monthly)
    signal_dates = monthly.get_column("date").unique().sort().to_list()
    completed = completed_sleeve_returns(
        candidates_by_sleeve, panel, signal_dates
    )
    selections = select_factor_by_trailing_returns(completed, signal_dates)
    candidates = select_candidates(candidates_by_sleeve, selections)
    del panel, monthly, candidates_by_sleeve
    gc.collect()
    result = academic.simulate_factor(
        candidates, raw_source, all_dates, action_dates, data_dir
    )
    active_days = candidates.get_column("entry_date").n_unique()
    active_fraction = active_days / len(action_dates)
    decision = evaluate_gate(result, benchmark, active_fraction)
    payload = {
        "schema_version": "p0-factor-momentum-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "configuration": {
            "initial_cash_cny": shared.INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "trailing_completed_months": TRAILING_MONTHS,
            "sleeves": SLEEVE_IDS,
            "estimated_sleeve_round_trip_cost": ESTIMATED_ROUND_TRIP_COST,
        },
        "data": {
            "completed_sleeve_months": len(completed),
            "scheduled_rebalance_days": len(action_dates),
            "active_rebalance_days": active_days,
            "active_rebalance_fraction": active_fraction,
            "selected_sleeve_counts": dict(
                sorted(Counter(row["sleeve_id"] for row in selections.values()).items())
            ),
            "selection_trace": [
                {"signal_date": signal_date, **selection}
                for signal_date, selection in sorted(selections.items())
            ],
        },
        "benchmark": benchmark,
        "strategy": result,
        "decision": decision,
        "promoted_to_independent_validation": decision["passed"],
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "data": payload["data"],
                "benchmark": benchmark,
                "metrics": result["metrics"],
                "execution": result["execution"],
                "integrity": result["integrity"],
                "account": result["account"],
                "decision": decision,
                "output": str(output),
                "sha256": digest,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_factor_momentum_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
