"""Screen all supported non-financial daily factors for a short-horizon ceiling."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

from app.backtest.factor import FACTOR_COLUMNS, FactorBacktestService  # noqa: E402
from app.backtest.fundamentals import FUNDAMENTAL_FACTOR_NAMES  # noqa: E402

import run_p0_deepseek_main_board_short_horizon_screen as shared  # noqa: E402

DISCOVERY_START = date(2014, 1, 1)
DISCOVERY_END = date(2017, 12, 31)
CONFIRMATION_START = date(2018, 1, 1)
CONFIRMATION_END = date(2020, 12, 31)
HORIZONS = (2, 5, 10)
PORTFOLIO_SIZE = 10
ROUND_TRIP_COST = 0.0024
BATCH_SIZE = 12
EXCLUDED_SCALE_FACTORS = frozenset({"macd_hist", "atr_14"})
FACTOR_NAMES = tuple(
    row["id"]
    for row in FACTOR_COLUMNS
    if row["id"] not in FUNDAMENTAL_FACTOR_NAMES
    and row["id"] not in EXCLUDED_SCALE_FACTORS
)


def attach_targets(
    panel: pl.DataFrame, trading_dates: list[date]
) -> pl.DataFrame:
    work = panel
    for horizon in HORIZONS:
        calendar = pl.DataFrame(
            {
                "date": trading_dates[: -(horizon + 1)],
                f"entry_date_{horizon}": trading_dates[1:-horizon],
                f"exit_date_{horizon}": trading_dates[horizon + 1 :],
            }
        )
        entry = panel.select(
            "symbol",
            pl.col("date").alias(f"entry_date_{horizon}"),
            pl.col("open").alias(f"entry_open_{horizon}"),
        )
        exit_ = panel.select(
            "symbol",
            pl.col("date").alias(f"exit_date_{horizon}"),
            pl.col("open").alias(f"exit_open_{horizon}"),
        )
        work = (
            work.join(calendar, on="date", how="left")
            .join(entry, on=["symbol", f"entry_date_{horizon}"], how="left")
            .join(exit_, on=["symbol", f"exit_date_{horizon}"], how="left")
            .with_columns(
                (
                    pl.col(f"exit_open_{horizon}")
                    / pl.col(f"entry_open_{horizon}")
                    - 1.0
                    - ROUND_TRIP_COST
                ).alias(f"net_return_{horizon}")
            )
        )
    return work


def signal_dates(
    trading_dates: list[date], start: date, end: date, horizon: int
) -> list[date]:
    scoped = [value for value in trading_dates if start <= value <= end]
    return scoped[::horizon]


def leg_returns(
    panel: pl.DataFrame,
    factor: str,
    horizon: int,
    direction: str,
    dates: list[date],
    end: date,
) -> pl.DataFrame:
    descending = direction == "HIGH"
    target = f"net_return_{horizon}"
    exit_date = f"exit_date_{horizon}"
    return (
        panel.filter(
            pl.col("date").is_in(dates)
            & (pl.col(exit_date) <= end)
            & pl.col(factor).is_finite()
            & pl.col(target).is_finite()
        )
        .sort(
            ["date", factor, "symbol"],
            descending=[False, descending, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("rank"))
        .filter(pl.col("rank") <= PORTFOLIO_SIZE)
        .group_by("date")
        .agg(
            pl.col(target).mean().alias("return"),
            pl.len().alias("positions"),
        )
        .filter(pl.col("positions") == PORTFOLIO_SIZE)
        .sort("date")
    )


def benchmark_returns(
    panel: pl.DataFrame,
    horizon: int,
    dates: list[date],
    end: date,
) -> pl.DataFrame:
    target = f"net_return_{horizon}"
    exit_date = f"exit_date_{horizon}"
    return (
        panel.filter(
            pl.col("date").is_in(dates)
            & (pl.col(exit_date) <= end)
            & pl.col(target).is_finite()
        )
        .group_by("date")
        .agg(pl.col(target).mean().alias("return"), pl.len().alias("positions"))
        .filter(pl.col("positions") >= 100)
        .sort("date")
    )


def metrics(frame: pl.DataFrame, horizon: int) -> dict[str, Any]:
    values = [float(value) for value in frame.get_column("return").to_list()]
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    yearly: dict[int, float] = {}
    for row_date, value in frame.select("date", "return").iter_rows():
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
        yearly[row_date.year] = yearly.get(row_date.year, 1.0) * (1.0 + value)
    annualized = (
        equity ** (252.0 / (len(values) * horizon)) - 1.0
        if values and equity > 0
        else None
    )
    return {
        "rebalances": len(values),
        "annualized": annualized,
        "total_return": equity - 1.0 if values else None,
        "max_drawdown": drawdown if values else None,
        "positive_years": sum(value > 1.0 for value in yearly.values()),
        "yearly": {str(year): value - 1.0 for year, value in sorted(yearly.items())},
    }


def evaluate_period(
    leg: dict[str, Any], benchmark: dict[str, Any], positive_years: int
) -> dict[str, bool]:
    annualized = float(leg.get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    return {
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": float(leg.get("max_drawdown") or -math.inf)
        >= -0.30,
        "positive_years_sufficient": int(leg.get("positive_years") or 0)
        >= positive_years,
        "at_least_100_rebalances": int(leg.get("rebalances") or 0) >= 100,
    }


def screen_factor(
    panel: pl.DataFrame,
    factor: str,
    horizon: int,
    discovery_dates: list[date],
    confirmation_dates: list[date],
    benchmarks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    discovery_legs = {
        direction: metrics(
            leg_returns(
                panel,
                factor,
                horizon,
                direction,
                discovery_dates,
                DISCOVERY_END,
            ),
            horizon,
        )
        for direction in ("LOW", "HIGH")
    }
    direction = max(
        discovery_legs,
        key=lambda name: discovery_legs[name].get("annualized") or -math.inf,
    )
    discovery = discovery_legs[direction]
    confirmation = metrics(
        leg_returns(
            panel,
            factor,
            horizon,
            direction,
            confirmation_dates,
            CONFIRMATION_END,
        ),
        horizon,
    )
    checks = {
        "discovery": evaluate_period(discovery, benchmarks["discovery"], 3),
        "confirmation": evaluate_period(
            confirmation, benchmarks["confirmation"], 2
        ),
    }
    return {
        "factor": factor,
        "horizon": horizon,
        "direction": direction,
        "direction_selected_on_discovery_only": True,
        "discovery": discovery,
        "confirmation": confirmation,
        "benchmarks": benchmarks,
        "passed": all(all(values.values()) for values in checks.values()),
        "checks": checks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    base = shared.load_base_panel(data_dir)
    trading_dates = base.get_column("date").unique().sort().to_list()
    targeted = attach_targets(base, trading_dates)
    investable = shared.investable_panel(targeted)
    del base
    gc.collect()

    calendar_by_horizon = {
        horizon: {
            "discovery": signal_dates(
                trading_dates, DISCOVERY_START, DISCOVERY_END, horizon
            ),
            "confirmation": signal_dates(
                trading_dates, CONFIRMATION_START, CONFIRMATION_END, horizon
            ),
        }
        for horizon in HORIZONS
    }
    benchmarks: dict[int, dict[str, dict[str, Any]]] = {}
    for horizon in HORIZONS:
        benchmarks[horizon] = {
            period: metrics(
                benchmark_returns(
                    investable,
                    horizon,
                    calendar_by_horizon[horizon][period],
                    DISCOVERY_END if period == "discovery" else CONFIRMATION_END,
                ),
                horizon,
            )
            for period in ("discovery", "confirmation")
        }

    records: list[dict[str, Any]] = []
    for offset in range(0, len(FACTOR_NAMES), BATCH_SIZE):
        batch = FACTOR_NAMES[offset : offset + BATCH_SIZE]
        factored_base = FactorBacktestService._compute_missing_factors(
            targeted, set(batch), assume_sorted=False
        )
        factored = shared.investable_panel(factored_base)
        for factor in batch:
            if factor not in factored.columns:
                records.append(
                    {"factor": factor, "error": "factor_not_materialized", "passed": False}
                )
                continue
            for horizon in HORIZONS:
                records.append(
                    screen_factor(
                        factored,
                        factor,
                        horizon,
                        calendar_by_horizon[horizon]["discovery"],
                        calendar_by_horizon[horizon]["confirmation"],
                        benchmarks[horizon],
                    )
                )
        del factored, factored_base
        gc.collect()

    promoted = [record for record in records if record.get("passed")]
    promoted.sort(
        key=lambda row: min(
            float(row["discovery"].get("annualized") or -math.inf)
            - float(row["benchmarks"]["discovery"].get("annualized") or -math.inf),
            float(row["confirmation"].get("annualized") or -math.inf)
            - float(row["benchmarks"]["confirmation"].get("annualized") or -math.inf),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "p0-main-board-short-horizon-factor-ceiling-v1",
        "contract_frozen": "2026-09-03",
        "periods": {
            "discovery": {"start": DISCOVERY_START, "end": DISCOVERY_END},
            "confirmation": {
                "start": CONFIRMATION_START,
                "end": CONFIRMATION_END,
            },
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "horizons": list(HORIZONS),
            "portfolio_size": PORTFOLIO_SIZE,
            "round_trip_cost": ROUND_TRIP_COST,
            "gross_ceiling_only": True,
            "exact_account_backtest": False,
        },
        "counts": {
            "factors": len(FACTOR_NAMES),
            "factor_horizon_trials": len(records),
            "direction_legs_evaluated": len(FACTOR_NAMES) * len(HORIZONS) * 2,
            "promoted": len(promoted),
        },
        "promoted": promoted,
        "all_trials": records,
        "decision": {
            "verdict": (
                "BUILD_EXACT_ACCOUNTS" if promoted else "TERMINATE_DAILY_FACTOR_FAMILY"
            ),
            "validation_read": False,
            "known_stress_read": False,
        },
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
                "counts": payload["counts"],
                "promoted": promoted,
                "decision": payload["decision"],
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
        default=Path(
            "/app/data/research/p0_main_board_short_horizon_factor_ceiling_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
