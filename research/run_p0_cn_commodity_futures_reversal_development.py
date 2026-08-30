"""Run the frozen Chinese commodity-futures cross-sectional reversal study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

RETURN_DAYS = 5
TAIL_COUNT = 4


def _load_execution_module():
    path = Path(__file__).with_name(
        "run_p0_cn_commodity_futures_trend_development.py"
    )
    spec = importlib.util.spec_from_file_location("p0_futures_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load execution module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


execution = _load_execution_module()


def weekly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%G-%V").alias("week"),
        )
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )


def build_reversal_signals(
    panel: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    candidates = (
        panel.with_columns(
            pl.col("return_index")
            .shift(RETURN_DAYS)
            .over("series")
            .alias("_index_lag"),
            pl.col("_global_index")
            .shift(RETURN_DAYS)
            .over("series")
            .alias("_global_lag"),
        )
        .with_columns(
            pl.when(
                pl.col("_global_index") == pl.col("_global_lag") + RETURN_DAYS
            )
            .then(pl.col("return_index") / pl.col("_index_lag") - 1.0)
            .otherwise(None)
            .alias("return_5d")
        )
        .join(
            schedule,
            left_on="date",
            right_on="signal_date",
            how="inner",
        )
        .filter(
            pl.col("return_5d").is_not_null()
            & pl.col("volatility_20d").is_not_null()
            & (pl.col("volatility_20d") > 0)
        )
        .with_columns(
            pl.col("return_5d")
            .rank(method="ordinal")
            .over("date")
            .alias("return_rank"),
            pl.len().over("date").alias("cross_section_count"),
        )
        .filter(pl.col("cross_section_count") >= 2 * TAIL_COUNT)
        .filter(
            (pl.col("return_rank") <= TAIL_COUNT)
            | (
                pl.col("return_rank")
                > pl.col("cross_section_count") - TAIL_COUNT
            )
        )
        .with_columns(
            pl.when(pl.col("return_rank") <= TAIL_COUNT)
            .then(1.0)
            .otherwise(-1.0)
            .alias("direction"),
            (
                1.0
                / pl.col("volatility_20d").clip(
                    lower_bound=execution.VOLATILITY_FLOOR
                )
            ).alias("inverse_volatility"),
        )
        .with_columns(
            pl.col("inverse_volatility")
            .sum()
            .over(["date", "direction"])
            .alias("side_inverse_sum")
        )
        .with_columns(
            (
                pl.col("direction")
                * (execution.GROSS_LEVERAGE / 2.0)
                * pl.col("inverse_volatility")
                / pl.col("side_inverse_sum")
            ).alias("target_weight")
        )
    )
    return candidates.select(
        "date",
        "entry_date",
        "series",
        "return_5d",
        "return_rank",
        "target_weight",
    ).sort(["entry_date", "series"])


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    return execution._json_default(value)


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "cn_commodity_futures"
    continuous = pl.read_parquet(root / "continuous_daily.parquet")
    mapping = pl.read_parquet(root / "main_mapping.parquet")
    contract_daily = pl.read_parquet(root / "contract_daily.parquet")
    contracts = pl.read_parquet(root / "contracts.parquet")
    panel = execution.prepare_signal_panel(continuous, mapping, contract_daily)
    schedule = weekly_schedule(panel)
    signals = build_reversal_signals(panel, schedule)
    all_dates = panel["date"].unique().sort().to_list()
    accounts = {
        name: execution.summarize_account(
            execution.simulate_account(
                signals,
                mapping,
                contract_daily,
                contracts,
                all_dates,
                cash,
            )
        )
        for name, cash in execution.CAPITAL_LEVELS.items()
    }
    benchmark = execution.benchmark_metrics(panel)
    decision = execution.evaluate_gate(accounts, benchmark)
    if decision["passed"]:
        decision["passed"] = False
        decision["verdict"] = "PENDING_PUBLISHED_LIMIT_DATA"
        decision["checks"]["published_preopen_limit_data_complete"] = False
    payload = {
        "schema_version": "p0-cn-commodity-futures-reversal-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": execution.DEVELOPMENT_START,
            "end": execution.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "return_days": RETURN_DAYS,
            "tail_count_each_side": TAIL_COUNT,
            "capital_levels_cny": execution.CAPITAL_LEVELS,
            "gross_leverage": execution.GROSS_LEVERAGE,
            "commission_pct": execution.COMMISSION_PCT,
            "slippage_pct": execution.SLIPPAGE_PCT,
            "daily_participation": execution.DAILY_PARTICIPATION,
            "price_limit_mode": "conservative_full_day_one_price_lock_proxy",
            "published_limit_data_status": "DATA_PERMISSION_BLOCKED",
        },
        "data": {
            "series": panel["series"].n_unique(),
            "trading_days": len(all_dates),
            "signal_rows": signals.height,
            "signal_weeks": signals["entry_date"].n_unique(),
        },
        "benchmark": benchmark,
        "accounts": accounts,
        "decision": decision,
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
                "accounts": {
                    name: {
                        "metrics": result["metrics"],
                        "execution": result["execution"],
                        "integrity": result["integrity"],
                        "ending_equity": result["ending_equity"],
                        "total_cost": result["total_cost"],
                    }
                    for name, result in accounts.items()
                },
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
        default=Path(
            "/app/data/research/p0_cn_commodity_futures_reversal_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
