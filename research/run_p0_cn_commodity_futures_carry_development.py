"""Run the frozen Chinese commodity-futures term-structure study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

TAIL_COUNT = 4
MIN_MAIN_DAYS_TO_EXPIRY = 10
MIN_CURVE_GAP_DAYS = 20


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


def build_curve_candidates(
    panel: pl.DataFrame,
    mapping: pl.DataFrame,
    contract_daily: pl.DataFrame,
    contracts: pl.DataFrame,
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    meta = contracts.select(
        "contract", pl.col("delist_date").alias("contract_delist_date")
    )
    quotes = contract_daily.select("contract", "date", "settle", "volume")
    main = (
        schedule.join(
            mapping,
            left_on="signal_date",
            right_on="date",
            how="inner",
        )
        .select(
            "signal_date",
            "entry_date",
            "series",
            pl.col("contract").alias("main_contract"),
        )
        .join(
            quotes,
            left_on=["signal_date", "main_contract"],
            right_on=["date", "contract"],
            how="inner",
        )
        .rename({"settle": "main_settle", "volume": "main_volume"})
        .join(
            meta,
            left_on="main_contract",
            right_on="contract",
            how="inner",
        )
        .rename({"contract_delist_date": "main_delist_date"})
    )
    contract_series = mapping.select("contract", "series").unique(
        subset=["contract"], keep="first"
    )
    farther = (
        quotes.join(contract_series, on="contract", how="inner")
        .join(meta, on="contract", how="inner")
        .select(
            pl.col("date").alias("signal_date"),
            "series",
            pl.col("contract").alias("far_contract"),
            pl.col("settle").alias("far_settle"),
            pl.col("volume").alias("far_volume"),
            pl.col("contract_delist_date").alias("far_delist_date"),
        )
    )
    volatility = panel.select(
        pl.col("date").alias("signal_date"), "series", "volatility_20d"
    )
    return (
        main.join(farther, on=["signal_date", "series"], how="inner")
        .with_columns(
            (pl.col("main_delist_date") - pl.col("signal_date"))
            .dt.total_days()
            .alias("main_days_to_expiry"),
            (pl.col("far_delist_date") - pl.col("main_delist_date"))
            .dt.total_days()
            .alias("curve_gap_days"),
        )
        .filter(
            (pl.col("main_volume") > 0)
            & (pl.col("far_volume") > 0)
            & (pl.col("main_settle") > 0)
            & (pl.col("far_settle") > 0)
            & (pl.col("main_days_to_expiry") >= MIN_MAIN_DAYS_TO_EXPIRY)
            & (pl.col("curve_gap_days") >= MIN_CURVE_GAP_DAYS)
        )
        .sort(["signal_date", "series", "far_delist_date"])
        .unique(subset=["signal_date", "series"], keep="first", maintain_order=True)
        .with_columns(
            (
                (pl.col("main_settle") / pl.col("far_settle")).log()
                * 365.0
                / pl.col("curve_gap_days")
            ).alias("annualized_carry")
        )
        .join(volatility, on=["signal_date", "series"], how="inner")
        .filter(
            pl.col("volatility_20d").is_not_null()
            & (pl.col("volatility_20d") > 0)
        )
    )


def rank_carry_signals(candidates: pl.DataFrame) -> pl.DataFrame:
    ranked = candidates.with_columns(
        pl.col("annualized_carry")
        .rank(method="ordinal")
        .over("signal_date")
        .alias("carry_rank"),
        pl.len().over("signal_date").alias("cross_section_count"),
    ).filter(pl.col("cross_section_count") >= 2 * TAIL_COUNT)
    selected = (
        ranked.filter(
            (pl.col("carry_rank") <= TAIL_COUNT)
            | (
                pl.col("carry_rank")
                > pl.col("cross_section_count") - TAIL_COUNT
            )
        )
        .with_columns(
            pl.when(
                pl.col("carry_rank")
                > pl.col("cross_section_count") - TAIL_COUNT
            )
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
            .over(["signal_date", "direction"])
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
    return selected.select(
        pl.col("signal_date").alias("date"),
        "entry_date",
        "series",
        "main_contract",
        "far_contract",
        "annualized_carry",
        "carry_rank",
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
    schedule = execution.monthly_schedule(panel)
    candidates = build_curve_candidates(
        panel, mapping, contract_daily, contracts, schedule
    )
    signals = rank_carry_signals(candidates)
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
    payload = {
        "schema_version": "p0-cn-commodity-futures-carry-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": execution.DEVELOPMENT_START,
            "end": execution.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "tail_count_each_side": TAIL_COUNT,
            "minimum_main_days_to_expiry": MIN_MAIN_DAYS_TO_EXPIRY,
            "minimum_curve_gap_days": MIN_CURVE_GAP_DAYS,
            "capital_levels_cny": execution.CAPITAL_LEVELS,
            "gross_leverage": execution.GROSS_LEVERAGE,
            "commission_pct": execution.COMMISSION_PCT,
            "slippage_pct": execution.SLIPPAGE_PCT,
            "daily_participation": execution.DAILY_PARTICIPATION,
        },
        "data": {
            "series": panel["series"].n_unique(),
            "trading_days": len(all_dates),
            "curve_candidates": candidates.height,
            "signal_rows": signals.height,
            "signal_months": signals["entry_date"].n_unique(),
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
            "/app/data/research/p0_cn_commodity_futures_carry_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
