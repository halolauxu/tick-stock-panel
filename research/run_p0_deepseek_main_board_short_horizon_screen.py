"""Screen frozen persisted DeepSeek hypotheses on a realistic main-board account."""

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

from app.backtest.factor import FactorBacktestService  # noqa: E402

import fixed_horizon_account as fixed  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_neglected_liquidity_premium as liquidity  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

WARMUP_START = date(2013, 8, 29)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_MARKET_CAP = 1_000_000_000.0
MIN_MARKET_CAP_PERCENTILE = 0.30
MIN_MEAN_AMOUNT_20D = 50_000_000.0
ENTRY_SCORE = 75.0
CONTROL_SCORE = 25.0
TOP_RANK = 20
TARGET_POSITIONS = 10
MAX_EXIT_DELAY = 20

MECHANISMS: tuple[dict[str, Any], ...] = (
    {
        "id": "limit_memory_supply_gap",
        "hypothesis_id": "ah-ai-47fb00c824123bdc4b7d",
        "title": "涨停记忆回落后的筹码断档",
        "holding_days": 10,
        "factors": {
            "limit_up_count_20d": (1, 0.4),
            "distance_from_low_60d": (-1, 0.3),
            "turnover_rate": (-1, 0.3),
        },
    },
    {
        "id": "volume_diffusion_momentum",
        "hypothesis_id": "ah-ai-4f900de8ae6d422612d3",
        "title": "交易量扩散下的羊群动量",
        "holding_days": 5,
        "factors": {
            "vol_price_corr_20d": (1, 0.4),
            "vol_trend_5_10": (1, 0.3),
            "up_days_20d": (1, 0.3),
        },
    },
    {
        "id": "high_volatility_panic_repair",
        "hypothesis_id": "ah-ai-d179a730982e04ec0e17",
        "title": "高波动尾盘低收的恐慌修复",
        "holding_days": 3,
        "factors": {
            "annual_vol_20d": (1, 0.4),
            "amplitude": (1, 0.3),
            "close_position": (-1, 0.3),
        },
    },
    {
        "id": "volume_macd_learning",
        "hypothesis_id": "ah-ai-d626dc1968a725615764",
        "title": "量能趋势与MACD柱的散户学习",
        "holding_days": 5,
        "factors": {
            "macd_hist_pct": (1, 0.4),
            "vol_trend_5_60": (1, 0.4),
            "rsi_14": (-1, 0.2),
        },
    },
    {
        "id": "uncrowded_momentum_avoidance",
        "hypothesis_id": "ah-ai-ee9c4a0ac9894259e18c",
        "title": "拥挤交易对动量确认的透支",
        "holding_days": 5,
        "factors": {
            "momentum_20d": (-1, 0.5),
            "vol_price_corr_20d": (-1, 0.5),
        },
    },
)


def load_base_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    panel = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "raw_close",
            "turnover_rate",
            "consecutive_limit_ups",
        )
        .filter(
            pl.col("date").is_between(WARMUP_START, DEVELOPMENT_END, closed="both")
            & pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN)
        )
        .collect(engine="streaming")
    )
    panel = baseline.attach_point_in_time_data(panel, data_dir)
    dates = panel.select("date").unique().sort("date").with_row_index("_global_index")
    return (
        panel.join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
            pl.col("_global_index").shift(19).over("symbol").alias("_index_19d"),
            pl.col("amount")
            .rolling_mean(20, min_samples=20)
            .over("symbol")
            .alias("_mean_amount_20d_raw"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("_index_19d") + 19)
            .then(pl.col("_mean_amount_20d_raw"))
            .otherwise(None)
            .alias("mean_amount_20d")
        )
    )


def investable_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.filter((pl.col("market_cap") > 0) & pl.col("mean_amount_20d").is_finite())
        .sort(["date", "market_cap", "symbol"])
        .with_columns(
            pl.len().over("date").alias("day_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("size_rank"),
        )
        .with_columns(
            (pl.col("size_rank") / pl.col("day_count")).alias(
                "market_cap_percentile"
            )
        )
        .filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & (pl.col("market_cap_percentile") > MIN_MARKET_CAP_PERCENTILE)
            & (pl.col("market_cap") >= MIN_MARKET_CAP)
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("amount") > 0)
        )
    )


def add_composite_score(
    panel: pl.DataFrame, factors: dict[str, tuple[int, float]]
) -> pl.DataFrame:
    required = list(factors)
    work = panel.drop_nulls(required)
    for index, (factor, (direction, weight)) in enumerate(factors.items()):
        rank_column = f"_rank_{index}"
        count_column = f"_count_{index}"
        work = work.with_columns(
            pl.col(factor)
            .rank(method="average", descending=direction < 0)
            .over("date")
            .alias(rank_column),
            pl.col(factor).count().over("date").alias(count_column),
        ).with_columns(
            (pl.col(rank_column) / pl.col(count_column) * 100.0 * weight).alias(
                f"_weighted_{index}"
            )
        )
    weighted = [pl.col(f"_weighted_{index}") for index in range(len(factors))]
    return work.with_columns(pl.sum_horizontal(weighted).alias("composite_score"))


def build_candidates(
    scored: pl.DataFrame,
    next_dates: pl.DataFrame,
    *,
    control: bool,
) -> pl.DataFrame:
    if control:
        selected = scored.filter(pl.col("composite_score") <= CONTROL_SCORE)
        descending = False
    else:
        selected = scored.filter(pl.col("composite_score") >= ENTRY_SCORE)
        descending = True
    return (
        selected.join(next_dates, on="date", how="inner")
        .sort(
            ["entry_date", "composite_score", "symbol"],
            descending=[False, descending, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank"))
        .filter(pl.col("cap_rank") <= TOP_RANK)
        .select(
            "date",
            "entry_date",
            "symbol",
            pl.col("amount").alias("signal_amount"),
            "composite_score",
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate(
    candidate: dict[str, Any], control: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    annualized = float(metrics.get("annualized") or -math.inf)
    control_annualized = float(control["metrics"].get("annualized") or -math.inf)
    benchmark_annualized = float(benchmark.get("annualized") or -math.inf)
    checks = {
        "annualized_at_least_20pct": annualized >= 0.20,
        "annualized_excess_at_least_10pp": annualized - benchmark_annualized >= 0.10,
        "max_drawdown_within_30pct": float(metrics.get("max_drawdown") or -math.inf)
        >= -0.30,
        "at_least_5_positive_years": int(metrics.get("positive_years") or 0) >= 5,
        "mean_cash_ratio_at_most_30pct": float(
            metrics.get("mean_cash_ratio") or math.inf
        )
        <= 0.30,
        "at_least_300_round_trips": int(candidate["account"].get("trade_count") or 0)
        // 2
        >= 300,
        "buy_execution_at_least_90pct": candidate["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "sell_execution_at_least_90pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "no_unresolved_positions": candidate["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciled": candidate["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
        "beats_inverted_control_by_5pp": annualized - control_annualized >= 0.05,
    }
    return {
        "passed": all(checks.values()),
        "annualized_excess": annualized - benchmark_annualized,
        "annualized_minus_control": annualized - control_annualized,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    base = load_base_panel(data_dir)
    raw_source = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    ).filter(pl.col("date") >= DEVELOPMENT_START)
    trading_dates = raw_source.get_column("date").unique().sort().to_list()
    next_dates = pl.DataFrame(
        {"date": trading_dates[:-1], "entry_date": trading_dates[1:]}
    )
    benchmark_panel = baseline.prepare_panel(base)
    benchmark = shared.benchmark_metrics(
        liquidity.benchmark_universe(benchmark_panel).filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
    )
    del benchmark_panel
    gc.collect()

    results: dict[str, Any] = {}
    candidate_counts: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        factor_names = set(mechanism["factors"])
        factor_panel = FactorBacktestService._compute_missing_factors(
            base, factor_names, assume_sorted=True
        )
        scored = add_composite_score(
            investable_panel(factor_panel), mechanism["factors"]
        )
        candidate_frame = build_candidates(scored, next_dates, control=False)
        control_frame = build_candidates(scored, next_dates, control=True)
        del factor_panel, scored
        gc.collect()
        scoped_quotes = fixed.prepare_quotes(
            pl.concat([candidate_frame, control_frame], how="vertical"),
            raw_source,
            data_dir,
        )
        candidate_result = fixed.simulate(
            candidate_frame,
            scoped_quotes.filter(
                pl.col("symbol").is_in(
                    candidate_frame.get_column("symbol").unique()
                )
            ),
            trading_dates,
            initial_cash=shared.INITIAL_CASH,
            target_positions=TARGET_POSITIONS,
            holding_trading_days=mechanism["holding_days"],
            maximum_exit_delay=MAX_EXIT_DELAY,
            period_start=DEVELOPMENT_START,
            period_end=DEVELOPMENT_END,
        )
        control_result = fixed.simulate(
            control_frame,
            scoped_quotes.filter(
                pl.col("symbol").is_in(control_frame.get_column("symbol").unique())
            ),
            trading_dates,
            initial_cash=shared.INITIAL_CASH,
            target_positions=TARGET_POSITIONS,
            holding_trading_days=mechanism["holding_days"],
            maximum_exit_delay=MAX_EXIT_DELAY,
            period_start=DEVELOPMENT_START,
            period_end=DEVELOPMENT_END,
        )
        evaluation = evaluate(candidate_result, control_result, benchmark)
        results[mechanism["id"]] = {
            "hypothesis_id": mechanism["hypothesis_id"],
            "title": mechanism["title"],
            "holding_days": mechanism["holding_days"],
            "candidate": candidate_result,
            "inverted_control": control_result,
            "evaluation": evaluation,
        }
        candidate_counts[mechanism["id"]] = {
            "candidate_rows": candidate_frame.height,
            "candidate_symbols": candidate_frame.get_column("symbol").n_unique(),
            "control_rows": control_frame.height,
            "control_symbols": control_frame.get_column("symbol").n_unique(),
        }
        del candidate_frame, control_frame, scoped_quotes
        gc.collect()

    promoted = [name for name, result in results.items() if result["evaluation"]["passed"]]
    selected = (
        max(
            promoted,
            key=lambda name: results[name]["evaluation"]["annualized_excess"],
        )
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-deepseek-main-board-short-horizon-screen-v1",
        "contract_frozen": "2026-09-03",
        "deepseek_reused_persisted_hypotheses": True,
        "new_model_calls": 0,
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "entry_score": ENTRY_SCORE,
            "control_score": CONTROL_SCORE,
            "top_rank": TOP_RANK,
            "target_positions": TARGET_POSITIONS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "initial_cash_cny": shared.INITIAL_CASH,
        },
        "data": candidate_counts,
        "benchmark": benchmark,
        "results": results,
        "decision": {
            "promoted": promoted,
            "selected": selected,
            "verdict": "FREEZE_SELECTED_FOR_VALIDATION" if selected else "TERMINATE_FAMILY",
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
            {**payload, "output": str(output), "sha256": digest},
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
            "/app/data/research/p0_deepseek_main_board_short_horizon_screen_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
