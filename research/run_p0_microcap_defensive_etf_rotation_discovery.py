"""Screen frozen micro-cap and defensive-ETF rotation candidates."""

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

import collect_p0_microcap_defensive_etf_data as collector  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_microcap_resilience_discovery as resilience  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

START = date(2014, 1, 1)
END = collector.END
MOMENTUM_DAYS = 120
LIQUIDITY_DAYS = 20
MIN_MEAN_AMOUNT = 50_000_000.0
VARIANT_IDS = (
    "absolute_switch",
    "relative_rotation",
    "microcap_70_defensive_30",
)


def prepare_etf_panel(
    daily: pl.DataFrame, adjustments: pl.DataFrame
) -> pl.DataFrame:
    dates = daily.select("date").unique().sort("date").with_row_index(
        "_global_index"
    )
    return (
        daily.join(adjustments, on=["symbol", "date"], how="inner")
        .join(dates, on="date", how="left")
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("open") * pl.col("adj_factor")).alias("adjusted_open"),
            (pl.col("close") * pl.col("adj_factor")).alias(
                "adjusted_close"
            ),
            pl.col("_global_index")
            .shift(MOMENTUM_DAYS)
            .over("symbol")
            .alias("_index_120d"),
            (pl.col("close") * pl.col("adj_factor"))
            .shift(MOMENTUM_DAYS)
            .over("symbol")
            .alias("_close_120d"),
            pl.col("amount")
            .rolling_mean(
                window_size=LIQUIDITY_DAYS, min_samples=LIQUIDITY_DAYS
            )
            .over("symbol")
            .alias("mean_amount_20d"),
        )
        .with_columns(
            pl.when(
                pl.col("_global_index")
                == pl.col("_index_120d") + MOMENTUM_DAYS
            )
            .then(pl.col("adjusted_close") / pl.col("_close_120d") - 1.0)
            .otherwise(None)
            .alias("momentum_120d")
        )
    )


def build_microcap_weekly(data_dir: Path) -> pl.DataFrame:
    source = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=END)
    )
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = resilience.attach_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    observations = baseline.build_weekly_observations(panel)
    del panel
    gc.collect()
    candidates = resilience.build_candidate_sets(observations)[
        "cap_smallest"
    ]
    return (
        candidates.group_by("date", "entry_date", maintain_order=True)
        .agg(
            pl.col("exit_date").drop_nulls().first().alias("exit_date"),
            pl.col("microcap_median_return_60d")
            .first()
            .alias("unused_median_return_60d"),
            pl.col("return_120d")
            .median()
            .alias("microcap_momentum_120d"),
            (
                pl.col("net_return").fill_null(0.0).sum()
                / resilience.TARGET_POSITIONS
            ).alias("microcap_return"),
        )
        .filter(pl.col("entry_date") >= pl.lit(START))
        .sort("entry_date")
    )


def build_best_etf_weekly(
    panel: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    signal = panel.join(schedule, on="date", how="inner").filter(
        (pl.col("momentum_120d") > 0)
        & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT)
    )
    entry = panel.select(
        "symbol",
        pl.col("date").alias("entry_date"),
        pl.col("adjusted_open").alias("entry_open"),
        pl.col("amount").alias("entry_amount"),
        pl.col("volume").alias("entry_volume"),
    )
    exit_quotes = panel.select(
        "symbol",
        pl.col("date").alias("exit_date"),
        pl.col("adjusted_open").alias("exit_open"),
        pl.col("amount").alias("exit_amount"),
        pl.col("volume").alias("exit_volume"),
    )
    ranked = (
        signal.join(entry, on=["symbol", "entry_date"], how="left")
        .join(exit_quotes, on=["symbol", "exit_date"], how="left")
        .filter(
            (pl.col("entry_open") > 0)
            & (pl.col("exit_open") > 0)
            & (pl.col("entry_volume") > 0)
            & (pl.col("exit_volume") > 0)
            & (pl.col("entry_amount") >= MIN_MEAN_AMOUNT)
            & (pl.col("exit_amount") >= MIN_MEAN_AMOUNT)
        )
        .sort(
            ["date", "momentum_120d", "mean_amount_20d", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("rank")
        )
        .filter(pl.col("rank") == 1)
        .with_columns(
            (
                (
                    pl.col("exit_open")
                    * (1.0 - baseline.COMMISSION_PCT - baseline.SLIPPAGE_PCT)
                )
                / (
                    pl.col("entry_open")
                    * (1.0 + baseline.COMMISSION_PCT + baseline.SLIPPAGE_PCT)
                )
                - 1.0
            ).alias("etf_return")
        )
    )
    return ranked.select(
        "date",
        "symbol",
        pl.col("momentum_120d").alias("etf_momentum_120d"),
        "etf_return",
    )


def build_variant_returns(
    microcap: pl.DataFrame, best_etf: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    joined = (
        microcap.join(best_etf, on="date", how="left")
        .with_columns(
            pl.col("etf_return").fill_null(0.0),
            pl.col("etf_momentum_120d").fill_null(-math.inf),
        )
        .sort("entry_date")
    )
    absolute = joined.with_columns(
        pl.when(pl.col("microcap_momentum_120d") > 0)
        .then(pl.col("microcap_return"))
        .otherwise(pl.col("etf_return"))
        .alias("weekly_return"),
        pl.when(pl.col("microcap_momentum_120d") > 0)
        .then(pl.lit("microcap"))
        .when(pl.col("symbol").is_not_null())
        .then(pl.col("symbol"))
        .otherwise(pl.lit("cash"))
        .alias("selected_asset"),
    )
    relative = joined.with_columns(
        pl.when(
            (pl.col("microcap_momentum_120d") > 0)
            & (
                pl.col("microcap_momentum_120d")
                >= pl.col("etf_momentum_120d")
            )
        )
        .then(pl.col("microcap_return"))
        .when(pl.col("symbol").is_not_null())
        .then(pl.col("etf_return"))
        .otherwise(0.0)
        .alias("weekly_return"),
        pl.when(
            (pl.col("microcap_momentum_120d") > 0)
            & (
                pl.col("microcap_momentum_120d")
                >= pl.col("etf_momentum_120d")
            )
        )
        .then(pl.lit("microcap"))
        .when(pl.col("symbol").is_not_null())
        .then(pl.col("symbol"))
        .otherwise(pl.lit("cash"))
        .alias("selected_asset"),
    )
    blend = joined.with_columns(
        (0.70 * pl.col("microcap_return") + 0.30 * pl.col("etf_return")).alias(
            "weekly_return"
        ),
        pl.concat_str(
            pl.lit("microcap70+"),
            pl.col("symbol").fill_null("cash"),
        ).alias("selected_asset"),
    )
    columns = ["date", "entry_date", "weekly_return", "selected_asset"]
    return {
        "absolute_switch": absolute.select(columns),
        "relative_rotation": relative.select(columns),
        "microcap_70_defensive_30": blend.select(columns),
    }


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    work = frame.with_columns(pl.col("entry_date").dt.year().alias("year"))
    yearly = []
    for year in range(START.year, END.year + 1):
        values = work.filter(pl.col("year") == year).get_column(
            "weekly_return"
        ).to_list()
        yearly.append(
            {"year": year, "return": baseline._compound(values), "weeks": len(values)}
        )
    values = work.get_column("weekly_return").to_list()
    asset_weeks = {
        row["selected_asset"]: row["len"]
        for row in work.group_by("selected_asset").len().to_dicts()
    }
    return {
        "metrics": {
            "annualized": baseline._annualized(values),
            "total_return": baseline._compound(values),
            "max_drawdown": baseline._max_drawdown(values),
            "yearly": yearly,
        },
        "asset_weeks": asset_weeks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    data_audit = json.loads(
        (data_dir / "research" / "p0_microcap_defensive_etf_data_v1.json")
        .read_text(encoding="utf-8")
    )
    if data_audit["status"] != "DATA_QUALIFIED":
        raise ValueError("defensive ETF data did not pass audit")
    root = data_dir / "research" / "microcap_defensive_etf_v1"
    etf_panel = prepare_etf_panel(
        pl.read_parquet(root / "daily_raw.parquet"),
        pl.read_parquet(root / "adjustments.parquet"),
    )
    microcap = build_microcap_weekly(data_dir)
    schedule = microcap.select("date", "entry_date", "exit_date")
    best_etf = build_best_etf_weekly(etf_panel, schedule)
    variants = build_variant_returns(microcap, best_etf)
    control = summarize(
        microcap.select(
            "date",
            "entry_date",
            pl.col("microcap_return").alias("weekly_return"),
        ).with_columns(pl.lit("microcap").alias("selected_asset"))
    )
    results = {variant: summarize(variants[variant]) for variant in VARIANT_IDS}
    control_years = {
        row["year"]: row["return"]
        for row in control["metrics"]["yearly"]
    }
    promoted = []
    for variant, result in results.items():
        yearly = result["metrics"]["yearly"]
        yearly_map = {row["year"]: row["return"] for row in yearly}
        for row in yearly:
            row["difference_vs_control"] = (
                row["return"] - control_years[row["year"]]
            )
        checks = {
            "return_2026_above_30pct": yearly_map[2026] > 0.30,
            "every_year_2014_2025_positive": all(
                yearly_map[year] > 0 for year in range(2014, 2026)
            ),
            "max_drawdown_not_worse_than_control": (
                result["metrics"]["max_drawdown"]
                >= control["metrics"]["max_drawdown"]
            ),
        }
        result["screen_checks"] = checks
        result["passed_screen"] = all(checks.values())
        if result["passed_screen"]:
            promoted.append(variant)
    payload = {
        "schema_version": "p0-microcap-defensive-etf-rotation-discovery-v1",
        "contract_frozen": "2026-09-02",
        "research_class": "known_full_history_mechanism_discovery",
        "period": {"start": START, "end": END},
        "assumptions": {
            "symbols": collector.SYMBOLS,
            "momentum_days": MOMENTUM_DAYS,
            "minimum_mean_amount": MIN_MEAN_AMOUNT,
            "rebalance": "weekly_signal_close_next_open",
            "account_confirmation_required": True,
        },
        "control": control,
        "results": results,
        "promoted_to_account_confirmation": promoted,
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
            "/app/data/research/"
            "p0_microcap_defensive_etf_rotation_discovery_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
