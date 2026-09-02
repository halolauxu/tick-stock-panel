"""Screen frozen resilience overlays inside the main-board micro-cap universe."""

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

import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

START = date(2014, 1, 1)
END = date(2026, 8, 28)
TARGET_POSITIONS = 20
VARIANT_IDS = (
    "cap_smallest",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "momentum_120d",
    "liquidity_20d",
    "low_volatility_20d",
    "resilience_composite",
    "adaptive_resilience",
    "cap_resilience_barbell",
)


def attach_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach signal-close features with explicit trading-index continuity."""
    work = panel.sort(["symbol", "date"]).with_columns(
        *(
            pl.col("_global_index")
            .shift(window)
            .over("symbol")
            .alias(f"_index_{window}d")
            for window in (5, 20, 60, 120)
        ),
        *(
            pl.col("close")
            .shift(window)
            .over("symbol")
            .alias(f"_close_{window}d")
            for window in (5, 20, 60, 120)
        ),
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d"),
        pl.col("daily_return")
        .rolling_std(window_size=20, min_samples=20)
        .over("symbol")
        .alias("volatility_20d"),
    )
    return work.with_columns(
        *(
            pl.when(
                pl.col("_global_index")
                == pl.col(f"_index_{window}d") + window
            )
            .then(pl.col("close") / pl.col(f"_close_{window}d") - 1.0)
            .otherwise(None)
            .alias(f"return_{window}d")
            for window in (5, 20, 60, 120)
        )
    )


def _ranked(
    frame: pl.DataFrame,
    columns: list[str],
    descending: list[bool],
    *,
    limit: int = TARGET_POSITIONS,
) -> pl.DataFrame:
    ordered = frame.sort(
        ["date", *columns, "market_cap", "symbol"],
        descending=[False, *descending, False, False],
        nulls_last=True,
    )
    return (
        ordered.with_columns(
            pl.int_range(1, pl.len() + 1)
            .over("date")
            .alias("selection_rank")
        )
        .filter(pl.col("selection_rank") <= limit)
        .sort(["entry_date", "selection_rank", "symbol"])
    )


def prepare_microcap_observations(observations: pl.DataFrame) -> pl.DataFrame:
    base = observations.filter(
        (pl.col("date") >= pl.lit(START))
        & (pl.col("date") <= pl.lit(END))
        & (pl.col("cap_decile") == 0)
    )
    count = pl.len().over("date").cast(pl.Float64)
    momentum_rank = (
        pl.col("return_60d").rank(method="average").over("date") / count
    )
    liquidity_rank = (
        pl.col("mean_amount_20d").rank(method="average").over("date")
        / count
    )
    low_volatility_rank = 1.0 - (
        pl.col("volatility_20d").rank(method="average").over("date") - 1.0
    ) / (count - 1.0)
    complete = (
        pl.col("return_60d").is_not_null()
        & pl.col("mean_amount_20d").is_not_null()
        & pl.col("volatility_20d").is_not_null()
    )
    return base.with_columns(
        pl.col("return_60d")
        .median()
        .over("date")
        .alias("microcap_median_return_60d"),
        pl.when(complete)
        .then((momentum_rank + liquidity_rank + low_volatility_rank) / 3.0)
        .otherwise(None)
        .alias("resilience_score"),
    )


def build_candidate_sets(
    observations: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    base = prepare_microcap_observations(observations)
    complete = base.filter(pl.col("resilience_score").is_not_null())
    result = {
        "cap_smallest": _ranked(base, ["market_cap"], [False]),
        "reversal_5d": _ranked(
            base.filter(pl.col("return_5d").is_not_null()),
            ["return_5d"],
            [False],
        ),
        "momentum_20d": _ranked(
            base.filter(pl.col("return_20d").is_not_null()),
            ["return_20d"],
            [True],
        ),
        "momentum_60d": _ranked(
            base.filter(pl.col("return_60d").is_not_null()),
            ["return_60d"],
            [True],
        ),
        "momentum_120d": _ranked(
            base.filter(pl.col("return_120d").is_not_null()),
            ["return_120d"],
            [True],
        ),
        "liquidity_20d": _ranked(
            base.filter(pl.col("mean_amount_20d").is_not_null()),
            ["mean_amount_20d"],
            [True],
        ),
        "low_volatility_20d": _ranked(
            base.filter(pl.col("volatility_20d").is_not_null()),
            ["volatility_20d"],
            [False],
        ),
        "resilience_composite": _ranked(
            complete,
            ["resilience_score"],
            [True],
        ),
    }
    strong = base.filter(pl.col("microcap_median_return_60d") >= 0)
    weak = complete.filter(pl.col("microcap_median_return_60d") < 0)
    result["adaptive_resilience"] = pl.concat(
        [
            _ranked(strong, ["market_cap"], [False]),
            _ranked(weak, ["resilience_score"], [True]),
        ],
        how="diagonal_relaxed",
    ).sort(["entry_date", "selection_rank", "symbol"])

    cap_ten = _ranked(base, ["market_cap"], [False], limit=10)
    cap_keys = cap_ten.select("date", "symbol")
    resilient_ten = _ranked(
        complete.join(cap_keys, on=["date", "symbol"], how="anti"),
        ["resilience_score"],
        [True],
        limit=10,
    ).with_columns((pl.col("selection_rank") + 10).alias("selection_rank"))
    result["cap_resilience_barbell"] = pl.concat(
        [cap_ten, resilient_ten], how="diagonal_relaxed"
    ).sort(["entry_date", "selection_rank", "symbol"])
    return result


def _weekly_returns(candidates: pl.DataFrame) -> pl.DataFrame:
    return (
        candidates.group_by("date", "entry_date", maintain_order=True)
        .agg(
            pl.len().alias("selected_count"),
            pl.col("net_return").count().alias("valid_return_count"),
            (
                pl.col("net_return").fill_null(0.0).sum()
                / TARGET_POSITIONS
            ).alias("weekly_return"),
        )
        .sort("entry_date")
        .with_columns(pl.col("entry_date").dt.year().alias("year"))
    )


def summarize_candidate(candidates: pl.DataFrame) -> dict[str, Any]:
    weekly = _weekly_returns(candidates)
    returns = weekly.get_column("weekly_return").to_list()
    yearly = []
    for year in range(START.year, END.year + 1):
        values = (
            weekly.filter(pl.col("year") == year)
            .get_column("weekly_return")
            .to_list()
        )
        yearly.append(
            {
                "year": year,
                "return": baseline._compound(values),
                "weeks": len(values),
            }
        )
    yearly_map = {row["year"]: row["return"] for row in yearly}
    complete_years_positive = all(
        yearly_map.get(year) is not None and yearly_map[year] > 0
        for year in range(2014, 2026)
    )
    return {
        "metrics": {
            "annualized": baseline._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "yearly": yearly,
        },
        "coverage": {
            "weeks": weekly.height,
            "mean_selected_count": weekly.get_column(
                "selected_count"
            ).mean(),
            "mean_valid_return_count": weekly.get_column(
                "valid_return_count"
            ).mean(),
            "fully_populated_weeks": weekly.filter(
                pl.col("selected_count") == TARGET_POSITIONS
            ).height,
        },
        "screen_checks": {
            "return_2026_above_30pct": (
                yearly_map.get(2026) is not None
                and yearly_map[2026] > 0.30
            ),
            "every_year_2014_2025_positive": complete_years_positive,
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=END)
    )
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = attach_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    observations = baseline.build_weekly_observations(panel)
    del panel
    gc.collect()
    candidates = build_candidate_sets(observations)
    del observations
    gc.collect()

    results = {
        variant: summarize_candidate(candidates[variant])
        for variant in VARIANT_IDS
    }
    control = results["cap_smallest"]["metrics"]
    control_yearly = {
        row["year"]: row["return"] for row in control["yearly"]
    }
    promoted = []
    for variant, result in results.items():
        yearly = result["metrics"]["yearly"]
        for row in yearly:
            control_return = control_yearly.get(row["year"])
            row["difference_vs_control"] = (
                row["return"] - control_return
                if row["return"] is not None and control_return is not None
                else None
            )
        checks = result["screen_checks"]
        checks["max_drawdown_not_worse_than_control"] = (
            result["metrics"]["max_drawdown"]
            >= control["max_drawdown"]
        )
        result["passed_screen"] = all(checks.values())
        if result["passed_screen"] and variant != "cap_smallest":
            promoted.append(variant)

    payload = {
        "schema_version": "p0-main-board-microcap-resilience-discovery-v1",
        "contract_frozen": "2026-09-02",
        "research_class": "known_full_history_mechanism_discovery",
        "period": {"start": START, "end": END},
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "target_positions": TARGET_POSITIONS,
            "rebalance": "weekly_signal_close_next_open",
            "missing_selected_return_treatment": "zero_return_cash_proxy",
            "account_confirmation_required": True,
        },
        "variant_ids": VARIANT_IDS,
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
            {
                "results": results,
                "promoted_to_account_confirmation": promoted,
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
            "/app/data/research/"
            "p0_main_board_microcap_resilience_discovery_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
