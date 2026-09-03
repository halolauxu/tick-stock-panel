"""Run the frozen R3-02 development account for industry diffusion peers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_forecast_drift_development as forecast  # noqa: E402
import run_p0_industry_confirmed_forecast_drift_discovery as industry  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_short_horizon_event_account as event_account  # noqa: E402

SCHEMA_VERSION = "p0-short-horizon-industry-diffusion-account-development-v1"
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
EVENT_FAMILY = "operating_forecast_industry_diffusion"
INPUT_NAME = "p0_short_horizon_industry_diffusion_candidates_v1.json"


def load_seeds(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    result_path = data_dir / "research" / INPUT_NAME
    candidate_path = result_path.with_suffix(".parquet")
    if not result_path.is_file() or not candidate_path.is_file():
        raise ValueError("passing R3-01 candidate audit is required")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("decision", {}).get("verdict") != "PASS_TO_INDUSTRY_ACCOUNT":
        raise ValueError("R3-01 candidate audit has not passed")
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if digest != result["candidate_table"]["sha256"]:
        raise ValueError("R3-01 candidate table hash mismatch")
    seeds = pl.read_parquet(candidate_path).sort(["entry_date", "symbol"])
    return seeds, {
        "seed_rows": seeds.height,
        "seed_symbols": seeds.get_column("symbol").n_unique(),
        "seed_announcement_days": seeds.get_column("ann_date").n_unique(),
        "seed_industries": seeds.get_column("l1_code").n_unique(),
        "candidate_audit_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "candidate_table_sha256": digest,
    }


def expand_candidates(
    seeds: pl.DataFrame, all_dates: list[date], horizon: int
) -> tuple[pl.DataFrame, dict[str, Any]]:
    calendar = pl.DataFrame({"entry_date": all_dates}).with_row_index("action_index")
    last_entry_index = len(all_dates) - horizon - event_account.MAX_EXIT_DELAY - 1
    accepted = seeds.join(calendar, on="entry_date", how="inner").filter(
        pl.col("action_index") <= last_entry_index
    )
    expanded = (
        accepted.with_columns(
            pl.int_ranges(pl.col("action_index"), pl.col("action_index") + horizon).alias(
                "_active_indices"
            )
        )
        .explode("_active_indices")
        .drop("entry_date", "action_index")
        .join(
            calendar.rename({"action_index": "_active_indices"}),
            on="_active_indices",
            how="inner",
        )
        .sort(
            [
                "entry_date",
                "symbol",
                "source_p_change_min",
                "prior_roe",
                "five_day_industry_residual",
            ],
            descending=[False, False, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["entry_date", "symbol"], keep="first", maintain_order=True)
        .sort(
            [
                "entry_date",
                "l1_code",
                "source_p_change_min",
                "prior_roe",
                "five_day_industry_residual",
                "symbol",
            ],
            descending=[False, False, True, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["entry_date", "l1_code"]).alias("industry_rank")
        )
        .filter(pl.col("industry_rank") == 1)
        .sort(
            [
                "entry_date",
                "source_p_change_min",
                "prior_roe",
                "five_day_industry_residual",
                "symbol",
            ],
            descending=[False, True, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank"),
            pl.lit(EVENT_FAMILY).alias("family"),
        )
        .filter(pl.col("cap_rank") <= event_account.TARGET_POSITIONS)
        .select(
            pl.col("ann_date").alias("date"),
            "entry_date",
            "symbol",
            "l1_code",
            "l1_name",
            pl.col("source_p_change_min").alias("p_change_min"),
            pl.col("source_p_change_max").alias("p_change_max"),
            pl.col("amount").alias("signal_amount"),
            "prior_roe",
            "five_day_industry_residual",
            "cap_rank",
            "family",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )
    return expanded, {
        "seed_rows": seeds.height,
        "accepted_seed_rows": accepted.height,
        "daily_candidate_rows": expanded.height,
        "active_account_days": expanded.get_column("entry_date").n_unique(),
        "candidate_symbols": expanded.get_column("symbol").n_unique(),
        "maximum_industries_per_day": (
            expanded.group_by("entry_date")
            .agg(pl.col("l1_code").n_unique().alias("count"))
            .get_column("count")
            .max()
        ),
        "last_accepted_entry_index": last_entry_index,
    }


def _event_rows(seeds: pl.DataFrame) -> pl.DataFrame:
    return (
        seeds.select(
            "ann_date",
            "symbol",
            "l1_code",
            "l1_name",
            pl.col("source_p_change_min").alias("p_change_min"),
            pl.col("source_p_change_max").alias("p_change_max"),
        )
        .with_columns(pl.lit(EVENT_FAMILY).alias("category"))
        .sort(["ann_date", "symbol"])
    )


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    seeds, seed_audit = load_seeds(data_dir)
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START - timedelta(days=45)
    )
    raw_source = main_board.filter_main_board(raw_all)
    all_dates = (
        raw_source.filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END))
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    event_panel = forecast.prepare_panel(
        forecast.load_panel(
            data_dir,
            start=DEVELOPMENT_START - timedelta(days=45),
            panel_end=DEVELOPMENT_END,
        ).filter(pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN))
    )
    membership = industry.load_point_in_time_membership(data_dir)
    baseline_path = data_dir / "research" / event_account.MICROCAP_BASELINE
    events = _event_rows(seeds)
    results = {}
    for horizon in event_account.HORIZONS:
        candidates, candidate_audit = expand_candidates(seeds, all_dates, horizon)
        event_summary, event_details = event_account.summarize_event_study(
            events, event_panel, membership, horizon
        )
        account_details = event_account.simulate_account_horizon(
            candidates,
            raw_source,
            all_dates,
            data_dir,
            horizon,
            baseline_path,
            "development",
        )
        results[str(horizon)] = {
            "candidate_audit": candidate_audit,
            "event_study": event_summary,
            "event_details": event_details,
            "account": account_details["metrics"],
            "orders": account_details["orders"],
            "settlements": account_details["settlements"],
            "cycles": account_details["cycles"],
            "daily_equity": account_details["daily_equity"],
        }
        gc.collect()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_frozen": "2026-09-04",
        "period": {
            "name": "development",
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "family": EVENT_FAMILY,
            "initial_cash_cny": event_account.INITIAL_CASH,
            "target_positions": event_account.TARGET_POSITIONS,
            "horizons": list(event_account.HORIZONS),
            "cooldown_sessions": event_account.COOLDOWN_SESSIONS,
            "one_position_per_sw_l1_industry": True,
        },
        "data": {
            **seed_audit,
            "microcap_baseline_path": str(baseline_path),
            "microcap_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        },
        "horizons": results,
        "decision": event_account.evaluate_development(results),
    }
    event_account._atomic_json(payload, output)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "period": payload["period"],
                "data": payload["data"],
                "horizons": {
                    key: {
                        "candidate_audit": value["candidate_audit"],
                        "event_study": value["event_study"],
                        "account": value["account"],
                    }
                    for key, value in results.items()
                },
                "decision": payload["decision"],
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            default=event_account._json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
