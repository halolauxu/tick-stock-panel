"""Build the outcome-blind R3-01 company-event industry diffusion candidates."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import audit_p0_short_horizon_event_facts as facts  # noqa: E402
import run_p0_forecast_drift_development as forecast  # noqa: E402
import run_p0_industry_confirmed_forecast_drift_discovery as industry  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_short_horizon_event_account as source_account  # noqa: E402

SCHEMA_VERSION = "p0-short-horizon-industry-diffusion-candidates-v1"
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_SIGNAL_AMOUNT = 50_000_000.0
MIN_FIVE_DAY_RESIDUAL = -0.10
MAX_FIVE_DAY_RESIDUAL = 0.0
MIN_TWENTY_DAY_RETURN = -0.10
MAX_PEERS_PER_SOURCE = 2


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def rank_peer_candidates(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.sort(
            [
                "source_event_id",
                "prior_roe",
                "five_day_industry_residual",
                "symbol",
            ],
            descending=[False, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("source_event_id").alias("source_peer_rank")
        )
        .filter(pl.col("source_peer_rank") <= MAX_PEERS_PER_SOURCE)
        .sort(
            [
                "entry_date",
                "symbol",
                "source_p_change_min",
                "source_event_id",
            ],
            descending=[False, False, True, False],
            nulls_last=True,
        )
        .unique(subset=["entry_date", "symbol"], keep="first", maintain_order=True)
        .sort(["entry_date", "l1_code", "source_peer_rank", "symbol"])
    )


def evaluate_data_gate(audit: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "at_least_150_candidate_rows": audit["candidate_rows"] >= 150,
        "at_least_100_candidate_symbols": audit["candidate_symbols"] >= 100,
        "at_least_50_source_announcement_days": audit["source_announcement_days"] >= 50,
        "at_least_8_industries": audit["industries"] >= 8,
        "no_duplicate_peer_days": audit["duplicate_peer_days"] == 0,
        "no_future_financial_rows": audit["future_financial_rows"] == 0,
        "entry_dates_complete": audit["missing_entry_dates"] == 0,
    }
    passed = all(checks.values())
    return {
        "verdict": ("PASS_TO_INDUSTRY_ACCOUNT" if passed else "BLOCKED_OR_REJECTED_SAMPLE"),
        "passed": passed,
        "checks": checks,
        "failures": [name for name, result in checks.items() if not result],
        "market_outcomes_read": False,
    }


def _panel_features(data_dir: Path) -> tuple[pl.DataFrame, list[date]]:
    raw = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START - timedelta(days=120)
    )
    panel = baseline.prepare_panel(
        baseline.attach_point_in_time_data(main_board.filter_main_board(raw), data_dir)
    ).sort(["symbol", "date"])
    panel = panel.with_columns(
        pl.col("close").shift(5).over("symbol").alias("_close_5"),
        pl.col("_global_index").shift(5).over("symbol").alias("_index_5"),
        pl.col("close").shift(20).over("symbol").alias("_close_20"),
        pl.col("_global_index").shift(20).over("symbol").alias("_index_20"),
    ).with_columns(
        pl.when(pl.col("_global_index") - pl.col("_index_5") == 5)
        .then(pl.col("close") / pl.col("_close_5") - 1.0)
        .otherwise(None)
        .alias("five_day_return"),
        pl.when(pl.col("_global_index") - pl.col("_index_20") == 20)
        .then(pl.col("close") / pl.col("_close_20") - 1.0)
        .otherwise(None)
        .alias("twenty_day_return"),
    )
    membership = industry.load_point_in_time_membership(data_dir)
    mapped = (
        panel.sort(["symbol", "date"])
        .join_asof(
            membership,
            left_on="date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("l1_code").is_not_null()
            & (pl.col("out_date").is_null() | (pl.col("date") <= pl.col("out_date")))
        )
    )
    industry_returns = mapped.group_by("date", "l1_code").agg(
        pl.col("five_day_return").median().alias("industry_five_day_return")
    )
    features = mapped.join(industry_returns, on=["date", "l1_code"], how="left").with_columns(
        (pl.col("five_day_return") - pl.col("industry_five_day_return")).alias(
            "five_day_industry_residual"
        )
    )
    all_dates = (
        panel.filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END))
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    return features, all_dates


def _source_times(sources: pl.DataFrame, all_dates: list[date]) -> pl.DataFrame:
    calendar = pl.DataFrame({"trading_date": all_dates}).sort("trading_date")
    return (
        sources.with_columns(
            pl.concat_str(
                "symbol",
                pl.col("ann_date").cast(pl.Utf8),
                pl.col("period_end").cast(pl.Utf8),
                separator="|",
            ).alias("source_event_id"),
            pl.col("symbol").alias("source_symbol"),
            pl.col("p_change_min").alias("source_p_change_min"),
            pl.col("p_change_max").alias("source_p_change_max"),
        )
        .sort("ann_date")
        .join_asof(
            calendar,
            left_on="ann_date",
            right_on="trading_date",
            strategy="backward",
        )
        .rename({"trading_date": "signal_quote_date"})
        .with_columns((pl.col("ann_date") + pl.duration(days=1)).alias("available_after"))
        .sort("available_after")
        .join_asof(
            calendar.rename({"trading_date": "entry_date"}),
            left_on="available_after",
            right_on="entry_date",
            strategy="forward",
        )
        .drop_nulls(["signal_quote_date", "entry_date"])
        .select(
            "source_event_id",
            "source_symbol",
            "ann_date",
            "period_end",
            "entry_date",
            "signal_quote_date",
            "l1_code",
            "l1_name",
            "source_p_change_min",
            "source_p_change_max",
        )
    )


def _own_positive_events(data_dir: Path) -> pl.DataFrame:
    return (
        forecast.categorize_events(forecast.load_forecasts(data_dir))
        .filter(
            pl.col("ann_date").is_between(DEVELOPMENT_START, DEVELOPMENT_END)
            & pl.col("category").is_in(industry.POSITIVE_FORECAST_CATEGORIES)
        )
        .select("symbol", "ann_date")
        .unique()
        .with_columns(pl.lit(True).alias("own_positive_forecast"))
    )


def build(data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    sources, source_audit = source_account.load_qualified_events(
        data_dir, DEVELOPMENT_START, DEVELOPMENT_END
    )
    features, all_dates = _panel_features(data_dir)
    source_times = _source_times(sources, all_dates)
    peers = (
        source_times.join(
            features,
            left_on=["signal_quote_date", "l1_code"],
            right_on=["date", "l1_code"],
            how="inner",
        )
        .filter(
            (pl.col("symbol") != pl.col("source_symbol"))
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("amount") >= MIN_SIGNAL_AMOUNT)
            & pl.col("five_day_industry_residual").is_between(
                MIN_FIVE_DAY_RESIDUAL, MAX_FIVE_DAY_RESIDUAL, closed="both"
            )
            & (pl.col("twenty_day_return") >= MIN_TWENTY_DAY_RETURN)
        )
        .join(_own_positive_events(data_dir), on=["symbol", "ann_date"], how="left")
        .filter(~pl.col("own_positive_forecast").fill_null(False))
    )
    metrics = facts._read_many(list((data_dir / "financials" / "metrics").glob("*.parquet")))
    cash_flow = facts._read_many(list((data_dir / "financials" / "cash_flow").glob("*.parquet")))
    enriched = facts.attach_prior_financials(peers, metrics, cash_flow)
    future_financial_rows = enriched.filter(
        (pl.col("prior_metrics_announce_date") >= pl.col("ann_date"))
        | (pl.col("prior_cash_announce_date") >= pl.col("ann_date"))
    ).height
    qualified = enriched.filter(
        (pl.col("prior_roe") > 0)
        & (pl.col("prior_operating_cash_to_revenue") > 0)
        & (pl.col("prior_net_operating_cash_flow") > 0)
    )
    ranked = rank_peer_candidates(qualified)
    duplicate_peer_days = ranked.height - ranked.unique(subset=["entry_date", "symbol"]).height
    audit = {
        **source_audit,
        "mapped_source_events": source_times.height,
        "raw_peer_pairs": peers.height,
        "financially_qualified_peer_pairs": qualified.height,
        "candidate_rows": ranked.height,
        "candidate_symbols": ranked.get_column("symbol").n_unique(),
        "source_events_used": ranked.get_column("source_event_id").n_unique(),
        "source_announcement_days": ranked.get_column("ann_date").n_unique(),
        "industries": ranked.get_column("l1_code").n_unique(),
        "duplicate_peer_days": duplicate_peer_days,
        "future_financial_rows": future_financial_rows,
        "missing_entry_dates": ranked.get_column("entry_date").null_count(),
    }
    return ranked, audit


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    candidates, audit = build(data_dir)
    candidate_path = output.with_suffix(".parquet")
    _atomic_parquet(candidates, candidate_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_frozen": "2026-09-04",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "market_outcomes_read": False,
        },
        "assumptions": {
            "source": "R2_01_operating_event_with_positive_prior_cash_quality",
            "industry_membership": "point_in_time_sw_l1",
            "five_day_industry_residual": [
                MIN_FIVE_DAY_RESIDUAL,
                MAX_FIVE_DAY_RESIDUAL,
            ],
            "minimum_twenty_day_return": MIN_TWENTY_DAY_RETURN,
            "max_peers_per_source": MAX_PEERS_PER_SOURCE,
            "ranking": "prior_roe_desc_residual_desc_symbol_asc",
        },
        "audit": audit,
        "candidate_table": {
            "path": str(candidate_path),
            "rows": candidates.height,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
        "decision": evaluate_data_gate(audit),
    }
    _atomic_json(payload, output)
    print(
        json.dumps(
            {
                **payload,
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
