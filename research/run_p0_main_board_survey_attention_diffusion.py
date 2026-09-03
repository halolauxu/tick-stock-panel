"""Run the frozen main-board 10-day institutional-survey diffusion study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_institutional_survey_attention_development as survey  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402

HOLD_TRADING_DAYS = 10
MAX_EXIT_DELAY = 20
CANDIDATE = "main_board_attention_spike"
CONTROL = "main_board_ordinary_survey_control"


def classify_attention_events(surveys: pl.DataFrame) -> pl.DataFrame:
    """Classify attention shocks using only each event's prior history."""
    rows: list[dict[str, Any]] = []
    source = main_board.filter_main_board(surveys).sort(
        ["symbol", "notice_date", "event_id"]
    )
    for symbol_frame in source.partition_by("symbol", maintain_order=True):
        history: list[tuple[date, int]] = []
        last_selected: dict[str, date | None] = {CANDIDATE: None, CONTROL: None}
        for row in symbol_frame.iter_rows(named=True):
            notice_date = row["notice_date"]
            count = int(row["institution_count"])
            history = [
                (past_date, past_count)
                for past_date, past_count in history
                if 0 < (notice_date - past_date).days <= survey.LOOKBACK_DAYS
            ]
            prior_median = (
                float(statistics.median(value for _, value in history))
                if history
                else None
            )
            multiple = count / prior_median if prior_median and prior_median > 0 else None
            category = None
            if (
                survey.DEVELOPMENT_START <= notice_date <= survey.DEVELOPMENT_END
                and count >= survey.MIN_INSTITUTIONS
                and multiple is not None
            ):
                if multiple >= survey.MIN_ATTENTION_MULTIPLE:
                    category = CANDIDATE
                elif 1.0 <= multiple < survey.MIN_ATTENTION_MULTIPLE:
                    category = CONTROL
            last_date = last_selected.get(category) if category else None
            cooldown_clear = (
                category is not None
                and (
                    last_date is None
                    or (notice_date - last_date).days >= survey.COOLDOWN_DAYS
                )
            )
            if category and cooldown_clear:
                rows.append(
                    {
                        **row,
                        "ann_date": notice_date,
                        "category": category,
                        "prior_365d_institution_median": prior_median,
                        "attention_multiple": multiple,
                    }
                )
                last_selected[category] = notice_date
            history.append((notice_date, count))
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["category", "ann_date", "symbol"]
    )


def summarize(trades: pl.DataFrame) -> dict[str, Any]:
    result = survey.summarize(trades)
    result.pop("promotion_passed", None)
    return result


def evaluate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = results[CANDIDATE]
    control = results[CONTROL]
    candidate_excess = float(candidate.get("mean_excess_return") or -math.inf)
    control_excess = float(control.get("mean_excess_return") or -math.inf)
    tradable = int(candidate.get("tradable_events") or 0)
    unresolved = int(candidate.get("unresolved_exits") or 0)
    checks = {
        "at_least_1000_tradable_events": tradable >= 1_000,
        "at_least_500_announcement_days": int(candidate.get("announcement_days") or 0)
        >= 500,
        "tradable_rate_at_least_90pct": float(candidate.get("tradable_rate") or 0)
        >= 0.90,
        "benchmark_coverage_at_least_99pct": float(
            candidate.get("benchmark_coverage") or 0
        )
        >= 0.99,
        "entry_capacity_at_least_95pct": float(
            candidate.get("entry_capacity_feasible_rate") or 0
        )
        >= 0.95,
        "unresolved_exit_rate_at_most_1pct": (
            unresolved / tradable if tradable else math.inf
        )
        <= 0.01,
        "mean_net_return_at_least_1_5pct": float(
            candidate.get("mean_net_return") or -math.inf
        )
        >= 0.015,
        "mean_excess_return_at_least_1pct": candidate_excess >= 0.01,
        "cluster_t_at_least_3": float(
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 3.0,
        "at_least_5_positive_excess_years": int(
            candidate.get("positive_excess_years") or 0
        )
        >= 5,
        "max_year_positive_share_at_most_40pct": float(
            candidate.get("max_year_positive_excess_share") or math.inf
        )
        <= 0.40,
        "beats_ordinary_control_by_25bp": candidate_excess - control_excess >= 0.0025,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "candidate_excess_minus_control": candidate_excess - control_excess,
        "checks": checks,
        "failures": [name for name, passed_check in checks.items() if not passed_check],
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw = survey.load_survey_events(data_dir)
    events = classify_attention_events(raw)
    if events.is_empty():
        raise ValueError("main-board survey attention classification produced no events")
    panel = main_board.filter_main_board(
        survey.prepare_panel(
            survey.load_panel(data_dir, survey.PANEL_START, survey.PANEL_END)
        )
    )
    benchmark = survey.build_market_benchmark(panel, HOLD_TRADING_DAYS)
    results: dict[str, dict[str, Any]] = {}
    for category in (CANDIDATE, CONTROL):
        category_events = events.filter(pl.col("category") == category)
        trades = survey.build_trades(
            category_events,
            panel,
            HOLD_TRADING_DAYS,
            max_exit_delay=MAX_EXIT_DELAY,
        )
        results[category] = summarize(
            survey.attach_market_excess(trades, benchmark)
        )
    decision = evaluate(results)
    payload = {
        "schema_version": "p0-main-board-survey-attention-diffusion-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": survey.DEVELOPMENT_START,
            "end": survey.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "minimum_institutions": survey.MIN_INSTITUTIONS,
            "minimum_attention_multiple": survey.MIN_ATTENTION_MULTIPLE,
            "lookback_calendar_days": survey.LOOKBACK_DAYS,
            "cooldown_calendar_days_per_group": survey.COOLDOWN_DAYS,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
        },
        "data": {
            "raw_company_notice_events": raw.height,
            "classified_main_board_events": events.height,
            "classified_main_board_symbols": events.get_column("symbol").n_unique(),
            "counts_by_category": {
                row["category"]: row["len"]
                for row in events.group_by("category").len().to_dicts()
            },
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "results": results,
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
            "/app/data/research/p0_main_board_survey_attention_diffusion_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
