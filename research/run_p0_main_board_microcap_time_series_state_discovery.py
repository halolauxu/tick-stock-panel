"""Screen frozen weekly momentum/reversal states on the micro-cap base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_microcap_defensive_etf_rotation_discovery as rotation  # noqa: E402

START = date(2014, 1, 1)
END = rotation.END
VARIANT_IDS = (
    "previous_week_reversal",
    "previous_week_momentum",
    "four_week_reversal",
    "four_week_momentum",
)


def build_state_variants(microcap: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = microcap.sort("entry_date").select(
        "date", "entry_date", "exit_date", "microcap_return"
    ).to_dicts()
    completed: deque[float] = deque(maxlen=4)
    outputs: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANT_IDS
    }
    for row in rows:
        one_ready = len(completed) >= 1
        four_ready = len(completed) >= 4
        previous = completed[-1] if one_ready else 0.0
        trailing_four = (
            float(baseline._compound(list(completed))) if four_ready else 0.0
        )
        decisions = {
            "previous_week_reversal": one_ready and previous <= 0,
            "previous_week_momentum": one_ready and previous > 0,
            "four_week_reversal": four_ready and trailing_four <= 0,
            "four_week_momentum": four_ready and trailing_four > 0,
        }
        for variant, active in decisions.items():
            outputs[variant].append(
                {
                    "date": row["date"],
                    "entry_date": row["entry_date"],
                    "weekly_return": (
                        float(row["microcap_return"]) if active else 0.0
                    ),
                    "selected_asset": "microcap" if active else "cash",
                }
            )
        completed.append(float(row["microcap_return"]))
    return {variant: pl.DataFrame(rows) for variant, rows in outputs.items()}


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
    return {
        "metrics": {
            "annualized": baseline._annualized(values),
            "total_return": baseline._compound(values),
            "max_drawdown": baseline._max_drawdown(values),
            "yearly": yearly,
        },
        "active_weeks": work.filter(
            pl.col("selected_asset") == "microcap"
        ).height,
        "cash_weeks": work.filter(pl.col("selected_asset") == "cash").height,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    microcap = rotation.build_microcap_weekly(data_dir)
    variants = build_state_variants(microcap)
    control = rotation.summarize(
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
        "schema_version": "p0-main-board-microcap-time-series-state-v1",
        "contract_frozen": "2026-09-02",
        "research_class": "known_full_history_mechanism_discovery",
        "period": {"start": START, "end": END},
        "assumptions": {
            "cash_return": 0.0,
            "signal_uses_completed_weeks_only": True,
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
            "p0_main_board_microcap_time_series_state_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
