"""Extend the frozen ETF rotation rules over the point-in-time full universe."""

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

import collect_p0_dynamic_etf_rotation_data as collector  # noqa: E402
import run_p0_etf_dual_momentum_development as dual  # noqa: E402
import run_p0_microcap_defensive_etf_rotation_discovery as overlay  # noqa: E402

START = date(2014, 1, 1)
END = collector.END
CAPITALS = (200_000.0, 1_000_000.0)
VARIANT_IDS = (
    "absolute_switch",
    "relative_rotation",
    "microcap_70_defensive_30",
)


def monthly_schedule(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
        )
        .group_by("month", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .filter(pl.col("signal_date").is_between(START, END, closed="both"))
    )


def run_monthly_engine(
    panel: pl.DataFrame,
) -> dict[str, Any]:
    old_start = dual.DEVELOPMENT_START
    old_end = dual.DEVELOPMENT_END
    dual.DEVELOPMENT_START = START
    dual.DEVELOPMENT_END = END
    try:
        schedule = monthly_schedule(panel)
        candidates = dual.build_candidates(panel, schedule)
        action_dates = schedule.get_column("entry_date").to_list()
        all_dates = (
            panel.filter(pl.col("date").is_between(START, END, closed="both"))
            .get_column("date")
            .unique()
            .sort()
            .to_list()
        )
        accounts = {
            f"cny_{int(capital)}": dual.simulate(
                candidates,
                panel,
                all_dates,
                action_dates,
                capital,
            )
            for capital in CAPITALS
        }
        benchmark = dual.benchmark_metrics(panel)
    finally:
        dual.DEVELOPMENT_START = old_start
        dual.DEVELOPMENT_END = old_end
    return {
        "data": {
            "signal_rows": candidates.height,
            "signal_symbols": candidates.get_column("symbol").n_unique(),
            "scheduled_rebalances": len(action_dates),
            "active_rebalances": candidates.get_column("entry_date").n_unique(),
        },
        "benchmark": benchmark,
        "accounts": accounts,
        "gate": evaluate_monthly_gate(accounts),
    }


def _year_return(metrics: dict[str, Any], year: int) -> float | None:
    return next(
        (
            row.get("account_return", row.get("return"))
            for row in metrics["yearly"]
            if row["year"] == year
        ),
        None,
    )


def evaluate_monthly_gate(
    accounts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for name, result in accounts.items():
        metrics = result["metrics"]
        checks[f"{name}_2026_at_least_25pct"] = (_year_return(metrics, 2026) or -math.inf) >= 0.25
        checks[f"{name}_drawdown_no_worse_than_30pct"] = (
            metrics.get("max_drawdown") or -math.inf
        ) >= -0.30
        checks[f"{name}_cash_reconciled"] = (
            result["integrity"]["max_cash_reconciliation_error"] <= 0.01
        )
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": "INDEPENDENT_ENGINE_ELIGIBLE" if passed else "INSUFFICIENT",
        "checks": checks,
    }


def weekly_etf_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.select(
        "symbol",
        "date",
        pl.col("open").alias("adjusted_open"),
        pl.col("close").alias("adjusted_close"),
        "volume",
        "amount",
        "momentum_120d",
        "mean_amount_20d",
        "listing_days",
    )


def build_weekly_best_etf(panel: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Freeze rank on signal data; never rerank with future exit availability."""
    ranked = (
        panel.join(schedule, on="date", how="inner")
        .filter(
            (pl.col("listing_days") >= dual.MIN_LISTING_DAYS)
            & (pl.col("momentum_120d") > 0)
            & (pl.col("mean_amount_20d") >= overlay.MIN_MEAN_AMOUNT)
        )
        .sort(
            ["date", "momentum_120d", "mean_amount_20d", "symbol"],
            descending=[False, True, True, False],
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("rank"))
        .filter(pl.col("rank") == 1)
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
        pl.col("volume").alias("exit_volume"),
    )
    end_marks = panel.filter(pl.col("date") == pl.lit(END)).select(
        "symbol", pl.col("adjusted_close").alias("end_mark")
    )
    return (
        ranked.join(entry, on=["symbol", "entry_date"], how="left")
        .join(exit_quotes, on=["symbol", "exit_date"], how="left")
        .join(end_marks, on="symbol", how="left")
        .with_columns(
            (
                (pl.col("entry_open") > 0)
                & (pl.col("entry_volume") > 0)
                & (pl.col("entry_amount") >= overlay.MIN_MEAN_AMOUNT)
            )
            .fill_null(False)
            .alias("entry_executable"),
            ((pl.col("exit_open") > 0) & (pl.col("exit_volume") > 0))
            .fill_null(False)
            .alias("exit_executable"),
            (pl.col("exit_date").is_null() & (pl.col("end_mark") > 0).fill_null(False)).alias(
                "marked_at_end"
            ),
        )
        .with_columns(
            pl.when(~pl.col("entry_executable"))
            .then(0.0)
            .when(pl.col("marked_at_end"))
            .then(
                pl.col("end_mark")
                / (
                    pl.col("entry_open")
                    * (1.0 + overlay.baseline.COMMISSION_PCT + overlay.baseline.SLIPPAGE_PCT)
                )
                - 1.0
            )
            .when(~pl.col("exit_executable"))
            .then(-1.0)
            .otherwise(
                (
                    pl.col("exit_open")
                    * (1.0 - overlay.baseline.COMMISSION_PCT - overlay.baseline.SLIPPAGE_PCT)
                )
                / (
                    pl.col("entry_open")
                    * (1.0 + overlay.baseline.COMMISSION_PCT + overlay.baseline.SLIPPAGE_PCT)
                )
                - 1.0
            )
            .alias("etf_return")
        )
        .select(
            "date",
            "symbol",
            pl.col("momentum_120d").alias("etf_momentum_120d"),
            "etf_return",
            "entry_executable",
            "exit_executable",
            "marked_at_end",
        )
    )


def evaluate_overlay_gate(
    control: dict[str, Any], variants: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    control_drawdown = control["metrics"]["max_drawdown"]
    checks: dict[str, dict[str, bool]] = {}
    for name, result in variants.items():
        metrics = result["metrics"]
        yearly = {row["year"]: row["return"] for row in metrics["yearly"]}
        checks[name] = {
            "2026_strictly_above_30pct": (yearly.get(2026) is not None and yearly[2026] > 0.30),
            "every_2014_2025_year_positive": all(
                yearly.get(year) is not None and yearly[year] > 0 for year in range(2014, 2026)
            ),
            "drawdown_no_worse_than_control": (metrics["max_drawdown"] >= control_drawdown),
        }
    promoted = [name for name, row in checks.items() if all(row.values())]
    return {
        "passed": bool(promoted),
        "verdict": (
            "PROMOTE_TO_ACCOUNT_CONFIRMATION" if promoted else "TERMINATE_EXPANDED_ETF_ROTATION"
        ),
        "promoted": promoted,
        "checks": checks,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = data_dir / "research" / "p0_dynamic_etf_rotation_data_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "DATA_QUALIFIED":
        raise ValueError("dynamic ETF data did not pass audit")
    root = data_dir / "research" / "dynamic_etf_rotation_v1"
    master = pl.read_parquet(root / "master.parquet")
    daily = pl.read_parquet(root / "daily_raw.parquet")
    adjustments = pl.read_parquet(root / "adjustments.parquet")

    # Build the stock leg first so its large source matrix can be released before
    # the ETF matrix is materialized.
    microcap = overlay.build_microcap_weekly(data_dir)
    microcap_schedule = microcap.select("date", "entry_date", "exit_date")
    gc.collect()
    panel = dual.prepare_panel(daily, adjustments, master)
    del daily, adjustments
    gc.collect()
    monthly = run_monthly_engine(panel)
    best_etf = build_weekly_best_etf(weekly_etf_panel(panel), microcap_schedule)
    frames = overlay.build_variant_returns(microcap, best_etf)
    control = overlay.summarize(
        microcap.select(
            "date",
            "entry_date",
            pl.col("microcap_return").alias("weekly_return"),
        ).with_columns(pl.lit("microcap").alias("selected_asset"))
    )
    variants = {name: overlay.summarize(frames[name]) for name in VARIANT_IDS}
    selection = (
        best_etf.group_by("symbol")
        .len()
        .sort("len", descending=True)
        .join(master.select("symbol", "name"), on="symbol", how="left")
        .head(30)
        .to_dicts()
    )
    weekly = {
        "data": {
            "weeks": microcap.height,
            "etf_selected_weeks": best_etf.height,
            "etf_selected_symbols": best_etf.get_column("symbol").n_unique(),
            "entry_rejections": best_etf.filter(~pl.col("entry_executable")).height,
            "missing_historical_exits": best_etf.filter(
                pl.col("entry_executable") & ~pl.col("exit_executable") & ~pl.col("marked_at_end")
            ).height,
            "end_marked_positions": best_etf.filter(pl.col("marked_at_end")).height,
            "top_selected_etfs": selection,
        },
        "control": control,
        "variants": variants,
        "gate": evaluate_overlay_gate(control, variants),
    }
    payload = {
        "schema_version": "p0-dynamic-etf-rotation-extension-v1",
        "contract_frozen": "2026-09-03",
        "period": {"start": START, "end": END},
        "data_audit": {
            "path": str(audit_path),
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "counts": audit["counts"],
        },
        "monthly_dual_momentum": monthly,
        "weekly_microcap_overlays": weekly,
        "decision": weekly["gate"],
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
                "monthly": {
                    "accounts": {
                        name: {
                            "metrics": result["metrics"],
                            "execution": result["execution"],
                            "integrity": result["integrity"],
                        }
                        for name, result in monthly["accounts"].items()
                    },
                    "gate": monthly["gate"],
                },
                "weekly": weekly,
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
        default=Path("/app/data/research/p0_dynamic_etf_rotation_extension_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
