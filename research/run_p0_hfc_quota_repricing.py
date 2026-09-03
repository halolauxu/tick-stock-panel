"""Run the preregistered one-year HFC quota-holder repricing account."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import collect_p0_hfc_quota_evidence as evidence  # noqa: E402
import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_forecast_drift_development as forecast  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402


END = date(2026, 8, 26)
DEVELOPMENT_END = date(2026, 4, 30)
HOLD_TRADING_DAYS = 5
INITIAL_CASH = 200_000.0
TARGET_EXPOSURE = 0.80
CANDIDATE_SYMBOLS = tuple(evidence.ISSUER_MAPPINGS)
CONTROL_SYMBOLS = evidence.CONTROL_SYMBOLS
MAIN_BOARD_PATTERN = r"^(?:600|601|603|605|000|001|002|003)\d{3}\.(?:SH|SZ)$"


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def load_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"HFC quota evidence is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != "PASS":
        raise ValueError("HFC quota evidence gate did not pass")
    event = payload.get("activation_event") or {}
    if event.get("symbol") != "605020.SH" or event.get("ann_date") != "2025-10-09":
        raise ValueError("frozen activation event changed")
    mapped = {
        row["symbol"]
        for row in payload.get("issuer_mapping", [])
        if row.get("quota_year") == 2025 and row.get("passed")
    }
    if mapped != set(CANDIDATE_SYMBOLS):
        raise ValueError("frozen candidate mapping is incomplete or changed")
    return payload


def load_stock_panel(data_dir: Path, activation: date) -> pl.DataFrame:
    symbols = [*CANDIDATE_SYMBOLS, *CONTROL_SYMBOLS]
    panel = forecast.load_panel(
        data_dir,
        start=activation,
        panel_end=END,
    ).filter(pl.col("symbol").is_in(symbols))
    present = set(panel.get_column("symbol").unique().to_list())
    if present != set(symbols):
        raise ValueError(f"stock panel missing frozen symbols: {sorted(set(symbols) - present)}")
    if panel.filter(pl.col("symbol").str.contains(MAIN_BOARD_PATTERN).not_()).height:
        raise ValueError("non-main-board symbol entered the frozen universe")
    return panel.sort(["symbol", "date"])


def build_schedule(
    panel: pl.DataFrame, activation: date
) -> tuple[list[date], list[date], date]:
    dates = (
        panel.filter(pl.col("date") > activation)
        .select("date")
        .unique()
        .sort("date")
        .get_column("date")
        .to_list()
    )
    if len(dates) <= HOLD_TRADING_DAYS:
        raise ValueError("not enough post-activation trading sessions")
    entry_dates = dates[::HOLD_TRADING_DAYS]
    entry_dates = [
        day
        for day in entry_dates
        if dates.index(day) + HOLD_TRADING_DAYS < len(dates)
    ]
    if not entry_dates:
        raise ValueError("no complete five-session cohort")
    final_exit = dates[dates.index(entry_dates[-1]) + HOLD_TRADING_DAYS]
    daily_dates = [day for day in dates if entry_dates[0] <= day <= final_exit]
    return entry_dates, daily_dates, final_exit


def build_candidates(
    panel: pl.DataFrame,
    symbols: tuple[str, ...],
    entry_dates: list[date],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry_date in entry_dates:
        prior = (
            panel.filter(
                pl.col("symbol").is_in(symbols) & (pl.col("date") < entry_date)
            )
            .sort(["symbol", "date"])
            .group_by("symbol", maintain_order=True)
            .tail(1)
        )
        amounts = {
            row["symbol"]: float(row.get("amount") or 0.0)
            for row in prior.to_dicts()
        }
        for rank, symbol in enumerate(symbols, start=1):
            rows.append(
                {
                    "date": entry_date,
                    "entry_date": entry_date,
                    "symbol": symbol,
                    "signal_amount": amounts.get(symbol, 0.0),
                    "cap_rank": rank,
                    "family": "hfc_quota_holder",
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["entry_date", "cap_rank", "symbol"]
    )


def simulate_basket(
    data_dir: Path,
    panel: pl.DataFrame,
    symbols: tuple[str, ...],
    entry_dates: list[date],
    daily_dates: list[date],
    final_exit: date,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    candidates = build_candidates(panel, symbols, entry_dates)
    quotes = account.prepare_quote_panel(
        panel.filter(pl.col("symbol").is_in(symbols))
    )
    action_dates = [*entry_dates, final_exit]
    grid = daily.build_action_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=len(symbols),
        action_dates=action_dates,
        force_rebalance_dates=set(entry_dates[1:]),
        allow_same_day_reentry=True,
        cost_multiplier=cost_multiplier,
        target_exposure_by_date={day: TARGET_EXPOSURE for day in action_dates},
    )
    account_daily, stale = account.build_daily_equity(
        simulation,
        quotes,
        daily_dates,
        initial_cash=INITIAL_CASH,
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    return {
        "metrics": {
            "trading_days": len(returns),
            "cohorts": len(entry_dates),
            "total_return": baseline._compound(returns),
            "annualized": shared._annualized(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
        },
        "periods": period_metrics(account_daily),
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
            "ending_positions": len(simulation["ending_positions"]),
        },
        "account": account.account_summary(simulation, account_daily),
        "daily": account_daily,
    }


def period_metrics(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for label, expression in {
        "development": pl.col("date") <= DEVELOPMENT_END,
        "validation": pl.col("date") > DEVELOPMENT_END,
    }.items():
        scoped = frame.filter(expression)
        returns = scoped.get_column("daily_return").drop_nulls().to_list()
        output[label] = {
            "trading_days": len(returns),
            "total_return": baseline._compound(returns),
            "annualized": shared._annualized(returns),
            "max_drawdown": baseline._max_drawdown(returns),
        }
    return output


def load_csi300(data_dir: Path, daily_dates: list[date]) -> dict[str, Any]:
    paths = sorted((data_dir / "kline_index_enriched").glob("date=*/part.parquet"))
    if not paths:
        paths = sorted((data_dir / "kline_index_daily").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("CSI 300 daily data is required")
    start, end = daily_dates[0], daily_dates[-1]
    frame = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("symbol") == "000300.SH")
            & pl.col("date").is_between(start, end, closed="both")
        )
        .select("date", "close")
        .sort("date")
        .collect(engine="streaming")
    )
    if frame.height != len(daily_dates):
        raise ValueError(
            f"CSI 300 coverage mismatch: {frame.height} != {len(daily_dates)}"
        )
    frame = frame.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("daily_return")
    )
    returns = frame.get_column("daily_return").drop_nulls().to_list()
    return {
        "trading_days": len(returns),
        "total_return": baseline._compound(returns),
        "annualized": shared._annualized(returns),
        "max_drawdown": baseline._max_drawdown(returns),
        "periods": period_metrics(
            frame.with_columns(
                (INITIAL_CASH * (1.0 + pl.col("daily_return").fill_null(0.0)).cum_prod())
                .alias("equity")
            )
        ),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def evaluate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    stressed: dict[str, Any],
    stressed_control: dict[str, Any],
    csi300: dict[str, Any],
) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    data_checks = {
        "at_least_20_complete_cohorts": candidate_metrics["cohorts"] >= 20,
        "candidate_buy_execution_at_least_90pct": candidate["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "candidate_sell_execution_at_least_90pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "control_buy_execution_at_least_90pct": control["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.90,
        "control_sell_execution_at_least_90pct": control["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.90,
        "candidate_no_unresolved_positions": candidate["integrity"][
            "ending_positions"
        ]
        == 0,
        "control_no_unresolved_positions": control["integrity"]["ending_positions"]
        == 0,
        "candidate_cash_reconciles": candidate["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 1e-6,
        "control_cash_reconciles": control["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 1e-6,
    }
    alpha_checks = {
        "candidate_annualized_at_least_15pct": (
            candidate_metrics["annualized"] or -math.inf
        )
        >= 0.15,
        "candidate_drawdown_no_worse_than_25pct": (
            candidate_metrics["max_drawdown"] or -math.inf
        )
        >= -0.25,
        "candidate_control_annualized_excess_at_least_8pp": (
            _difference(
                candidate_metrics["annualized"], control_metrics["annualized"]
            )
            or -math.inf
        )
        >= 0.08,
        "candidate_csi300_annualized_excess_at_least_10pp": (
            _difference(candidate_metrics["annualized"], csi300["annualized"])
            or -math.inf
        )
        >= 0.10,
        "development_excess_positive": (
            _difference(
                candidate["periods"]["development"]["total_return"],
                control["periods"]["development"]["total_return"],
            )
            or -math.inf
        )
        > 0,
        "validation_excess_positive": (
            _difference(
                candidate["periods"]["validation"]["total_return"],
                control["periods"]["validation"]["total_return"],
            )
            or -math.inf
        )
        > 0,
        "double_friction_excess_positive": (
            _difference(
                stressed["metrics"]["annualized"],
                stressed_control["metrics"]["annualized"],
            )
            or -math.inf
        )
        > 0,
    }
    if not all(data_checks.values()):
        decision = "DATA_GAP"
    elif all(alpha_checks.values()):
        decision = "ADMIT_FORWARD"
    else:
        decision = "REJECT"
    return {
        "data_checks": data_checks,
        "alpha_checks": alpha_checks,
        "candidate_control_annualized_excess": _difference(
            candidate_metrics["annualized"], control_metrics["annualized"]
        ),
        "candidate_csi300_annualized_excess": _difference(
            candidate_metrics["annualized"], csi300["annualized"]
        ),
        "decision": decision,
    }


def public_account(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "daily"}


def run(data_dir: Path, evidence_path: Path, output: Path) -> dict[str, Any]:
    evidence_payload = load_evidence(evidence_path)
    activation = date.fromisoformat(
        evidence_payload["activation_event"]["ann_date"]
    )
    panel = load_stock_panel(data_dir, activation)
    entry_dates, daily_dates, final_exit = build_schedule(panel, activation)
    candidate = simulate_basket(
        data_dir,
        panel,
        CANDIDATE_SYMBOLS,
        entry_dates,
        daily_dates,
        final_exit,
    )
    control = simulate_basket(
        data_dir,
        panel,
        CONTROL_SYMBOLS,
        entry_dates,
        daily_dates,
        final_exit,
    )
    stressed = simulate_basket(
        data_dir,
        panel,
        CANDIDATE_SYMBOLS,
        entry_dates,
        daily_dates,
        final_exit,
        cost_multiplier=2.0,
    )
    stressed_control = simulate_basket(
        data_dir,
        panel,
        CONTROL_SYMBOLS,
        entry_dates,
        daily_dates,
        final_exit,
        cost_multiplier=2.0,
    )
    csi300 = load_csi300(data_dir, daily_dates)
    decision = evaluate(candidate, control, stressed, stressed_control, csi300)
    payload = {
        "schema_version": "p0-hfc-quota-repricing-v1",
        "contract": "docs/p0-hfc-quota-repricing-contract.md",
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "activation_event": evidence_payload["activation_event"],
        "account_contract": {
            "initial_cash": INITIAL_CASH,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "target_gross_exposure": TARGET_EXPOSURE,
            "candidate_symbols": list(CANDIDATE_SYMBOLS),
            "control_symbols": list(CONTROL_SYMBOLS),
            "first_entry": entry_dates[0],
            "last_entry": entry_dates[-1],
            "final_exit": final_exit,
            "development_end": DEVELOPMENT_END,
            "validation_start": date(2026, 5, 1),
            "validation_end": END,
        },
        "candidate": public_account(candidate),
        "control": public_account(control),
        "double_friction": {
            "candidate": public_account(stressed),
            "control": public_account(stressed_control),
        },
        "csi300": csi300,
        "evaluation": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "control": payload["control"],
                "double_friction": payload["double_friction"],
                "csi300": csi300,
                "evaluation": decision,
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("/app/data/research/p0_hfc_quota_evidence_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_hfc_quota_repricing_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.evidence, args.output)


if __name__ == "__main__":
    main()
