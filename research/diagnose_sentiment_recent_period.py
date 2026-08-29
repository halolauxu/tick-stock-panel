"""Trade-level diagnosis for the recent sentiment-strategy drawdown."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common
from app.backtest.strategy import BacktestResultPolicy
from app.backtest.matrix import matrix_feature
from strategies.independent_alpha_families import _market_state

START = date(2026, 5, 27)
END = date(2026, 8, 26)
POLICY = BacktestResultPolicy(
    required_stats=None,
    include_monte_carlo=False,
    include_curves=True,
    include_trades=True,
    include_per_symbol_stats=False,
    include_return_distribution=False,
    include_benchmark=False,
    include_strategy_info=True,
)


def _trade_summary(rows: list[dict]) -> dict:
    pnl = [float(row["pnl_amount"]) for row in rows]
    returns = [float(row["pnl_pct"]) for row in rows]
    return {
        "trades": len(rows),
        "wins": sum(value > 0 for value in pnl),
        "pnl_amount": round(sum(pnl), 2),
        "mean_trade_return": round(float(np.mean(returns)) if returns else 0.0, 6),
        "median_trade_return": round(float(np.median(returns)) if returns else 0.0, 6),
    }


def _group(rows: list[dict], key) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(key(row))].append(row)
    return {
        name: _trade_summary(values)
        for name, values in sorted(grouped.items(), key=lambda item: item[0])
    }


def _industry_for_trade(trade: dict, memberships: dict) -> str:
    entry_date = date.fromisoformat(str(trade["entry_date"])[:10])
    rows = memberships.get((trade["symbol"],))
    if rows is None:
        return "未知"
    for row in rows.iter_rows(named=True):
        if row["in_date"] <= entry_date and (
            row["out_date"] is None or entry_date <= row["out_date"]
        ):
            return str(row["l1_name"])
    return "未知"


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                recent.WARMUP_START,
                recent.END,
                params={**study.BASELINE_PARAMS, "eligibility_mode": "none"},
                max_positions=10,
                max_hold_days=15,
                stop_loss=-0.06,
                basic_filter_override=study.PIT_FILTER,
            )
        ],
    )
    market = common._attach_industry_context(loader.market_data, data_dir)
    market, _ = common._attach_point_in_time_universe(market, data_dir)
    config = study._config(recent.SPECS["sentiment_risk_on"], START, END)
    prepared = common._prepared(service, [config], market)
    try:
        result = service.run(config, prepared=prepared, result_policy=POLICY)
        if result.error:
            raise RuntimeError(result.error)
        trades = result.trades
        label_to_row = {
            value[:10]: index for index, value in enumerate(market.timestamp_labels)
        }
        symbol_to_column = {
            symbol: index for index, symbol in enumerate(market.symbols)
        }
        feature_values = {
            name: matrix_feature(market, name)
            for name in (
                "momentum_60d",
                "rsi_14",
                "change_pct",
                "vol_ratio_5d",
                "industry_momentum_20d",
                "industry_breadth_5d",
            )
        }
        ma20 = matrix_feature(market, "ma20")
        for trade in trades:
            row_id = label_to_row[str(trade["entry_signal_date"])[:10]]
            column_id = symbol_to_column[trade["symbol"]]
            trade["entry_features"] = {
                name: round(float(values[row_id, column_id]), 6)
                for name, values in feature_values.items()
            }
            trade["entry_features"]["distance_above_ma20"] = round(
                float(market.close[row_id, column_id] / ma20[row_id, column_id] - 1.0),
                6,
            )
        membership = pl.read_parquet(
            data_dir / "research" / "sw_l1_membership.parquet"
        )
        memberships = membership.partition_by(
            "symbol", as_dict=True, include_key=False
        )
        for trade in trades:
            trade["industry"] = _industry_for_trade(trade, memberships)

        breadth, above_ma20 = _market_state(market)
        labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
        period_ids = [
            index for index, value in enumerate(labels) if START <= value <= END
        ]
        crossings = []
        defensive_days = []
        for index in period_ids:
            if index > 0 and above_ma20[index] >= 0.50 > above_ma20[index - 1]:
                crossings.append(
                    {
                        "date": labels[index].isoformat(),
                        "above_ma20": round(float(above_ma20[index]), 6),
                        "breadth": round(float(breadth[index]), 6),
                    }
                )
            if above_ma20[index] < 0.40 or breadth[index] < 0.30:
                defensive_days.append(labels[index].isoformat())

        batches = _group(trades, lambda row: row.get("entry_signal_date"))
        market_by_date = {
            labels[index].isoformat(): {
                "above_ma20": round(float(above_ma20[index]), 6),
                "breadth": round(float(breadth[index]), 6),
            }
            for index in period_ids
        }
        for signal_date, row in batches.items():
            row["market_state"] = market_by_date.get(signal_date)
            batch_trades = [
                trade
                for trade in trades
                if str(trade.get("entry_signal_date")) == signal_date
            ]
            row["entry_feature_medians"] = {
                name: round(
                    float(
                        np.median(
                            [trade["entry_features"][name] for trade in batch_trades]
                        )
                    ),
                    6,
                )
                for name in (
                    "momentum_60d",
                    "rsi_14",
                    "change_pct",
                    "vol_ratio_5d",
                    "industry_momentum_20d",
                    "industry_breadth_5d",
                    "distance_above_ma20",
                )
            }

        equity = result.equity_curve
        peak = float(equity[0]["value"])
        worst = None
        for row in equity:
            value = float(row["value"])
            peak = max(peak, value)
            drawdown = value / peak - 1.0
            if worst is None or drawdown < worst["drawdown"]:
                worst = {
                    "date": str(row["date"]),
                    "value": round(value, 2),
                    "drawdown": round(drawdown, 6),
                }

        payload = {
            "range": [START.isoformat(), END.isoformat()],
            "stats": result.stats,
            "trade_summary": _trade_summary(trades),
            "by_entry_signal_date": batches,
            "by_exit_reason": _group(trades, lambda row: row["exit_reason"]),
            "by_industry": _group(trades, lambda row: row["industry"]),
            "market_state": {
                "entry_crossings": crossings,
                "defensive_days": len(defensive_days),
                "first_defensive_day": defensive_days[0] if defensive_days else None,
                "mean_above_ma20": round(
                    float(np.mean(above_ma20[period_ids])), 6
                ),
                "mean_breadth": round(float(np.mean(breadth[period_ids])), 6),
            },
            "worst_drawdown": worst,
            "largest_losses": sorted(
                trades, key=lambda row: float(row["pnl_amount"])
            )[:10],
            "largest_wins": sorted(
                trades, key=lambda row: float(row["pnl_amount"]), reverse=True
            )[:10],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    finally:
        prepared.compute_cache.close()
        loader.compute_cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/sentiment_recent_period_diagnosis.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
