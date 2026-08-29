"""Compare causal entry features of historical and 2026 accumulation trades."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import analyze_recent_winner_features as winner_features
import run_reversal_study as common
import run_winner_pool_study as winner_study
from app.backtest.matrix import matrix_feature, valid_rolling_max, valid_shift
from strategies import independent_alpha_families as strategy_module

FEATURE_NAMES = (
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "rsi_14",
    "ma20_bias",
    "close_position",
    "vol_ratio_5d",
    "turnover_rate",
    "turnover_z_60d",
    "vol_price_corr_20d",
    "vwap_bias",
    "vol_trend_5_60",
    "boll_position",
    "annual_vol_20d",
    "ret_skew_20d",
    "limit_up_count_20d",
    "limit_up_count_60d",
    "industry_momentum_20d",
    "industry_breadth_5d",
    "roe_latest",
    "net_margin_latest",
    "revenue_yoy_latest",
    "debt_ratio_latest",
    "amihud_20d",
)


def _median(rows: list[dict], name: str) -> float | None:
    values = np.array([row["features"].get(name, np.nan) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return round(float(np.median(values)), 6) if len(values) else None


def _summary(rows: list[dict]) -> dict:
    return {
        "trades": len(rows),
        "mean_pnl": round(float(np.mean([row["pnl_pct"] for row in rows])), 6)
        if rows
        else None,
        "median_pnl": round(float(np.median([row["pnl_pct"] for row in rows])), 6)
        if rows
        else None,
        "feature_medians": {
            name: _median(rows, name)
            for name in (*FEATURE_NAMES, "prior_max_turnover_z_5d", "industry_breadth_acceleration_5d", "market_breadth", "market_above_ma20", "log_market_cap")
        },
        "top_industries": Counter(row["industry"] for row in rows).most_common(8),
    }


def run(data_dir: Path, research_dir: Path, trade_input: Path, output: Path) -> None:
    payload = json.loads(trade_input.read_text(encoding="utf-8"))
    historical = payload["periods"]["backward_oos_2014_2025"][
        "accumulation_trades"
    ]
    design = payload["periods"]["design_year_2026"]["accumulation_trades"]

    _, service = common._engine(data_dir, research_dir)
    loader, market, pit_context = winner_study._base_market(
        service, data_dir, winner_study.LONG_LOAD_START
    )
    try:
        labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
        date_to_row = {value: index for index, value in enumerate(labels)}
        symbol_to_column = {
            symbol: index for index, symbol in enumerate(market.symbols)
        }
        features = {name: matrix_feature(market, name) for name in FEATURE_NAMES}
        turnover_z = features["turnover_z_60d"]
        features["prior_max_turnover_z_5d"] = valid_rolling_max(
            valid_shift(turnover_z, 1),
            np.isfinite(valid_shift(turnover_z, 1)),
            5,
        )
        features["industry_breadth_acceleration_5d"] = (
            features["industry_breadth_5d"]
            - valid_shift(features["industry_breadth_5d"], 5)
        )
        market_breadth, market_above_ma20 = strategy_module._market_state(market)
        total_shares = matrix_feature(market, "total_shares")
        market_cap = market.close * total_shares
        log_market_cap = np.full(market.shape, np.nan, dtype=np.float32)
        np.log(market_cap, out=log_market_cap, where=market_cap > 0)
        features["log_market_cap"] = log_market_cap

        map_cache = {}

        def enrich(trade: dict) -> dict:
            signal_date = date.fromisoformat(trade["entry_signal_date"])
            row = date_to_row[signal_date]
            column = symbol_to_column[trade["symbol"]]
            if signal_date not in map_cache:
                map_cache[signal_date] = winner_features._name_industry_maps(
                    data_dir, signal_date
                )
            _, industry_map = map_cache[signal_date]
            snapshot = winner_features._snapshot(features, row, column)
            snapshot["market_breadth"] = winner_features._finite(market_breadth[row])
            snapshot["market_above_ma20"] = winner_features._finite(
                market_above_ma20[row]
            )
            return {
                "symbol": trade["symbol"],
                "name": trade["name"],
                "industry": industry_map.get(trade["symbol"], "未知"),
                "entry_signal_date": trade["entry_signal_date"],
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "pnl_pct": float(trade["pnl_pct"]),
                "exit_reason": trade["exit_reason"],
                "features": snapshot,
            }

        historical_rows = [enrich(trade) for trade in historical]
        design_rows = [enrich(trade) for trade in design]
        clusters = defaultdict(list)
        for row in design_rows:
            clusters[row["entry_signal_date"]].append(row)
        result = {
            "phase": "accumulation_trade_cohort_diagnosis",
            "point_in_time_context": pit_context,
            "cohorts": {
                "historical_winners": _summary(
                    [row for row in historical_rows if row["pnl_pct"] > 0]
                ),
                "historical_losers": _summary(
                    [row for row in historical_rows if row["pnl_pct"] <= 0]
                ),
                "historical_big_winners": _summary(
                    [row for row in historical_rows if row["pnl_pct"] >= 0.10]
                ),
                "design_2026_losses": _summary(design_rows),
            },
            "design_2026_trades": design_rows,
            "design_2026_signal_clusters": [
                {
                    "signal_date": signal_date,
                    "trades": len(rows),
                    "mean_pnl": round(
                        float(np.mean([row["pnl_pct"] for row in rows])), 6
                    ),
                    "symbols": [row["symbol"] for row in rows],
                }
                for signal_date, rows in sorted(clusters.items())
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        loader.compute_cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--trade-input",
        type=Path,
        default=Path("/app/data/research/barbell_portfolio_validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/accumulation_trade_cohorts.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.trade_input, args.output)


if __name__ == "__main__":
    main()
