"""Ex-post cohort study of recent A-share winners using only causal input features."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import run_independent_alpha_study as study
import run_recent_year_alpha_study as recent
import run_reversal_study as common
from app.backtest.matrix import (
    build_basic_filter_mask,
    matrix_feature,
    valid_rolling_max,
    valid_shift,
)
from strategies import independent_alpha_families as strategy_module

START = date(2026, 5, 27)
END = date(2026, 8, 26)
SIGNAL_DATES = (date(2026, 7, 29), date(2026, 8, 20))
FEATURE_NAMES = (
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "rsi_14",
    "ma20_bias",
    "change_pct",
    "gap_return",
    "intraday_return",
    "close_position",
    "vol_ratio_5d",
    "amount_ratio_5d",
    "turnover_rate",
    "turnover_ratio_5d",
    "turnover_z_60d",
    "vol_price_corr_20d",
    "vwap_bias",
    "vol_trend_5_60",
    "boll_width",
    "boll_position",
    "macd_hist_pct",
    "annual_vol_20d",
    "atr_pct",
    "max_ret_20d",
    "ret_skew_20d",
    "up_days_20d",
    "limit_up_count_20d",
    "limit_up_count_60d",
    "consecutive_limit_ups",
    "industry_momentum_20d",
    "industry_breadth_5d",
    "roe_latest",
    "net_margin_latest",
    "revenue_yoy_latest",
    "debt_ratio_latest",
    "amihud_20d",
)


def _finite(value) -> float | None:
    value = float(value)
    return round(value, 6) if np.isfinite(value) else None


def _snapshot(features: dict[str, np.ndarray], row: int, column: int) -> dict:
    return {name: _finite(values[row, column]) for name, values in features.items()}


def _cohort_medians(
    features: dict[str, np.ndarray], row: int, mask: np.ndarray
) -> dict:
    output = {}
    for name, values in features.items():
        sample = values[row, mask]
        sample = sample[np.isfinite(sample)]
        output[name] = round(float(np.median(sample)), 6) if len(sample) else None
    return output


def _future_return(market, row: int, horizon: int) -> np.ndarray:
    target = min(row + horizon, market.shape[0] - 1)
    entry = market.open[row + 1]
    exit_ = market.close[target]
    result = np.full(market.shape[1], np.nan, dtype=np.float32)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_)
    np.divide(exit_, entry, out=result, where=valid)
    result[valid] -= np.float32(1.0)
    return result


def _name_industry_maps(data_dir: Path, as_of: date) -> tuple[dict, dict]:
    names = pl.read_parquet(data_dir / "research" / "historical_stock_names.parquet")
    names = names.filter(
        (pl.col("start_date") <= as_of)
        & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of))
    )
    name_map = dict(zip(names["symbol"].to_list(), names["name"].to_list(), strict=True))
    memberships = pl.read_parquet(data_dir / "research" / "sw_l1_membership.parquet")
    memberships = memberships.filter(
        (pl.col("in_date") <= as_of)
        & (pl.col("out_date").is_null() | (pl.col("out_date") >= as_of))
    )
    industry_map = dict(
        zip(
            memberships["symbol"].to_list(),
            memberships["l1_name"].to_list(),
            strict=True,
        )
    )
    return name_map, industry_map


def _quintile_profile(values: np.ndarray, returns: np.ndarray) -> list[dict]:
    valid = np.isfinite(values) & np.isfinite(returns)
    values = values[valid]
    returns = returns[valid]
    if len(values) < 50 or len(np.unique(values)) < 5:
        return []
    bounds = np.quantile(values, np.linspace(0.0, 1.0, 6))
    output = []
    for index in range(5):
        if index == 4:
            mask = (values >= bounds[index]) & (values <= bounds[index + 1])
        else:
            mask = (values >= bounds[index]) & (values < bounds[index + 1])
        bucket_returns = returns[mask]
        if not len(bucket_returns):
            continue
        output.append(
            {
                "bucket": index + 1,
                "lower": round(float(bounds[index]), 6),
                "upper": round(float(bounds[index + 1]), 6),
                "samples": int(mask.sum()),
                "mean_forward_20d": round(float(np.mean(bucket_returns)), 6),
                "median_forward_20d": round(float(np.median(bucket_returns)), 6),
                "hit_20pct": round(float(np.mean(bucket_returns >= 0.20)), 6),
            }
        )
    return output


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
    market, pit_context = common._attach_point_in_time_universe(market, data_dir)
    try:
        labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
        date_to_row = {value: index for index, value in enumerate(labels)}
        symbol_to_column = {
            symbol: index for index, symbol in enumerate(market.symbols)
        }
        symbols = np.asarray(market.symbols)
        main_board = np.array(
            [
                (symbol.endswith(".SH") and symbol.startswith("60"))
                or (
                    symbol.endswith(".SZ")
                    and symbol.startswith(("000", "001", "002", "003"))
                )
                for symbol in symbols
            ],
            dtype=bool,
        )
        basic = build_basic_filter_mask(market, strategy_module.META["basic_filter"])
        eligible = matrix_feature(market, "pit_eligible") > np.float32(0.5)
        features = {name: matrix_feature(market, name) for name in FEATURE_NAMES}
        industry_momentum = features["industry_momentum_20d"]
        industry_breadth = features["industry_breadth_5d"]
        features["industry_momentum_acceleration_5d"] = (
            industry_momentum - valid_shift(industry_momentum, 5)
        )
        features["industry_breadth_acceleration_5d"] = (
            industry_breadth - valid_shift(industry_breadth, 5)
        )
        turnover_z = features["turnover_z_60d"]
        amount_ratio = features["amount_ratio_5d"]
        features["prior_max_turnover_z_5d"] = valid_rolling_max(
            valid_shift(turnover_z, 1),
            np.isfinite(valid_shift(turnover_z, 1)),
            5,
        )
        features["prior_max_amount_ratio_5d"] = valid_rolling_max(
            valid_shift(amount_ratio, 1),
            np.isfinite(valid_shift(amount_ratio, 1)),
            5,
        )
        total_shares = matrix_feature(market, "total_shares")
        market_cap = market.close * total_shares
        log_market_cap = np.full(market.shape, np.nan, dtype=np.float32)
        np.log(market_cap, out=log_market_cap, where=market_cap > 0)
        features["log_market_cap"] = log_market_cap

        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        candidate_mask = (
            (market.close > ma20)
            & (ma20 > ma60)
            & (features["momentum_60d"] >= np.float32(0.10))
            & eligible
            & basic
            & main_board[None, :]
        )
        anti_chase_mask = (
            candidate_mask
            & (features["momentum_60d"] <= np.float32(0.50))
            & (features["ma20_bias"] <= np.float32(0.15))
            & (features["rsi_14"] <= np.float32(70.0))
        )

        selected_by_signal = {
            date(2026, 7, 29): {
                "600012.SH", "603580.SH", "001965.SZ", "000429.SZ", "002832.SZ",
                "601156.SH", "600428.SH", "601919.SH", "000759.SZ", "600882.SH",
            },
            date(2026, 8, 20): {
                "605179.SH", "600127.SH", "603823.SH", "002674.SZ", "002827.SZ",
                "603580.SH", "603065.SH", "600664.SH", "000048.SZ", "600206.SH",
            },
        }
        signal_analysis = {}
        for signal_date in SIGNAL_DATES:
            row = date_to_row[signal_date]
            horizon = min(10, date_to_row[END] - row)
            forward = _future_return(market, row, horizon)
            pool = candidate_mask[row] & np.isfinite(forward)
            repaired_pool = anti_chase_mask[row] & np.isfinite(forward)
            selected_columns = np.array(
                [symbol_to_column[symbol] for symbol in selected_by_signal[signal_date]],
                dtype=np.int32,
            )
            selected = np.zeros(market.shape[1], dtype=bool)
            selected[selected_columns] = True
            selected &= np.isfinite(forward)
            pool_returns = forward[pool]
            upper = float(np.quantile(pool_returns, 0.8))
            lower = float(np.quantile(pool_returns, 0.2))
            winners = pool & (forward >= upper)
            losers = pool & (forward <= lower)
            top_columns = np.where(pool)[0][np.argsort(pool_returns)[-15:][::-1]]
            name_map, industry_map = _name_industry_maps(data_dir, signal_date)
            signal_analysis[signal_date.isoformat()] = {
                "horizon_sessions": horizon,
                "candidate_pool": int(pool.sum()),
                "anti_chase_pool": int(repaired_pool.sum()),
                "selected_mean_return": round(float(np.mean(forward[selected])), 6),
                "pool_mean_return": round(float(np.mean(pool_returns)), 6),
                "winner_cutoff": round(upper, 6),
                "loser_cutoff": round(lower, 6),
                "selected_medians": _cohort_medians(features, row, selected),
                "winner_medians": _cohort_medians(features, row, winners),
                "loser_medians": _cohort_medians(features, row, losers),
                "unselected_winners": [
                    {
                        "symbol": str(symbols[column]),
                        "name": name_map.get(str(symbols[column]), ""),
                        "industry": industry_map.get(str(symbols[column]), "未知"),
                        "forward_return": round(float(forward[column]), 6),
                        "features": _snapshot(features, row, column),
                    }
                    for column in top_columns
                    if not selected[column]
                ][:10],
            }

        start_row = date_to_row[START]
        end_row = date_to_row[END]
        full_period_return = np.full(market.shape[1], np.nan, dtype=np.float32)
        valid_period = (
            np.isfinite(market.close[start_row])
            & (market.close[start_row] > 0)
            & np.isfinite(market.close[end_row])
        )
        np.divide(
            market.close[end_row],
            market.close[start_row],
            out=full_period_return,
            where=valid_period,
        )
        full_period_return[valid_period] -= np.float32(1.0)
        start_universe = (
            eligible[start_row]
            & basic[start_row]
            & main_board
            & np.isfinite(full_period_return)
        )
        threshold = float(np.quantile(full_period_return[start_universe], 0.90))
        top_period = start_universe & (full_period_return >= threshold)
        period_name_map, period_industry_map = _name_industry_maps(data_dir, START)
        universe_industries = Counter(
            period_industry_map.get(str(symbols[column]), "未知")
            for column in np.where(start_universe)[0]
        )
        winner_industries = Counter(
            period_industry_map.get(str(symbols[column]), "未知")
            for column in np.where(top_period)[0]
        )
        industry_enrichment = sorted(
            (
                {
                    "industry": industry,
                    "universe": universe_industries[industry],
                    "winners": winner_industries[industry],
                    "winner_rate": round(
                        winner_industries[industry] / universe_industries[industry], 6
                    ),
                    "lift_vs_universe": round(
                        (winner_industries[industry] / universe_industries[industry])
                        / (int(top_period.sum()) / int(start_universe.sum())),
                        6,
                    ),
                }
                for industry in universe_industries
                if winner_industries[industry] >= 3
            ),
            key=lambda row: (row["lift_vs_universe"], row["winners"]),
            reverse=True,
        )
        top_period_columns = np.where(top_period)[0][
            np.argsort(full_period_return[top_period])[-30:][::-1]
        ]

        sample_rows = [
            index
            for index in range(start_row, end_row - 20)
            if labels[index].weekday() < 5
        ]
        sample_masks = [
            eligible[row]
            & basic[row]
            & main_board
            & np.isfinite(_future_return(market, row, 20))
            for row in sample_rows
        ]
        forward_samples = np.concatenate(
            [_future_return(market, row, 20)[mask] for row, mask in zip(sample_rows, sample_masks, strict=True)]
        )
        feature_profiles = {}
        for name, values in features.items():
            samples = np.concatenate(
                [values[row, mask] for row, mask in zip(sample_rows, sample_masks, strict=True)]
            )
            profile = _quintile_profile(samples, forward_samples)
            if not profile:
                continue
            feature_profiles[name] = {
                "quintiles": profile,
                "best_bucket": max(
                    profile, key=lambda item: item["mean_forward_20d"]
                )["bucket"],
                "spread_best_minus_worst": round(
                    max(item["mean_forward_20d"] for item in profile)
                    - min(item["mean_forward_20d"] for item in profile),
                    6,
                ),
            }

        ranked_features = sorted(
            feature_profiles,
            key=lambda name: feature_profiles[name]["spread_best_minus_worst"],
            reverse=True,
        )
        payload = {
            "phase": "recent_winner_feature_cohort_study",
            "range": [START.isoformat(), END.isoformat()],
            "point_in_time_context": pit_context,
            "method": {
                "feature_time": "signal date only",
                "labels": "ex-post returns used only for diagnosis",
                "rolling_target": "next open to 20th-session close",
            },
            "signal_date_analysis": signal_analysis,
            "full_period_top_decile": {
                "universe": int(start_universe.sum()),
                "top_decile_cutoff": round(threshold, 6),
                "winner_medians": _cohort_medians(features, start_row, top_period),
                "universe_medians": _cohort_medians(features, start_row, start_universe),
                "industry_enrichment": industry_enrichment,
                "top_30": [
                    {
                        "symbol": str(symbols[column]),
                        "name": period_name_map.get(str(symbols[column]), ""),
                        "industry": period_industry_map.get(str(symbols[column]), "未知"),
                        "period_return": round(float(full_period_return[column]), 6),
                        "features_at_start": _snapshot(features, start_row, column),
                    }
                    for column in top_period_columns
                ],
            },
            "rolling_20d_feature_profiles": feature_profiles,
            "ranked_features": ranked_features,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "signal_date_analysis": signal_analysis,
                    "full_period_top_decile": payload["full_period_top_decile"],
                    "top_ranked_features": ranked_features[:15],
                    "top_feature_profiles": {
                        name: feature_profiles[name] for name in ranked_features[:15]
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
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
        default=Path("/app/data/research/recent_winner_feature_analysis.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
