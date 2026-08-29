"""Cross-family A-share strategy study on the corrected 2013-2026 dataset."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import date
from pathlib import Path

import numpy as np

import run_reversal_study as common
from app.backtest.matrix import matrix_feature

START = date(2015, 1, 1)
END = date(2026, 8, 26)

FOLDS = tuple(
    (
        f"{year}{half}",
        date(year, 1 if half == "H1" else 7, 1),
        (
            date(year, 6, 30)
            if half == "H1"
            else min(date(year, 12, 31), END)
        ),
    )
    for year in range(2015, 2027)
    for half in ("H1", "H2")
    if date(year, 1 if half == "H1" else 7, 1) <= END
)

PIT_FILTER = {**common.RESEARCH_BASIC_FILTER, "exclude_st": False}
BASELINE_PARAMS = {
    **common.COMMON,
    "family": "baseline_ranked",
    "score_mode": "baseline",
    "eligibility_mode": "pit",
}

CANDIDATES = {
    "reversal_recovery_p15": {
        "strategy_id": "reversal_first_principles",
        "params": {
            **common.COMMON,
            "family": "baseline_ranked",
            "score_mode": "recovery",
            "eligibility_mode": "pit",
        },
        "execution": {"max_positions": 15, "max_hold_days": 15, "stop_loss": -0.06},
        "mechanism": "超跌反转后按修复质量排序并分散持仓",
    },
    "regime_reversal_quality": {
        "strategy_id": "independent_alpha_families",
        "params": {
            "family": "regime_reversal_quality",
            "eligibility_mode": "pit",
        },
        "execution": {"max_positions": 15, "max_hold_days": 15, "stop_loss": -0.06},
        "mechanism": "非压力环境运行新低修复排序；压力环境清仓并切换月频质量动量",
    },
}


def _config(spec: dict, start: date, end: date):
    return common._config(
        spec["strategy_id"],
        start,
        end,
        params=spec["params"],
        basic_filter_override=PIT_FILTER,
        **spec["execution"],
    )


def _baseline_config(start: date, end: date):
    return common._config(
        "reversal_first_principles",
        start,
        end,
        params=BASELINE_PARAMS,
        max_positions=10,
        max_hold_days=15,
        stop_loss=-0.06,
        basic_filter_override=PIT_FILTER,
    )


def _run(service, config, prepared) -> dict:
    result = service.run(config, prepared=prepared, result_policy=common.POLICY)
    if not result.error:
        return {key: result.stats.get(key) for key in common.POLICY.required_stats}
    if "未产生买入信号" not in result.error:
        raise RuntimeError(result.error)
    return {
        "total_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "verified_no_signal": True,
    }


def _regime_by_fold(market) -> dict[str, dict]:
    change = matrix_feature(market, "change_pct")
    ma20 = matrix_feature(market, "ma20")
    eligible = matrix_feature(market, "pit_eligible") > np.float32(0.5)
    labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
    raw = {}
    for name, start, end in FOLDS:
        row_ids = np.array(
            [index for index, value in enumerate(labels) if start <= value <= end],
            dtype=np.int32,
        )
        valid = np.isfinite(change[row_ids]) & eligible[row_ids]
        counts = valid.sum(axis=1)
        breadth = np.divide(
            ((change[row_ids] > 0) & valid).sum(axis=1),
            counts,
            out=np.zeros_like(counts, dtype=np.float32),
            where=counts > 0,
        )
        valid_ma = (
            np.isfinite(ma20[row_ids])
            & np.isfinite(market.close[row_ids])
            & eligible[row_ids]
        )
        ma_counts = valid_ma.sum(axis=1)
        above = np.divide(
            ((market.close[row_ids] > ma20[row_ids]) & valid_ma).sum(axis=1),
            ma_counts,
            out=np.zeros_like(ma_counts, dtype=np.float32),
            where=ma_counts > 0,
        )
        mean_above = float(np.nanmean(above))
        raw[name] = {
            "mean_above_ma20": round(mean_above, 4),
            "mean_breadth": round(float(np.nanmean(breadth)), 4),
        }
    values = np.array(
        [row["mean_above_ma20"] for row in raw.values()], dtype=np.float32
    )
    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    output = {}
    for name, row in raw.items():
        value = row["mean_above_ma20"]
        if value >= upper:
            regime = "risk_on"
        elif value <= lower:
            regime = "stress"
        else:
            regime = "mixed"
        output[name] = {**row, "regime": regime}
    return output


def _summarize(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    full_candidate: dict,
    full_baseline: dict,
    fold_regimes: dict[str, dict],
) -> dict:
    candidate_returns = [float(row["stats"]["total_return"]) for row in candidate_rows]
    baseline_returns = [float(row["stats"]["total_return"]) for row in baseline_rows]
    excess = [
        left - right
        for left, right in zip(candidate_returns, baseline_returns, strict=True)
    ]
    positive = sum(value > 0 for value in candidate_returns)
    baseline_positive = sum(value > 0 for value in baseline_returns)
    beats = sum(
        left > right
        for left, right in zip(candidate_returns, baseline_returns, strict=True)
    )
    regime_results = {}
    for regime in ("risk_on", "mixed", "stress"):
        ids = [
            index
            for index, row in enumerate(candidate_rows)
            if fold_regimes[row["label"]]["regime"] == regime
        ]
        if not ids:
            continue
        candidate_mean = statistics.mean(candidate_returns[index] for index in ids)
        baseline_mean = statistics.mean(baseline_returns[index] for index in ids)
        regime_results[regime] = {
            "folds": len(ids),
            "candidate_mean_return": round(candidate_mean, 6),
            "baseline_mean_return": round(baseline_mean, 6),
            "beats_baseline": candidate_mean > baseline_mean,
        }
    regime_wins = sum(row["beats_baseline"] for row in regime_results.values())
    full_excess = float(full_candidate["total_return"]) - float(
        full_baseline["total_return"]
    )
    qualified = (
        float(full_candidate["total_return"]) > 0
        and full_excess > 0
        and float(full_candidate["sharpe"]) > float(full_baseline["sharpe"])
        and float(full_candidate["max_drawdown"])
        >= float(full_baseline["max_drawdown"]) - 0.05
        and positive >= baseline_positive
        and beats >= math.ceil(len(candidate_rows) * 0.55)
        and regime_wins >= 2
        and int(full_candidate["n_trades"]) >= 100
    )
    return {
        "qualified": qualified,
        "positive_folds": positive,
        "baseline_positive_folds": baseline_positive,
        "beats_baseline_folds": beats,
        "folds": len(candidate_rows),
        "median_excess": round(statistics.median(excess), 6),
        "regime_wins": regime_wins,
        "regime_results": regime_results,
        "full": full_candidate,
        "baseline_full": full_baseline,
    }


def _head_to_head(
    candidate_rows: list[dict],
    benchmark_rows: list[dict],
    full_candidate: dict,
    full_benchmark: dict,
    fold_regimes: dict[str, dict],
) -> dict:
    candidate_returns = [float(row["stats"]["total_return"]) for row in candidate_rows]
    benchmark_returns = [float(row["stats"]["total_return"]) for row in benchmark_rows]
    beats = sum(
        left > right
        for left, right in zip(candidate_returns, benchmark_returns, strict=True)
    )
    positive = sum(value > 0 for value in candidate_returns)
    benchmark_positive = sum(value > 0 for value in benchmark_returns)
    regime_results = {}
    for regime in ("risk_on", "mixed", "stress"):
        ids = [
            index
            for index, row in enumerate(candidate_rows)
            if fold_regimes[row["label"]]["regime"] == regime
        ]
        candidate_mean = statistics.mean(candidate_returns[index] for index in ids)
        benchmark_mean = statistics.mean(benchmark_returns[index] for index in ids)
        regime_results[regime] = {
            "folds": len(ids),
            "candidate_mean_return": round(candidate_mean, 6),
            "benchmark_mean_return": round(benchmark_mean, 6),
            "beats_benchmark": candidate_mean > benchmark_mean,
        }
    regime_wins = sum(row["beats_benchmark"] for row in regime_results.values())
    passed = (
        float(full_candidate["total_return"]) > float(full_benchmark["total_return"])
        and float(full_candidate["sharpe"]) > float(full_benchmark["sharpe"])
        and float(full_candidate["max_drawdown"])
        >= float(full_benchmark["max_drawdown"]) - 0.05
        and positive >= benchmark_positive
        and beats >= math.ceil(len(candidate_rows) * 0.55)
        and regime_wins >= 2
        and int(full_candidate["n_trades"]) >= 100
    )
    return {
        "passed": passed,
        "benchmark": "reversal_recovery_p15",
        "beats_benchmark_folds": beats,
        "positive_folds": positive,
        "benchmark_positive_folds": benchmark_positive,
        "regime_wins": regime_wins,
        "regime_results": regime_results,
        "full_candidate": full_candidate,
        "full_benchmark": full_benchmark,
    }


def run(data_dir: Path, research_dir: Path, output: Path) -> None:
    _, service = common._engine(data_dir, research_dir)
    loader_params = {**BASELINE_PARAMS, "eligibility_mode": "none"}
    loader = common._prepared(
        service,
        [
            common._config(
                "reversal_first_principles",
                START,
                END,
                params=loader_params,
                max_positions=10,
                max_hold_days=15,
                stop_loss=-0.06,
                basic_filter_override=PIT_FILTER,
            )
        ],
    )
    market = common._attach_industry_context(loader.market_data, data_dir)
    market, pit_context = common._attach_point_in_time_universe(market, data_dir)
    fold_regimes = _regime_by_fold(market)
    try:
        names = list(CANDIDATES)
        fold_candidate = {name: [] for name in names}
        fold_baseline = []
        for label, fold_start, fold_end in FOLDS:
            configs = [_config(CANDIDATES[name], fold_start, fold_end) for name in names]
            prepared_by_name, prepared_objects = common._prepared_groups(
                service, names, configs, market
            )
            baseline_config = _baseline_config(fold_start, fold_end)
            baseline_prepared = common._prepared(service, [baseline_config], market)
            try:
                baseline_stats = _run(service, baseline_config, baseline_prepared)
                fold_baseline.append({"label": label, "stats": baseline_stats})
                for name, config in zip(names, configs, strict=True):
                    fold_candidate[name].append(
                        {
                            "label": label,
                            "stats": _run(
                                service, config, prepared_by_name[name]
                            ),
                        }
                    )
            finally:
                for prepared in prepared_objects:
                    prepared.compute_cache.close()
                baseline_prepared.compute_cache.close()

        configs = [_config(CANDIDATES[name], START, END) for name in names]
        prepared_by_name, prepared_objects = common._prepared_groups(
            service, names, configs, market
        )
        baseline_config = _baseline_config(START, END)
        baseline_prepared = common._prepared(service, [baseline_config], market)
        try:
            full_baseline = _run(service, baseline_config, baseline_prepared)
            summaries = {}
            full_candidates = {}
            for name, config in zip(names, configs, strict=True):
                full_candidates[name] = _run(
                    service, config, prepared_by_name[name]
                )
                summaries[name] = _summarize(
                    fold_candidate[name],
                    fold_baseline,
                    full_candidates[name],
                    full_baseline,
                    fold_regimes,
                )
        finally:
            for prepared in prepared_objects:
                prepared.compute_cache.close()
            baseline_prepared.compute_cache.close()

        head_to_head = _head_to_head(
            fold_candidate["regime_reversal_quality"],
            fold_candidate["reversal_recovery_p15"],
            full_candidates["regime_reversal_quality"],
            full_candidates["reversal_recovery_p15"],
            fold_regimes,
        )
        qualified = [name for name in names if summaries[name]["qualified"]]
        payload = {
            "phase": "cross_family_robustness",
            "range": [START.isoformat(), END.isoformat()],
            "method": "24 half-year folds plus causal market-state switching; no claim of untouched OOS",
            "point_in_time_context": pit_context,
            "baseline": {
                "id": "point_in_time_new_low_reversal",
                "params": BASELINE_PARAMS,
                "execution": {
                    "max_positions": 10,
                    "max_hold_days": 15,
                    "stop_loss": -0.06,
                },
                "full": full_baseline,
            },
            "candidate_specs": CANDIDATES,
            "fold_regimes": fold_regimes,
            "fold_baseline": fold_baseline,
            "fold_candidates": fold_candidate,
            "summaries": summaries,
            "qualified": qualified,
            "head_to_head": head_to_head,
            "winner": "regime_reversal_quality" if head_to_head["passed"] else None,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "winner": payload["winner"],
                    "qualified": qualified,
                    "summaries": summaries,
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
        default=Path("/app/data/research/independent_alpha_families.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
