from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import run_p0_main_board_neglected_liquidity_premium as study  # noqa: E402


def test_turnover_feature_requires_twenty_consecutive_market_rows() -> None:
    consecutive = list(range(20))
    gapped = [*list(range(19)), 20]
    frame = pl.DataFrame(
        {
            "symbol": ["600000.SH"] * 20 + ["600001.SH"] * 20,
            "date": [date(2020, 1, 1)] * 40,
            "_global_index": consecutive + gapped,
            "amount": [100.0] * 40,
            "market_cap": [1_000.0] * 40,
        }
    )

    result = study.attach_turnover_features(frame)

    assert result.filter(pl.col("symbol") == "600000.SH")[
        "mean_turnover_20d"
    ][-1] == 0.1
    assert result.filter(pl.col("symbol") == "600001.SH")[
        "mean_turnover_20d"
    ][-1] is None


def test_rank_excludes_bottom_thirty_percent_and_size_neutralizes_turnover() -> None:
    count = 100
    signal = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * count,
            "entry_date": [date(2020, 1, 6)] * count,
            "symbol": [f"600{i:03d}.SH" for i in range(count)],
            "market_cap": [1_000_000_000.0 + i * 1_000_000.0 for i in range(count)],
            "mean_amount_20d": [100_000_000.0] * count,
            "amount": [100_000_000.0] * count,
            "raw_close": [10.0] * count,
            "mean_turnover_20d": [float(i + 1) for i in range(count)],
        }
    )

    ranked = study.rank_investable(signal)
    low = study.build_candidates(ranked, study.LOW_TURNOVER)
    high = study.build_candidates(ranked, study.HIGH_TURNOVER)

    assert ranked.height == 70
    assert ranked["market_cap_percentile"].min() > 0.30
    assert low.height == study.TARGET_POSITIONS
    assert high.height == study.TARGET_POSITIONS
    assert set(low["size_bin"].to_list()) == {0, 1, 2, 3, 4}
    assert set(high["size_bin"].to_list()) == {0, 1, 2, 3, 4}


def test_gate_requires_candidate_to_beat_control() -> None:
    account = {
        "metrics": {
            "annualized": 0.25,
            "max_drawdown": -0.20,
            "positive_years": 5,
            "mean_cash_ratio": 0.10,
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
        "account": {"trade_count": 400},
    }
    control = {**account, "metrics": {**account["metrics"], "annualized": 0.19}}
    benchmark = {"annualized": 0.12}

    assert study.evaluate(account, control, benchmark)["passed"] is True
    control["metrics"]["annualized"] = 0.21
    assert study.evaluate(account, control, benchmark)["passed"] is False
