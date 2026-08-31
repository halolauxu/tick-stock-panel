from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_corporate_action_technical_forest_development.py"
)
SPEC = importlib.util.spec_from_file_location("corporate_action_forest", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _business_dates(start: date, count: int) -> list[date]:
    output = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return output


def test_evidence_commit_prefers_explicit_runtime_commit(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_GIT_COMMIT", "59e932b-full-sha")
    assert study.implementation_git_commit() == "59e932b-full-sha"


def test_active_pool_maps_after_close_and_weekend_without_future_data() -> None:
    dates = _business_dates(date(2020, 1, 2), 25)
    calendar = pl.DataFrame({"date": dates}).with_row_index("trade_index")
    events = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "ann_date": [dates[2], date(2020, 1, 11)],
            "family": ["repurchase", "holder_increase"],
        }
    )
    active = study.build_active_pool(events, calendar)
    a = active.filter(pl.col("symbol") == "A")
    b = active.filter(pl.col("symbol") == "B")
    assert a.height == study.ACTIVE_DAYS
    assert a.get_column("trade_index").min() == 2
    assert b.get_column("date").min() == date(2020, 1, 10)
    assert b.get_column("trade_index").max() == 25 - 1


def test_training_window_excludes_t_minus_one_unfinished_label() -> None:
    lower, upper = study.training_window(100)
    assert lower == 39
    assert upper == 98
    assert upper < 99


class _MeanModel:
    feature_importances_ = [1.0 / len(study.FEATURE_COLUMNS)] * len(
        study.FEATURE_COLUMNS
    )

    def fit(self, x, y):
        self.mean = float(y.mean())
        return self

    def predict(self, x):
        return x[:, 0] + self.mean * 0.0


def test_walk_forward_training_never_reads_t_minus_one_label(monkeypatch) -> None:
    monkeypatch.setattr(study, "MIN_TRAINING_ROWS", 1)
    monkeypatch.setattr(study, "MIN_TRAINING_DATES", 1)
    dates = _business_dates(date(2014, 9, 1), 100)
    rows = []
    for index, day in enumerate(dates):
        for symbol, value in (("A", 0.1), ("B", -0.1)):
            row = {
                "date": day,
                "trade_index": index,
                "symbol": symbol,
                "event_start_index": 0,
                "families": "repurchase",
                "label_excess": value,
            }
            row.update({column: value for column in study.FEATURE_COLUMNS})
            rows.append(row)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    predictions, audit = study.generate_predictions(
        frame, model_factory=_MeanModel
    )
    assert not predictions.is_empty()
    assert (
        predictions.get_column("trade_index").max()
        <= frame["trade_index"].max() - study.MAX_EXIT_DELAY - 2
    )
    fitted = next(row for row in audit["fits"] if row["fitted"])
    assert fitted["training_end_index"] == fitted["decision_index"] - 2


def test_feature_panel_turns_infinite_inputs_into_neutral_ranks() -> None:
    dates = _business_dates(date(2019, 1, 2), 70)
    rows = []
    for symbol in ("A", "B"):
        for day in dates:
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    "amount": 100_000_000.0,
                    "raw_close": 10.0,
                    "excluded_name": False,
                }
            )
    panel = pl.DataFrame(rows, infer_schema_length=None)
    indexed = study.attach_trade_index(panel)
    active = indexed.filter(pl.col("trade_index") >= 60).select(
        "symbol", "trade_index", "date"
    ).with_columns(
        pl.lit(60).alias("event_start_index"),
        pl.lit("repurchase").alias("families"),
    )
    features = study.build_feature_panel(panel, active)
    finite = features.select(
        pl.all_horizontal(pl.col(column).is_finite() for column in study.FEATURE_COLUMNS)
        .all()
        .alias("all_finite")
    ).item()
    assert finite is True
    assert features.select(study.FEATURE_COLUMNS).to_numpy().max() == 0.0


def test_candidate_ranking_never_substitutes_below_frozen_top_ten() -> None:
    predictions = pl.DataFrame(
        {
            "trade_index": [1] * 12,
            "event_start_index": list(range(12)),
            "symbol": [f"S{index:02d}" for index in range(12)],
            "score": [float(index) for index in range(12)],
        }
    )
    ranked = study.rank_candidates(predictions, control=False)
    assert ranked.height == study.TARGET_POSITIONS
    assert ranked.get_column("symbol").to_list() == [
        f"S{index:02d}" for index in range(11, 1, -1)
    ]


def test_quote_lookup_keeps_adjusted_close_for_position_valuation() -> None:
    dates = _business_dates(date(2020, 1, 2), 3)
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * 3,
            "date": dates,
            "name": ["A"] * 3,
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0],
            "close": [10.05, 10.15, 10.25],
            "raw_open": [10.0, 10.1, 10.2],
            "raw_high": [10.2, 10.3, 10.4],
            "raw_low": [9.8, 9.9, 10.0],
            "raw_close": [10.05, 10.15, 10.25],
            "amount": [100_000_000.0] * 3,
            "volume": [1_000_000.0] * 3,
            "excluded_name": [False] * 3,
        }
    )
    ranked = pl.DataFrame({"symbol": ["A"], "entry_index": [1]})
    lookup, calendar_dates = study.build_quote_lookup(panel, [ranked])
    assert calendar_dates == dates
    assert lookup[("A", 1)]["close"] == pytest.approx(10.15)
    assert lookup[("A", 1)]["exact_quote"] is True


def _quote(index: int, raw_open: float = 10.0, *, limit_up: bool = False) -> dict:
    return {
        "date": date(2020, 1, 2) + timedelta(days=index),
        "trade_index": index,
        "open": raw_open,
        "close": raw_open,
        "raw_open": raw_open,
        "entry_volume": 1_000_000.0,
        "entry_amount": 100_000_000.0,
        "limit_up_price": raw_open if limit_up else raw_open * 1.1,
        "limit_down_price": raw_open * 0.9,
        "is_excluded_name": False,
        "exact_quote": True,
    }


def test_account_buys_next_open_and_sells_only_after_t_plus_one() -> None:
    dates = _business_dates(date(2020, 1, 2), 5)
    ranked = pl.DataFrame(
        {
            "date": [dates[0]],
            "trade_index": [0],
            "entry_index": [1],
            "symbol": ["A"],
            "event_start_index": [0],
            "families": ["repurchase"],
            "score": [1.0],
            "label_excess": [0.01],
            "fit_id": [0],
            "rank": [0],
        }
    )
    quotes = {("A", index): _quote(index, 10.0 + index) for index in range(1, 4)}
    summary, records = study.simulate_account(ranked, quotes, dates, 200_000.0)
    buys = records.filter((pl.col("side") == "BUY") & (pl.col("status") == "FILLED"))
    sells = records.filter((pl.col("side") == "SELL") & (pl.col("status") == "FILLED"))
    assert buys.get_column("trade_index").to_list() == [1]
    assert sells.get_column("trade_index").to_list() == [2]
    assert summary["completed_sells"] == 1
    assert summary["maximum_cash_ledger_error"] == pytest.approx(0.0)


def test_account_skips_open_limit_up_without_manufacturing_fill() -> None:
    dates = _business_dates(date(2020, 1, 2), 4)
    ranked = pl.DataFrame(
        {
            "date": [dates[0]],
            "trade_index": [0],
            "entry_index": [1],
            "symbol": ["A"],
            "event_start_index": [0],
            "families": ["repurchase"],
            "score": [1.0],
            "label_excess": [0.01],
            "fit_id": [0],
            "rank": [0],
        }
    )
    quotes = {("A", 1): _quote(1, limit_up=True)}
    summary, _ = study.simulate_account(ranked, quotes, dates, 200_000.0)
    assert summary["filled_buys"] == 0
    assert summary["buy_reject_reasons"] == {"limit_up": 1}


def test_account_counts_any_ending_position_as_unresolved() -> None:
    dates = _business_dates(date(2020, 1, 2), 3)
    ranked = pl.DataFrame(
        {
            "date": [dates[0]],
            "trade_index": [0],
            "entry_index": [1],
            "symbol": ["A"],
            "event_start_index": [0],
            "families": ["repurchase"],
            "score": [1.0],
            "label_excess": [0.01],
            "fit_id": [0],
            "rank": [0],
        }
    )
    quotes = {("A", 1): _quote(1)}
    summary, _ = study.simulate_account(ranked, quotes, dates, 200_000.0)
    assert summary["ending_positions"] == 1
    assert summary["unresolved_exits"] == 1


def test_gate_requires_return_control_ic_and_clean_exits() -> None:
    candidate = {
        "annualized_return": 0.60,
        "max_drawdown": -0.20,
        "positive_years": 5,
        "completed_sells": 600,
        "execution_rate": 0.95,
        "unresolved_exits": 1,
        "largest_positive_year_share": 0.30,
        "maximum_cash_ledger_error": 0.0,
    }
    control = {"annualized_return": 0.50}
    benchmark = {"annualized_return": 0.20}
    audit = {
        "block_ic_mean": 0.02,
        "block_ic_ir": 0.60,
        "maximum_group_feature_importance": 0.20,
    }
    decision = study.evaluate_gate(candidate, control, benchmark, audit)
    assert decision["passed"] is False
    assert decision["checks"]["event_control_increment_at_least_15pp"] is False
    assert decision["checks"]["block_ic_mean_at_least_003"] is False
    assert decision["checks"]["no_unresolved_exits"] is False
