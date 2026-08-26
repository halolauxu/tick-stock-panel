from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from app.backtest.fundamentals import FUNDAMENTAL_FACTOR_NAMES
from app.backtest.matrix import build_market_data_matrix, validate_signal_matrix
from app.backtest.strategy import StrategyDependencyResolver
from app.services.screener import ScreenerService
from app.strategy.builtin import serenity_bottleneck_research as serenity
from app.strategy.engine import StrategyEngine

STRATEGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "strategy"
    / "builtin"
    / "serenity_bottleneck_research.py"
)


def _panel(*, include_financials: bool = True) -> pl.DataFrame:
    rows: list[dict] = []
    start = date(2026, 1, 1)
    symbols = ("000001.SZ", "600000.SH", "300001.SZ")
    for offset in range(70):
        for asset_id, symbol in enumerate(symbols):
            close = 10.0 + offset * (0.07 - asset_id * 0.01)
            row = {
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1_000_000.0,
                "amount": 150_000_000.0 * (2.0 if offset == 69 else 1.0),
                "turnover_rate": 3.0,
                "total_shares": 500_000_000.0,
            }
            if include_financials:
                row.update({
                    "revenue_yoy_latest": (40.0, 20.0, -5.0)[asset_id],
                    "net_income_yoy_latest": (35.0, 12.0, -10.0)[asset_id],
                    "roe_latest": (20.0, 12.0, 4.0)[asset_id],
                    "gross_margin_latest": (45.0, 25.0, 8.0)[asset_id],
                    "debt_ratio_latest": (30.0, 55.0, 85.0)[asset_id],
                    "pb_latest": (3.0, 5.0, 20.0)[asset_id],
                })
            rows.append(row)
    return pl.DataFrame(rows).sort(["symbol", "date"])


def _params(**overrides) -> dict:
    params = {
        item["id"]: item["default"]
        for item in serenity.META["params"]
    }
    params.update(overrides)
    return params


def test_strategy_is_public_matrix_strategy_with_only_36_automatic_points():
    loaded = StrategyEngine._load_file(STRATEGY_PATH)

    assert loaded.meta["id"] == "serenity_bottleneck_research"
    assert loaded.meta.get("research_only") is not True
    assert loaded.execution_backend == "matrix_native"
    assert loaded.matrix_strategy.required_warmup_bars({}) == 61
    assert FUNDAMENTAL_FACTOR_NAMES >= serenity._FINANCIAL_FIELDS
    assert "36" in loaded.meta["description"]
    assert "64" in loaded.meta["description"]


def test_strategy_ranks_only_financially_qualified_candidates():
    panel = _panel()
    market = build_market_data_matrix(
        panel,
        field_columns=serenity.MATRIX_STRATEGY.required_fields() | {"total_shares"},
    )
    signals = serenity.MATRIX_STRATEGY.compute_signals(
        market,
        _params(entry_auto_score=0.0, exit_auto_score=0.0, top_rank=1),
    )

    validate_signal_matrix(signals, market.shape)
    selected = [
        symbol
        for symbol, hit in zip(market.symbols, signals.entry[-1], strict=True)
        if hit
    ]
    assert selected == ["000001.SZ"]
    assert 0.0 < float(signals.score[-1].max()) <= 36.0
    assert signals.entry_signal_ids == ("signal_serenity_auto_screen",)
    assert signals.exit_signal_ids == ("signal_serenity_auto_downgrade",)


def test_strategy_fails_closed_when_financial_snapshot_is_missing(tmp_path):
    panel = _panel(include_financials=False)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    service = ScreenerService(repo)
    service._load_enriched_for_date = lambda _day: panel.filter(  # type: ignore[method-assign]
        pl.col("date") == panel["date"].max()
    )
    service._load_enriched_history = lambda _day, _bars: panel  # type: ignore[method-assign]
    engine = StrategyEngine(strategy_dirs=[STRATEGY_PATH.parent])
    as_of = panel["date"].max()

    context = service.build_strategy_context(
        engine,
        as_of,
        ["serenity_bottleneck_research"],
    )
    result = engine.run("serenity_bottleneck_research", context)

    assert result.total == 0
    assert result.rows == []


def test_screener_matrix_attaches_point_in_time_financial_fields(tmp_path):
    panel = _panel(include_financials=False)
    financial_dir = tmp_path / "financials" / "metrics"
    financial_dir.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001.SZ", "600000.SH", "300001.SZ"],
        "period_end": ["2025-12-31"] * 3,
        "announce_date": ["2026-01-05"] * 3,
        "eps_basic": [1.0] * 3,
        "eps_diluted": [1.0] * 3,
        "bps": [4.0, 3.0, 1.0],
        "ocfps": [1.0] * 3,
        "roe": [20.0, 12.0, 4.0],
        "roe_diluted": [20.0, 12.0, 4.0],
        "roa": [10.0, 6.0, 1.0],
        "gross_margin": [45.0, 25.0, 8.0],
        "net_margin": [20.0, 10.0, -2.0],
        "debt_to_asset_ratio": [30.0, 55.0, 85.0],
        "revenue_yoy": [40.0, 20.0, -5.0],
        "net_income_yoy": [35.0, 12.0, -10.0],
        "operating_cash_to_revenue": [0.2, 0.1, -0.1],
        "inventory_turnover": [5.0, 4.0, 1.0],
    }).write_parquet(financial_dir / "part.parquet")

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    service = ScreenerService(repo)
    service._load_enriched_for_date = lambda _day: panel.filter(  # type: ignore[method-assign]
        pl.col("date") == panel["date"].max()
    )
    service._load_enriched_history = lambda _day, _bars: panel  # type: ignore[method-assign]
    engine = StrategyEngine(strategy_dirs=[STRATEGY_PATH.parent])
    as_of = panel["date"].max()
    context = service.build_strategy_context(
        engine,
        as_of,
        ["serenity_bottleneck_research"],
    )

    result = engine.run(
        "serenity_bottleneck_research",
        context,
        params=_params(entry_auto_score=0.0, exit_auto_score=0.0, top_rank=2),
    )

    assert result.total == 2
    assert [row["symbol"] for row in result.rows] == ["000001.SZ", "600000.SH"]
    assert all(np.isfinite(row["score"]) for row in result.rows)


def test_dependency_plan_requests_point_in_time_financial_fields():
    strategy = StrategyEngine._load_file(STRATEGY_PATH)
    plan = StrategyDependencyResolver().resolve(
        strategy,
        params=_params(),
        basic_filter=strategy.basic_filter,
        entry_signals=strategy.entry_signals,
        exit_signals=strategy.exit_signals,
    )

    assert plan.fundamental_columns == serenity._FINANCIAL_FIELDS
