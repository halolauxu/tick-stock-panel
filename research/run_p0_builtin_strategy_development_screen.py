"""Screen all frozen builtin stock strategies on the development period."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_reversal_study as common  # noqa: E402
from app.backtest.engine import BacktestEngine  # noqa: E402
from app.backtest.strategy import (  # noqa: E402
    BacktestResultPolicy,
    StrategyBacktestService,
)
from app.strategy.engine import StrategyEngine  # noqa: E402
from app.tickflow.repository import DataStore, KlineRepository  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_ANNUAL_RETURN = 0.50
MIN_SHARPE = 1.0
MIN_MAX_DRAWDOWN = -0.30
MIN_TRADES = 100

PIT_FILTER = {
    "enabled": True,
    "price_min": 3,
    "price_max": 300,
    "market_cap_min": 10e8,
    "amount_min": 0.5e8,
    "exclude_st": False,
    "exclude_new_days": 180,
}

POLICY = BacktestResultPolicy(
    required_stats=frozenset(
        {
            "total_return",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_fees",
        }
    ),
    include_monte_carlo=False,
    include_curves=False,
    include_trades=False,
    include_per_symbol_stats=False,
    include_return_distribution=False,
    include_benchmark=False,
    include_strategy_info=False,
)


def eligible_strategy_ids(metadata: list[dict[str, Any]]) -> list[str]:
    return sorted(
        str(row["id"])
        for row in metadata
        if row.get("source") == "builtin"
        and not row.get("research_only")
        and "stock" in row.get("asset_types", [])
        and row.get("execution_backend") == "matrix_native"
    )


def evaluate_result(stats: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "annual_return_at_least_50pct": float(
            stats.get("annual_return") or -999.0
        )
        >= MIN_ANNUAL_RETURN,
        "max_drawdown_no_worse_than_30pct": float(
            stats.get("max_drawdown") or -999.0
        )
        >= MIN_MAX_DRAWDOWN,
        "sharpe_at_least_one": float(stats.get("sharpe") or -999.0)
        >= MIN_SHARPE,
        "at_least_100_trades": int(stats.get("n_trades") or 0) >= MIN_TRADES,
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_service(
    data_dir: Path, research_dir: Path
) -> tuple[StrategyEngine, StrategyBacktestService]:
    builtin = ROOT / "backend" / "app" / "strategy" / "builtin"
    engine = StrategyEngine(
        [builtin, research_dir],
        override_loader=lambda _strategy_id: {},
    )
    if engine.load_errors():
        raise RuntimeError(f"strategy load errors: {engine.load_errors()}")
    repo = KlineRepository(DataStore(data_dir))
    return engine, StrategyBacktestService(BacktestEngine(repo), engine)


def run(data_dir: Path, research_dir: Path, output: Path) -> dict[str, Any]:
    engine, service = build_service(data_dir, research_dir)
    strategy_ids = eligible_strategy_ids(engine.list_strategies())
    if not strategy_ids:
        raise ValueError("no eligible builtin stock strategies")
    configs = [
        common._config(
            strategy_id,
            DEVELOPMENT_START,
            DEVELOPMENT_END,
            max_positions=10,
            basic_filter_override=PIT_FILTER,
        )
        for strategy_id in strategy_ids
    ]
    for config in configs:
        config.enforce_t_plus_one = True
    results: dict[str, Any] = {}
    promoted = []
    pit_context: dict[str, Any] | None = None
    for strategy_id, config in zip(strategy_ids, configs, strict=True):
        loader = common._prepared(service, [config])
        prepared = None
        try:
            market, current_pit_context = common._attach_point_in_time_universe(
                loader.market_data, data_dir
            )
            if pit_context is None:
                pit_context = current_pit_context
            prepared = common._prepared(service, [config], market)
            result = service.run(
                config,
                prepared=prepared,
                result_policy=POLICY,
            )
            if result.error:
                results[strategy_id] = {
                    "error": result.error,
                    "decision": {"passed": False, "checks": {}},
                }
                continue
            stats = {key: result.stats.get(key) for key in POLICY.required_stats}
            decision = evaluate_result(stats)
            results[strategy_id] = {"stats": stats, "decision": decision}
            if decision["passed"]:
                promoted.append(strategy_id)
        finally:
            if prepared is not None:
                prepared.compute_cache.close()
            loader.compute_cache.close()
    payload = {
        "schema_version": "p0-builtin-strategy-development-screen-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "account": {
            "initial_cash_cny": 200_000.0,
            "max_positions": 10,
            "t_plus_one": True,
            "entry_exit": "next trading-day open",
        },
        "point_in_time_context": pit_context,
        "strategy_ids": strategy_ids,
        "results": results,
        "promoted_to_independent_validation": promoted,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "strategy_count": len(strategy_ids),
                "results": results,
                "promoted_to_independent_validation": promoted,
                "output": str(output),
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--research-dir", type=Path, default=Path("/app/research/strategies")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_builtin_strategy_development_screen.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.research_dir, args.output)


if __name__ == "__main__":
    main()
