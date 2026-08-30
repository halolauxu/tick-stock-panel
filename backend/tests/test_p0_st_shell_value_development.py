from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_st_shell_value_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_st_shell", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_point_in_time_pool_keeps_only_ordinary_st(tmp_path: Path) -> None:
    research = tmp_path / "research"
    shares_dir = tmp_path / "financials" / "shares"
    research.mkdir(parents=True)
    shares_dir.mkdir(parents=True)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    pl.DataFrame(
        {
            "symbol": symbols,
            "list_date": [date(2010, 1, 1)] * 4,
            "delist_date": [None] * 4,
        }
    ).write_parquet(research / "historical_stock_universe_all_a.parquet")
    pl.DataFrame(
        {
            "symbol": symbols,
            "name": ["ST甲", "*ST乙", "退市丙", "普通丁"],
            "start_date": [date(2019, 1, 1)] * 4,
            "end_date": [None] * 4,
        }
    ).write_parquet(research / "historical_stock_names_all_a.parquet")
    pl.DataFrame(
        {
            "symbol": symbols,
            "announce_date": [date(2019, 1, 1)] * 4,
            "period_end": [date(2018, 12, 31)] * 4,
            "total_shares": [100_000_000.0] * 4,
            "float_shares": [50_000_000.0] * 4,
        }
    ).write_parquet(shares_dir / "part.parquet")
    panel = pl.DataFrame(
        {
            "symbol": symbols,
            "date": [date(2020, 1, 2)] * 4,
            "open": [5.0] * 4,
            "close": [5.0] * 4,
            "volume": [100.0] * 4,
            "amount": [1_000_000.0] * 4,
            "raw_close": [5.0] * 4,
        }
    )

    result = study.attach_st_point_in_time_data(panel, tmp_path)

    assert result.get_column("symbol").to_list() == ["000001.SZ"]


def test_candidate_clock_uses_week_end_close_and_next_available_date() -> None:
    days = [
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
        date(2020, 1, 7),
    ]
    rows = []
    for day in days:
        for symbol_index in range(35):
            rows.append(
                {
                    "symbol": f"000{symbol_index:03d}.SZ",
                    "date": day,
                    "market_cap": float(symbol_index + 1),
                    "amount": 2_000_000.0,
                    "daily_return": 0.0,
                }
            )
    result = study.build_signal_candidates(pl.DataFrame(rows))

    first = result.filter(pl.col("date") == date(2020, 1, 3))
    assert first.height == study.CANDIDATE_QUEUE
    assert first.get_column("entry_date").unique().to_list() == [date(2020, 1, 6)]
    assert first.get_column("cap_rank").max() == study.CANDIDATE_QUEUE


def test_development_gate_requires_return_drawdown_and_execution() -> None:
    main = {
        "metrics": {
            "account_annualized": 0.60,
            "annualized_excess": 0.40,
            "account_max_drawdown": -0.20,
            "yearly": [
                {"year": year, "account_return": 0.10}
                for year in range(2014, 2021)
            ],
        },
        "execution": {
            "buy": {"execution_rate": 0.95},
            "sell": {"execution_rate": 0.95},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }

    assert study.evaluate_development(main)["passed"] is True
    main["execution"]["sell"]["execution_rate"] = 0.80
    assert study.evaluate_development(main)["passed"] is False
