from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_main_board_microcap_account.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_main_board_microcap_account", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("000001.SZ", True),
        ("001289.SZ", True),
        ("002028.SZ", True),
        ("003816.SZ", True),
        ("600000.SH", True),
        ("601799.SH", True),
        ("603279.SH", True),
        ("605499.SH", True),
        ("300001.SZ", False),
        ("301001.SZ", False),
        ("688001.SH", False),
        ("689009.SH", False),
        ("430001.BJ", False),
        ("510300.SH", False),
        ("0000010.SZ", False),
        ("60000.SH", False),
    ],
)
def test_is_main_board_symbol_matches_paper_account_policy(
    symbol: str, expected: bool
) -> None:
    assert study.is_main_board_symbol(symbol) is expected


def test_filter_main_board_removes_chinext_star_and_etf() -> None:
    frame = pl.DataFrame(
        {
            "symbol": [
                "000001.SZ",
                "002028.SZ",
                "600000.SH",
                "605499.SH",
                "300001.SZ",
                "688001.SH",
                "510300.SH",
            ]
        }
    )

    result = study.filter_main_board(frame)

    assert result.get_column("symbol").to_list() == [
        "000001.SZ",
        "002028.SZ",
        "600000.SH",
        "605499.SH",
    ]


def test_drawdown_episode_records_peak_trough_and_recovery() -> None:
    rows = [
        {"date": date(2024, 1, 2), "equity": 100.0},
        {"date": date(2024, 1, 3), "equity": 120.0},
        {"date": date(2024, 1, 4), "equity": 72.0},
        {"date": date(2024, 1, 5), "equity": 100.0},
        {"date": date(2024, 1, 8), "equity": 121.0},
    ]

    result = study.drawdown_episode(rows)

    assert result["drawdown"] == pytest.approx(-0.40)
    assert result["peak_date"] == date(2024, 1, 3)
    assert result["trough_date"] == date(2024, 1, 4)
    assert result["recovery_date"] == date(2024, 1, 8)


def _period_result(*, annualized: float = 0.20, drawdown: float = -0.20) -> dict:
    return {
        "metrics": {
            "account_annualized": annualized,
            "annualized_excess": 0.12,
            "account_max_drawdown": drawdown,
            "positive_account_years": 2,
        },
        "execution": {
            "buy": {"execution_rate": 0.90},
            "sell": {"execution_rate": 0.90},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }


def test_evaluate_account_distinguishes_forward_research_and_terminate() -> None:
    results = {
        "validation": _period_result(),
        "known_stress": _period_result(),
    }
    assert study.evaluate_account(results)["verdict"] == "FORWARD_ELIGIBLE"

    results["known_stress"] = _period_result(drawdown=-0.40)
    decision = study.evaluate_account(results)
    assert decision["verdict"] == "RESEARCH_ONLY"
    assert "known_stress:max_drawdown_within_25pct" in decision["failures"]

    results["known_stress"] = _period_result(annualized=0.10)
    assert study.evaluate_account(results)["verdict"] == "TERMINATE"
