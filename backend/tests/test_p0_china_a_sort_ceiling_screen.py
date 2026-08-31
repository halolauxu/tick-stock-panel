from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_china_a_sort_ceiling_screen.py"
    )
    spec = importlib.util.spec_from_file_location("china_a_sort_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_factor_names_require_all_ten_deciles() -> None:
    fields = ["date", *[f"alpha{index}" for index in range(10)], "partial0"]

    assert study.factor_names(fields) == ["alpha"]


def test_read_returns_stops_before_parsing_sealed_rows(tmp_path: Path) -> None:
    source = tmp_path / "returns.csv"
    source.write_text(
        "date," + ",".join(f"alpha{i}" for i in range(10)) + "\n"
        "2018-12-31," + ",".join(["0.01"] * 10) + "\n"
        "2019-01-31," + ",".join(["SEALED"] * 10) + "\n",
        encoding="utf-8",
    )

    result = study.read_returns_through(source, date(2018, 12, 31))

    assert len(result) == 1
    assert result[0]["date"] == date(2018, 12, 31)


def test_metrics_use_monthly_compounding() -> None:
    rows = [(date(2014, month, 28), 0.01) for month in range(1, 13)]

    result = study.metrics(rows)

    assert result["months"] == 12
    assert result["annualized"] == pytest.approx(1.01**12 - 1.0)
    assert result["positive_years"] == 1
    assert result["max_drawdown"] == 0.0


def test_candidate_gate_requires_both_periods_and_both_weightings() -> None:
    def metric(annualized: float, positive_years: int, months: int) -> dict:
        return {
            "annualized": annualized,
            "max_drawdown": -0.20,
            "positive_years": positive_years,
            "months": months,
        }

    record = {
        "equal_weight": {
            "discovery": metric(0.60, 12, 168),
            "confirmation": metric(0.55, 5, 60),
        },
        "equal_weight_benchmark": {
            "discovery": metric(0.30, 10, 168),
            "confirmation": metric(0.30, 4, 60),
        },
        "value_weight": {
            "discovery": metric(0.30, 10, 168),
            "confirmation": metric(0.25, 4, 60),
        },
        "value_weight_benchmark": {
            "discovery": metric(0.20, 10, 168),
            "confirmation": metric(0.15, 4, 60),
        },
    }

    assert study.evaluate_candidate(record)["passed"] is True
    record["equal_weight"]["confirmation"]["annualized"] = 0.49
    assert study.evaluate_candidate(record)["passed"] is False

