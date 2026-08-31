from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "collect_p0_50etf_ivs_extension.py"
    )
    spec = importlib.util.spec_from_file_location("p0_50etf_ivs_extension", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_normalize_shibor_parses_percent_curve_and_deduplicates() -> None:
    rows = [
        {
            "date": "20240102",
            "on": "1.5",
            "1w": "1.6",
            "2w": "1.7",
            "1m": "1.8",
            "3m": "1.9",
            "6m": "2.0",
            "9m": "2.1",
            "1y": "2.2",
        },
        {
            "date": "20240102",
            "on": "1.6",
            "1w": "1.7",
            "2w": "1.8",
            "1m": "1.9",
            "3m": "2.0",
            "6m": "2.1",
            "9m": "2.2",
            "1y": "2.3",
        },
    ]
    result = study.normalize_shibor(rows)
    assert result.height == 1
    assert result["date"][0] == date(2024, 1, 2)
    assert result["1m"][0] == 1.9


def test_audit_extension_fails_closed_when_history_is_short() -> None:
    master = pl.DataFrame(
        {
            "contract": ["C"],
            "exercise_price": [2.5],
            "opt_multiplier": [10_000.0],
            "min_price_chg": [0.0001],
        }
    )
    fund = pl.DataFrame({"date": [study.START, study.END]})
    options = pl.DataFrame(
        {
            "contract": ["C"],
            "date": [study.START],
            "volume": [1.0],
            "open": [0.1],
            "close": [0.1],
            "settle": [0.1],
        }
    )
    shibor = pl.DataFrame({"date": [study.START]})
    result = study.audit_extension(master, fund, options, shibor)
    assert result["status"] == "DATA_GAP"
    assert result["returns_evaluated"] is False
