from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_broker_gold_stock_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_broker_gold_stock_data", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "recommendation_month": [date(2021, 7, 1)] * 4,
            "available_after": [date(2021, 7, 3)] * 4,
            "broker": ["甲证券", "乙证券", "甲证券", "丙证券"],
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "600000.SH"],
            "name_at_source": ["甲", "甲", "甲", "乙"],
            "partition_year": [2021] * 4,
            "partition_month": [7] * 4,
        }
    )


def test_required_periods_cover_exactly_74_months() -> None:
    periods = audit_module.required_periods()

    assert len(periods) == 74
    assert periods[0] == (2020, 7)
    assert periods[-1] == (2026, 8)


def test_consensus_counts_distinct_brokers_only() -> None:
    consensus = audit_module.build_consensus(_events())

    assert consensus.filter(pl.col("symbol") == "000001.SZ")["broker_count"][0] == 2
    assert consensus.filter(pl.col("symbol") == "600000.SH")["broker_count"][0] == 1


def test_audit_rejects_duplicate_broker_stock_rows() -> None:
    result = audit_module.audit(_events(), partition_count=74)

    assert result["status"] == "DATA_GAP"
    assert result["outcome_fields_read"] is False
    assert result["data"]["duplicate_rows"] == 1
    assert result["candidate_sample_sizes"]["brokers_gte_2"]["signals"] == 1
