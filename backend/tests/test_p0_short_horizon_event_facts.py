from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "audit_p0_short_horizon_event_facts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_short_horizon_event_facts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_reason_classifier_separates_operating_and_one_time_facts() -> None:
    assert study.classify_reason("产品销量增长,主营业务收入提升") == "OPERATING"
    assert study.classify_reason("收到政府补助并处置子公司股权") == "ONE_TIME"
    assert study.classify_reason("产品销量增长,同时收到政府补助") == "MIXED"
    assert study.classify_reason("预计业绩同比上升") == "UNCLASSIFIED"
    assert study.classify_reason(None) == "MISSING"

    classified = pl.DataFrame({"change_reason": [None, "主营产品销量增长"]}).select(
        pl.col("change_reason")
        .map_elements(
            study.classify_reason,
            return_dtype=pl.String,
            skip_nulls=False,
        )
        .alias("reason_class")
    )
    assert classified.get_column("reason_class").to_list() == [
        "MISSING",
        "OPERATING",
    ]


def test_prior_financial_join_never_uses_same_day_or_future_report() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "ann_date": [date(2020, 4, 10), date(2020, 4, 20)],
        }
    )
    metrics = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ", "A.SZ"],
            "announce_date": [
                date(2020, 3, 31),
                date(2020, 4, 10),
                date(2020, 4, 15),
            ],
            "period_end": [
                date(2019, 12, 31),
                date(2020, 3, 31),
                date(2020, 3, 31),
            ],
            "roe": [8.0, 9.0, 10.0],
            "operating_cash_to_revenue": [0.1, 0.2, 0.3],
        }
    )
    cash = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "announce_date": [date(2020, 3, 20), date(2020, 4, 20)],
            "period_end": [date(2019, 12, 31), date(2020, 3, 31)],
            "net_operating_cash_flow": [100.0, 200.0],
        }
    )

    result = study.attach_prior_financials(events, metrics, cash)

    assert result.get_column("prior_metrics_announce_date").to_list() == [
        date(2020, 3, 31),
        date(2020, 4, 15),
    ]
    assert result.get_column("prior_cash_announce_date").to_list() == [
        date(2020, 3, 20),
        date(2020, 3, 20),
    ]
    assert result.filter(
        (pl.col("prior_metrics_announce_date") >= pl.col("ann_date"))
        | (pl.col("prior_cash_announce_date") >= pl.col("ann_date"))
    ).is_empty()


def test_data_gate_requires_usable_sample_and_point_in_time_coverage() -> None:
    passed = study.evaluate_data_gate(
        {
            "duplicate_event_keys": 0,
            "missing_required_keys": 0,
            "industry_mapping_rate": 0.99,
            "reason_text_rate": 0.98,
            "collection_source_rate": 1.0,
            "prior_metrics_rate": 0.90,
            "prior_cash_flow_rate": 0.85,
            "future_financial_rows": 0,
            "fact_qualified_events": 80,
            "fact_qualified_years": 7,
        }
    )
    assert passed["verdict"] == "PASS_TO_EVENT_ACCOUNT"

    failed = study.evaluate_data_gate(
        {
            "duplicate_event_keys": 0,
            "missing_required_keys": 0,
            "industry_mapping_rate": 0.99,
            "reason_text_rate": 0.98,
            "collection_source_rate": 1.0,
            "prior_metrics_rate": 0.90,
            "prior_cash_flow_rate": 0.40,
            "future_financial_rows": 0,
            "fact_qualified_events": 20,
            "fact_qualified_years": 2,
        }
    )
    assert failed["verdict"] == "BLOCKED_DATA"
    assert "at_least_50_fact_qualified_events" in failed["failures"]
