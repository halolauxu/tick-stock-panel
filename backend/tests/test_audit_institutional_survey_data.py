from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_institutional_survey_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_institutional_survey_data", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()

SCHEMA = {
    "event_id": pl.String,
    "notice_date": pl.Date,
    "symbol": pl.String,
    "institution_count": pl.UInt32,
    "survey_session_count": pl.UInt32,
    "institution_detail_rows": pl.UInt32,
    "provider_sum_max": pl.Int64,
    "org_types": pl.String,
    "source_url": pl.String,
}


def _event(symbol: str, notice_date: date, count: int, suffix: str) -> dict:
    return {
        "event_id": f"survey-{symbol}-{notice_date:%Y%m%d}-{suffix}",
        "notice_date": notice_date,
        "symbol": symbol,
        "institution_count": count,
        "survey_session_count": 1,
        "institution_detail_rows": count,
        "provider_sum_max": count,
        "org_types": "证券公司",
        "source_url": "AN1",
    }


def _write_complete(tmp_path: Path, *, spikes: int) -> None:
    rows: dict[tuple[int, int], list[dict]] = {}
    start = date(2014, 1, 1)
    for index in range(spikes):
        signal_date = start + timedelta(days=index // 2)
        symbol = f"{index + 1:06d}.SZ"
        baseline_date = signal_date - timedelta(days=30)
        rows.setdefault((baseline_date.year, baseline_date.month), []).append(
            _event(symbol, baseline_date, 5, "baseline")
        )
        rows.setdefault((signal_date.year, signal_date.month), []).append(
            _event(symbol, signal_date, 10, "spike")
        )
    root = tmp_path / "event_data" / "institutional_survey"
    for year in range(2013, 2021):
        for month in range(1, 13):
            target = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
            target.parent.mkdir(parents=True)
            pl.DataFrame(rows.get((year, month), []), schema=SCHEMA).write_parquet(
                target
            )


def test_missing_months_do_not_open_outcomes(tmp_path) -> None:
    result = audit_module.audit(tmp_path)

    assert result["status"] == "DATA_INCOMPLETE"
    assert result["future_returns_read"] is False
    assert result["price_data_read"] is False


def test_complete_sufficient_metadata_selects_spikes_without_prices(tmp_path) -> None:
    _write_complete(tmp_path, spikes=600)

    result = audit_module.audit(tmp_path)

    assert result["status"] == "SAMPLE_SUFFICIENT"
    assert result["attention_spikes"] == 600
    assert result["attention_spike_days"] == 300
    assert result["duplicate_event_ids"] == 0
    assert result["future_returns_read"] is False
    assert result["price_data_read"] is False


def test_complete_but_small_signal_population_is_sparse(tmp_path) -> None:
    _write_complete(tmp_path, spikes=10)

    result = audit_module.audit(tmp_path)

    assert result["status"] == "SAMPLE_SPARSE"
    assert result["attention_spikes"] == 10
