"""Audit point-in-time facts for the V2 company-specific forecast event lane.

This stage does not read market outcomes.  It verifies event keys, first-known
dates, point-in-time industry membership, reason-text provenance, and strictly
prior financial context before any semantic model or account test is allowed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_forecast_drift_development as forecast  # noqa: E402
import run_p0_industry_confirmed_forecast_drift_discovery as industry  # noqa: E402

SCHEMA_VERSION = "p0-short-horizon-event-facts-audit-v1"
START_YEAR = 2014

OPERATING_PATTERN = re.compile(
    r"主营|业务|产品|订单|销量|销售|营业收入|营收|收入增长|市场拓展|客户|"
    r"价格上涨|售价|毛利|产能|产量|投产|交付|降本|成本下降|费用下降|"
    r"经营改善|经营业绩|行业景气|需求增长"
)
ONE_TIME_PATTERN = re.compile(
    r"非经常性|一次性|政府补助|补贴|资产处置|处置收益|股权转让|投资收益|"
    r"公允价值|拆迁|征收补偿|债务重组|诉讼|仲裁|赔偿|会计政策|"
    r"汇兑收益|土地收储|重组收益|营业外收入"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def classify_reason(value: str | None) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    if not text:
        return "MISSING"
    operating = OPERATING_PATTERN.search(text) is not None
    one_time = ONE_TIME_PATTERN.search(text) is not None
    if operating and one_time:
        return "MIXED"
    if operating:
        return "OPERATING"
    if one_time:
        return "ONE_TIME"
    return "UNCLASSIFIED"


def _prepare_financial(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("announce_date").cast(pl.Utf8).str.to_date(strict=False),
        pl.col("period_end").cast(pl.Utf8).str.to_date(strict=False),
    ).filter(pl.col("announce_date").is_not_null())


def attach_prior_financials(
    events: pl.DataFrame,
    metrics: pl.DataFrame,
    cash_flow: pl.DataFrame,
) -> pl.DataFrame:
    """Attach only reports announced before the event announcement date."""

    left = events.with_columns(
        (pl.col("ann_date") - pl.duration(days=1)).alias("_information_cutoff")
    ).sort(["symbol", "_information_cutoff"])
    metric_fields = [
        field
        for field in (
            "roe",
            "revenue_yoy",
            "net_income_yoy",
            "operating_cash_to_revenue",
            "debt_to_asset_ratio",
        )
        if field in metrics.columns
    ]
    prepared_metrics = (
        _prepare_financial(metrics)
        .select("symbol", "announce_date", "period_end", *metric_fields)
        .sort(["symbol", "announce_date", "period_end"])
        .unique(subset=["symbol", "announce_date"], keep="last")
        .rename(
            {
                "announce_date": "prior_metrics_announce_date",
                "period_end": "prior_metrics_period_end",
                **{field: f"prior_{field}" for field in metric_fields},
            }
        )
        .sort(["symbol", "prior_metrics_announce_date"])
    )
    joined = left.join_asof(
        prepared_metrics,
        left_on="_information_cutoff",
        right_on="prior_metrics_announce_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    cash_fields = [
        field
        for field in (
            "net_operating_cash_flow",
            "net_investing_cash_flow",
            "net_financing_cash_flow",
            "capex",
        )
        if field in cash_flow.columns
    ]
    prepared_cash = (
        _prepare_financial(cash_flow)
        .select("symbol", "announce_date", "period_end", *cash_fields)
        .sort(["symbol", "announce_date", "period_end"])
        .unique(subset=["symbol", "announce_date"], keep="last")
        .rename(
            {
                "announce_date": "prior_cash_announce_date",
                "period_end": "prior_cash_period_end",
                **{field: f"prior_{field}" for field in cash_fields},
            }
        )
        .sort(["symbol", "prior_cash_announce_date"])
    )
    return (
        joined.sort(["symbol", "_information_cutoff"])
        .join_asof(
            prepared_cash,
            left_on="_information_cutoff",
            right_on="prior_cash_announce_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("_information_cutoff")
        .sort(["ann_date", "symbol"])
    )


def _read_many(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        raise ValueError("required parquet input is missing")
    return pl.concat(
        [pl.read_parquet(path) for path in sorted(paths)],
        how="diagonal_relaxed",
    )


def load_forecasts(data_dir: Path) -> tuple[pl.DataFrame, list[int]]:
    paths = sorted((data_dir / "event_data" / "forecast").glob("year=*/part.parquet"))
    years: list[int] = []
    selected: list[Path] = []
    for path in paths:
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if year >= START_YEAR:
            years.append(year)
            selected.append(path)
    if not selected:
        raise ValueError("forecast yearly partitions are required")
    return _read_many(selected).sort(["ann_date", "symbol"]), years


def evaluate_data_gate(audit: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "event_key_unique": audit["duplicate_event_keys"] == 0,
        "required_keys_complete": audit["missing_required_keys"] == 0,
        "industry_mapping_at_least_95pct": audit["industry_mapping_rate"] >= 0.95,
        "reason_text_at_least_95pct": audit["reason_text_rate"] >= 0.95,
        "collection_source_complete": audit["collection_source_rate"] == 1.0,
        "prior_metrics_at_least_80pct": audit["prior_metrics_rate"] >= 0.80,
        "prior_cash_flow_at_least_70pct": audit["prior_cash_flow_rate"] >= 0.70,
        "no_future_financial_rows": audit["future_financial_rows"] == 0,
        "at_least_50_fact_qualified_events": audit["fact_qualified_events"] >= 50,
        "at_least_5_fact_qualified_years": audit["fact_qualified_years"] >= 5,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "verdict": "PASS_TO_EVENT_ACCOUNT" if not failures else "BLOCKED_DATA",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw, years = load_forecasts(data_dir)
    event_key = [
        "symbol",
        "ann_date",
        "period_end",
        "type",
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
    ]
    duplicate_event_keys = raw.height - raw.unique(subset=event_key).height
    missing_required_keys = raw.filter(
        pl.any_horizontal(
            pl.col("symbol").is_null(),
            pl.col("ann_date").is_null(),
            pl.col("period_end").is_null(),
        )
    ).height
    categorized = forecast.categorize_events(raw)
    membership = industry.load_point_in_time_membership(data_dir)
    mapped, mapping_audit = industry.attach_industry(categorized, membership)
    classified = industry.classify_events(mapped)
    events = classified.filter(pl.col("category") == industry.NEGATIVE_CONTROL)
    events = events.with_columns(
        pl.col("change_reason")
        .map_elements(classify_reason, return_dtype=pl.String, skip_nulls=False)
        .alias("reason_class")
    )

    metrics = _read_many(list((data_dir / "financials" / "metrics").glob("*.parquet")))
    cash_flow = _read_many(list((data_dir / "financials" / "cash_flow").glob("*.parquet")))
    enriched = attach_prior_financials(events, metrics, cash_flow)
    event_count = enriched.height
    reason_text_count = enriched.filter(
        pl.col("change_reason").fill_null("").str.strip_chars().str.len_chars() > 0
    ).height
    source_count = enriched.filter(
        pl.col("collection_source").fill_null("").str.len_chars() > 0
    ).height
    prior_metrics_count = enriched.filter(
        pl.col("prior_metrics_announce_date").is_not_null()
    ).height
    prior_cash_count = enriched.filter(pl.col("prior_cash_announce_date").is_not_null()).height
    future_financial_rows = enriched.filter(
        (pl.col("prior_metrics_announce_date") >= pl.col("ann_date"))
        | (pl.col("prior_cash_announce_date") >= pl.col("ann_date"))
    ).height
    qualified = enriched.filter(
        (pl.col("reason_class") == "OPERATING")
        & pl.col("prior_metrics_announce_date").is_not_null()
        & pl.col("prior_cash_announce_date").is_not_null()
    )
    fact_qualified_years = qualified.get_column("ann_date").dt.year().n_unique()
    reason_counts = dict(sorted(Counter(enriched["reason_class"].to_list()).items()))
    by_year = (
        enriched.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("events"),
            (pl.col("reason_class") == "OPERATING").sum().alias("operating_events"),
            pl.col("symbol").n_unique().alias("symbols"),
        )
        .sort("year")
        .to_dicts()
    )
    audit = {
        "forecast_partitions": years,
        "forecast_rows": raw.height,
        "first_announcement": raw["ann_date"].min(),
        "last_announcement": raw["ann_date"].max(),
        "duplicate_event_keys": duplicate_event_keys,
        "missing_required_keys": missing_required_keys,
        "categorized_first_events": categorized.height,
        "main_board_events": mapping_audit["main_board_events"],
        "mapped_events": mapping_audit["mapped_events"],
        "industry_mapping_rate": mapping_audit["mapping_rate"],
        "idiosyncratic_events": event_count,
        "idiosyncratic_symbols": enriched["symbol"].n_unique(),
        "reason_text_rate": reason_text_count / event_count if event_count else 0.0,
        "collection_source_rate": source_count / event_count if event_count else 0.0,
        "prior_metrics_rate": prior_metrics_count / event_count if event_count else 0.0,
        "prior_cash_flow_rate": prior_cash_count / event_count if event_count else 0.0,
        "future_financial_rows": future_financial_rows,
        "reason_classification": reason_counts,
        "fact_qualified_events": qualified.height,
        "fact_qualified_symbols": qualified["symbol"].n_unique(),
        "fact_qualified_years": fact_qualified_years,
        "model_review_candidates": sum(
            reason_counts.get(key, 0) for key in ("MIXED", "UNCLASSIFIED")
        ),
        "by_year": by_year,
        "original_pdf_linkage": {
            "events_with_pdf_identifier": 0,
            "status": "TUSHARE_STRUCTURED_REASON_ONLY",
            "rule": "original PDF is required only before paid semantic scoring",
        },
    }
    decision = evaluate_data_gate(audit)
    events_path = output.with_suffix(".events.parquet")
    _atomic_parquet(enriched, events_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "point_in_time_fact_audit_without_market_outcomes",
        "contract": {
            "event_family": "company_specific_positive_forecast_in_weak_peer_day",
            "operating_fact_standard": "reason_class_OPERATING_only",
            "financial_cutoff": "strictly_before_announcement_date",
            "deepseek_used": False,
            "market_outcomes_read": False,
        },
        "audit": audit,
        "event_table": {
            "path": str(events_path),
            "sha256": _sha256(events_path),
            "rows": enriched.height,
        },
        "decision": decision,
    }
    _atomic_json(payload, output)
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": _sha256(output)},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
