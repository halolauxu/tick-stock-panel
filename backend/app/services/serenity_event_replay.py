"""Bounded 2-10 session event study for the Serenity chokepoint hypothesis.

The chokepoint thesis is a slow research prior.  This module tests the separate
short-horizon claim that a *new, official event* at a candidate company can be
followed by an executable 2-10 session return.  Announcement titles are only a
discovery lane; they never become production BUY signals without document review.

All artifacts live below one isolated research root.  Production market data and
the public strategy registry are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from app.config import settings
from app.services.serenity_pilot import (
    DEFAULT_MAX_DOCUMENT_BYTES,
    DEFAULT_MAX_OCR_PAGES,
    CninfoClient,
    PilotStore,
    _atomic_json,
    _json_default,
    _pilot_lock,
    _stable_hash,
    analyze_pdf,
)

EVENT_REPLAY_VERSION = "1.0.0"
EVENT_HORIZONS = (2, 3, 5, 10)
TRAINING_FRACTION = 2 / 3
DEFAULT_EVENT_COST_BPS = 20.0
DEFAULT_MAX_EVENT_DOCUMENTS = 600
DEFAULT_MAX_EVENT_DOCUMENTS_PER_COMPANY = 10
DEFAULT_MAX_EVENT_RAW_BYTES = 1_000_000_000
DEFAULT_BENCHMARK = "000300.SH"

_ROUTINE_REPORT_RE = re.compile(
    r"(?:年度报告|半年度报告|第一季度报告|第三季度报告|季度报告)(?:摘要|全文)?$"
)
_EVENT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "ORDER_CONTRACT",
        re.compile(r"中标|订单|重大合同|框架协议|采购协议|定点通知|项目预中标"),
        "LONG_CANDIDATE",
    ),
    (
        "CUSTOMER_VALIDATION",
        re.compile(r"客户认证|产品认证|验证通过|供应商资格|批量供货|正式量产"),
        "LONG_CANDIDATE",
    ),
    (
        "PRICE_OR_SUPPLY",
        re.compile(r"产品涨价|价格调整|停产|限产|供应短缺|不可抗力"),
        "LONG_CANDIDATE",
    ),
    (
        "CAPACITY_MILESTONE",
        re.compile(r"扩产|投产|产能建设|建设项目|项目开工|项目竣工|募投项目"),
        "LONG_CANDIDATE",
    ),
    (
        "POSITIVE_EARNINGS_REVISION",
        re.compile(r"业绩预增|扭亏为盈|上修业绩|盈利预测上调"),
        "LONG_CANDIDATE",
    ),
    (
        "RISK_INVALIDATION",
        re.compile(
            r"减持|股权质押|立案|处罚|诉讼|仲裁|项目终止|合同终止|订单取消|"
            r"延期|下修|业绩预减|预计亏损|退市风险|风险提示"
        ),
        "NEGATIVE_VETO",
    ),
)


def classify_event_title(title: str) -> dict[str, Any] | None:
    """Classify an official title as discovery-only; never infer a trade from it."""
    normalized = re.sub(r"\s+", "", str(title or ""))
    normalized = re.sub(r"^(?:关于|公司关于)", "", normalized)
    if not normalized or _ROUTINE_REPORT_RE.search(normalized):
        return None
    hits = [
        (event_type, polarity)
        for event_type, pattern, polarity in _EVENT_PATTERNS
        if pattern.search(normalized)
    ]
    if not hits:
        return None
    event_types = [event_type for event_type, _ in hits]
    polarities = {polarity for _, polarity in hits}
    polarity = next(iter(polarities)) if len(polarities) == 1 else "MIXED_REVIEW"
    return {
        "primary_event_type": event_types[0],
        "event_types": event_types,
        "polarity": polarity,
        "status": "DISCOVERY_ONLY_REVIEW_REQUIRED",
    }


class EventReplayStore(PilotStore):
    def _initialize(self) -> None:
        super()._initialize()
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_daily_prices (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                pre_close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                adj_factor DOUBLE,
                asset_type VARCHAR NOT NULL,
                collected_at TIMESTAMP NOT NULL,
                PRIMARY KEY (symbol, date)
            );
            CREATE TABLE IF NOT EXISTS price_collection_status (
                symbol VARCHAR PRIMARY KEY,
                asset_type VARCHAR NOT NULL,
                query_start DATE NOT NULL,
                query_end DATE NOT NULL,
                row_count INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                error VARCHAR,
                checked_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_candidates (
                event_id VARCHAR PRIMARY KEY,
                announcement_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                published_at TIMESTAMP,
                title VARCHAR NOT NULL,
                primary_event_type VARCHAR NOT NULL,
                event_types_json VARCHAR NOT NULL,
                polarity VARCHAR NOT NULL,
                metadata_hash VARCHAR NOT NULL,
                decision_date DATE,
                entry_date DATE,
                status VARCHAR NOT NULL,
                discovered_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS announcement_collection_status (
                symbol VARCHAR PRIMARY KEY,
                query_start DATE NOT NULL,
                query_end DATE NOT NULL,
                row_count INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                error VARCHAR,
                checked_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_model_reviews (
                event_id VARCHAR NOT NULL,
                input_hash VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                raw_output_json VARCHAR NOT NULL,
                output_hash VARCHAR NOT NULL,
                review_status VARCHAR NOT NULL,
                reviewed_at TIMESTAMP NOT NULL,
                PRIMARY KEY (event_id, input_hash)
            );
            CREATE TABLE IF NOT EXISTS event_signals (
                event_id VARCHAR PRIMARY KEY,
                decision_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                score DOUBLE,
                input_hash VARCHAR NOT NULL,
                frozen_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_discovery_outcomes (
                event_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                primary_event_type VARCHAR NOT NULL,
                horizon INTEGER NOT NULL,
                decision_date DATE,
                entry_date DATE,
                exit_date DATE,
                gross_return DOUBLE,
                net_return DOUBLE,
                benchmark_return DOUBLE,
                chain_return DOUBLE,
                mae DOUBLE,
                mfe DOUBLE,
                status VARCHAR NOT NULL,
                settled_at TIMESTAMP,
                PRIMARY KEY (event_id, horizon)
            );
            """
        )


def _copy_universe(source: PilotStore, target: EventReplayStore) -> int:
    rows = source.connection.execute(
        "SELECT * FROM universe ORDER BY chain_id, sample_rank"
    ).fetchall()
    if rows:
        target.connection.executemany(
            "INSERT OR IGNORE INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def initialize_event_replay(
    root: Path,
    source_root: Path,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    if start_date >= end_date:
        raise ValueError("event replay requires start_date before end_date")
    source = PilotStore(source_root)
    target = EventReplayStore(root)
    try:
        existing = target.get_meta("event_manifest")
        if existing:
            expected = (start_date.isoformat(), end_date.isoformat())
            actual = (existing.get("start_date"), existing.get("end_date"))
            if actual != expected:
                raise RuntimeError("event replay root is bound to a different date range")
            return existing
        count = _copy_universe(source, target)
        if count != 100:
            raise RuntimeError(f"event replay requires the frozen 100-company universe; got {count}")
        universe_hash = _stable_hash(
            *[
                f"{row['symbol']}:{row['chain_id']}"
                for row in sorted(target.universe(), key=lambda item: item["symbol"])
            ]
        )
        manifest = {
            "version": EVENT_REPLAY_VERSION,
            "replay_id": root.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "universe_size": count,
            "universe_hash": universe_hash,
            "source_universe_root": source_root.name,
            "strategy_contract": {
                "research_prior": "verified_chokepoint",
                "trigger": "new_official_event",
                "decision_time": "after_close",
                "entry": "next_global_trading_day_open",
                "horizons": list(EVENT_HORIZONS),
                "maximum_holding_sessions": 10,
                "cost_bps": DEFAULT_EVENT_COST_BPS,
                "benchmark": DEFAULT_BENCHMARK,
                "title_classification_is_trade_signal": False,
                "model_review_required_for_trade_signal": True,
                "validation": "chronological_first_two_thirds_train_last_third_validate",
            },
            "qualification": {
                "status": "RETROSPECTIVE_EVENT_DISCOVERY_NOT_ALPHA",
                "universe_membership": "CURRENT_SNAPSHOT_RETROSPECTIVE_BIAS",
                "announcement_source": "CNINFO_OFFICIAL",
                "price_source": "LOCAL_ENRICHED_DAILY_AND_INDEX",
                "price_window": "ACTUAL_LOCAL_COVERAGE_ONLY",
                "claim_boundary": "EVENT_STUDY_ONLY_NO_BUY_SIGNAL",
            },
            "limits": {
                "max_documents": DEFAULT_MAX_EVENT_DOCUMENTS,
                "max_documents_per_company": DEFAULT_MAX_EVENT_DOCUMENTS_PER_COMPANY,
                "max_raw_bytes": DEFAULT_MAX_EVENT_RAW_BYTES,
                "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
                "max_ocr_pages": DEFAULT_MAX_OCR_PAGES,
            },
        }
        target.set_meta("event_manifest", manifest)
        _atomic_json(root / "event-manifest.json", manifest)
        return manifest
    finally:
        source.close()
        target.close()


def _normalize_local_rows(rows: list[dict[str, Any]], asset_type: str) -> list[list[Any]]:
    now = datetime.now()
    values: list[list[Any]] = []
    for row in rows:
        raw_date = row.get("date")
        if not isinstance(raw_date, date):
            continue
        required = [row.get(field) for field in ("open", "high", "low", "close")]
        if any(value is None or not math.isfinite(float(value)) for value in required):
            continue
        open_, high, low, close = (float(value) for value in required)
        volume = float(row["volume"]) if row.get("volume") is not None else None
        if (
            min(open_, high, low, close) <= 0
            or high < max(open_, close)
            or low > min(open_, close)
            or high < low
            or (volume is not None and volume < 0)
        ):
            continue
        values.append(
            [
                str(row.get("symbol") or ""),
                raw_date,
                open_,
                high,
                low,
                close,
                float(row["pre_close"]) if row.get("pre_close") is not None else None,
                volume,
                float(row["amount"]) if row.get("amount") is not None else None,
                1.0,
                asset_type,
                now,
            ]
        )
    return values


def _local_price_window(data_dir: Path) -> tuple[date, date]:
    date_sets: list[set[date]] = []
    for table in ("kline_daily_enriched", "kline_index_daily"):
        values: set[date] = set()
        for path in (data_dir / table).glob("date=*"):
            try:
                values.add(date.fromisoformat(path.name.removeprefix("date=")))
            except ValueError:
                continue
        if not values:
            raise RuntimeError(f"local {table} has no dated partitions")
        date_sets.append(values)
    common_dates = date_sets[0] & date_sets[1]
    if not common_dates:
        raise RuntimeError("stock and index daily stores have no common date window")
    return min(common_dates), max(common_dates)


def resolve_local_price_window(
    data_dir: Path,
    requested_start: date | None = None,
    requested_end: date | None = None,
) -> tuple[date, date]:
    available_start, available_end = _local_price_window(data_dir)
    start = requested_start or available_start
    end = requested_end or available_end
    if start < available_start or end > available_end:
        raise RuntimeError(
            "requested replay range exceeds local price coverage "
            f"{available_start.isoformat()}..{available_end.isoformat()}"
        )
    if start >= end:
        raise ValueError("event replay requires start_date before end_date")
    return start, end


def _read_local_prices(
    data_dir: Path,
    table: str,
    symbols: list[str],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    path = data_dir / table
    files = list(path.glob("date=*/*.parquet"))
    if not files:
        raise RuntimeError(f"local {table} has no parquet files")
    frame = (
        pl.scan_parquet(str(path / "date=*" / "*.parquet"))
        .select("symbol", "date", "open", "high", "low", "close", "volume", "amount")
        .filter(
            pl.col("symbol").is_in(symbols)
            & pl.col("date").is_between(start, end, closed="both")
        )
        .sort("symbol", "date")
        .collect()
    )
    duplicate_count = (
        frame.group_by("symbol", "date")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise RuntimeError(f"local {table} has {duplicate_count} duplicate symbol-date keys")
    frame = frame.with_columns(
        pl.col("close").shift(1).over("symbol").alias("pre_close")
    )
    return frame.to_dicts()


def collect_research_prices(store: EventReplayStore, data_dir: Path) -> dict[str, Any]:
    manifest = store.get_meta("event_manifest")
    start = date.fromisoformat(manifest["start_date"])
    end = date.fromisoformat(manifest["end_date"])
    targets = [(row["symbol"], "stock") for row in store.universe()]
    targets.append((DEFAULT_BENCHMARK, "index"))
    already_complete = {
        row[0]
        for row in store.connection.execute(
            """
            SELECT symbol FROM price_collection_status
            WHERE query_start=? AND query_end=? AND status='ok'
            """,
            [start, end],
        ).fetchall()
    }
    remaining = [(symbol, kind) for symbol, kind in targets if symbol not in already_complete]
    if not remaining:
        return {
            "queried": 0,
            "inserted_rows": 0,
            "failures": 0,
            "validation_split": freeze_validation_split(store),
        }
    stock_symbols = [symbol for symbol, kind in remaining if kind == "stock"]
    raw_rows: list[tuple[str, list[dict[str, Any]]]] = []
    if stock_symbols:
        raw_rows.append(
            (
                "stock",
                _read_local_prices(
                    data_dir,
                    "kline_daily_enriched",
                    stock_symbols,
                    start,
                    end,
                ),
            )
        )
    if (DEFAULT_BENCHMARK, "index") in remaining:
        raw_rows.append(
            (
                "index",
                _read_local_prices(
                    data_dir,
                    "kline_index_daily",
                    [DEFAULT_BENCHMARK],
                    start,
                    end,
                ),
            )
        )
    values_by_symbol: dict[str, list[list[Any]]] = defaultdict(list)
    rejected_by_symbol: dict[str, int] = defaultdict(int)
    for asset_type, rows in raw_rows:
        for row in rows:
            normalized = _normalize_local_rows([row], asset_type)
            symbol = str(row.get("symbol") or "")
            if normalized:
                values_by_symbol[symbol].extend(normalized)
            else:
                rejected_by_symbol[symbol] += 1
    inserted = failures = 0
    all_values: list[list[Any]] = []
    status_values: list[list[Any]] = []
    for symbol, asset_type in remaining:
        values = values_by_symbol.get(symbol, [])
        status = "ok" if values else "empty"
        error = (
            f"rejected_invalid_rows={rejected_by_symbol[symbol]}"
            if rejected_by_symbol[symbol]
            else None
        )
        if values:
            all_values.extend(values)
            inserted += len(values)
        else:
            failures += 1
        status_values.append(
            [symbol, asset_type, start, end, len(values), status, error, datetime.now()]
        )
    store.connection.execute("BEGIN TRANSACTION")
    try:
        if all_values:
            store.connection.executemany(
                "INSERT OR REPLACE INTO research_daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                all_values,
            )
        store.connection.executemany(
            "INSERT OR REPLACE INTO price_collection_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            status_values,
        )
        store.connection.execute("COMMIT")
    except Exception:
        store.connection.execute("ROLLBACK")
        raise
    return {
        "queried": len(remaining),
        "inserted_rows": inserted,
        "failures": failures,
        "validation_split": freeze_validation_split(store),
    }


def freeze_validation_split(store: EventReplayStore) -> dict[str, Any]:
    calendar = [
        row[0]
        for row in store.connection.execute(
            "SELECT date FROM research_daily_prices WHERE symbol=? ORDER BY date",
            [DEFAULT_BENCHMARK],
        ).fetchall()
    ]
    if len(calendar) < 3:
        raise RuntimeError("at least three benchmark sessions are required for validation split")
    training_sessions = max(1, min(len(calendar) - 1, int(len(calendar) * TRAINING_FRACTION)))
    split = {
        "method": "CHRONOLOGICAL_FIRST_TWO_THIRDS_TRAIN_LAST_THIRD_VALIDATE",
        "calendar_hash": _stable_hash(*[value.isoformat() for value in calendar]),
        "training_start": calendar[0].isoformat(),
        "training_end": calendar[training_sessions - 1].isoformat(),
        "training_sessions": training_sessions,
        "validation_start": calendar[training_sessions].isoformat(),
        "validation_end": calendar[-1].isoformat(),
        "validation_sessions": len(calendar) - training_sessions,
    }
    existing = store.get_meta("validation_split")
    if existing and existing != split:
        raise RuntimeError("frozen validation split conflicts with current local calendar")
    if not existing:
        store.set_meta("validation_split", split)
        _atomic_json(store.root / "validation-split.json", split)
    return split


def collect_event_metadata(store: EventReplayStore) -> dict[str, Any]:
    manifest = store.get_meta("event_manifest")
    start = date.fromisoformat(manifest["start_date"])
    end = date.fromisoformat(manifest["end_date"])
    completed = {
        row[0]
        for row in store.connection.execute(
            """
            SELECT symbol FROM announcement_collection_status
            WHERE query_start=? AND query_end=? AND status IN ('ok', 'empty')
            """,
            [start, end],
        ).fetchall()
    }
    companies = [row for row in store.universe() if row["symbol"] not in completed]
    client = CninfoClient()
    discovered = candidates = failures = 0
    try:
        for company in companies:
            try:
                rows = client.announcements(company["code"], start, end)
            except Exception as exc:
                failures += 1
                store.connection.execute(
                    "INSERT OR REPLACE INTO announcement_collection_status VALUES (?, ?, ?, 0, 'failed', ?, ?)",
                    [company["symbol"], start, end, str(exc)[:300], datetime.now()],
                )
                continue
            announcement_values: list[list[Any]] = []
            candidate_values: list[list[Any]] = []
            for item in rows:
                announcement_values.append(
                    [
                        item["announcement_id"],
                        company["symbol"],
                        item["announce_time"],
                        item["title"],
                        item["pdf_url"],
                        item["announced_size_kb"],
                        datetime.now(),
                    ]
                )
                classification = classify_event_title(item["title"])
                if classification is None:
                    continue
                metadata_hash = _stable_hash(
                    item["announcement_id"],
                    company["symbol"],
                    str(item["announce_time"]),
                    item["title"],
                    item["pdf_url"],
                )
                candidate_values.append(
                    [
                        item["announcement_id"],
                        item["announcement_id"],
                        company["symbol"],
                        item["announce_time"],
                        item["title"],
                        classification["primary_event_type"],
                        json.dumps(classification["event_types"], ensure_ascii=False),
                        classification["polarity"],
                        metadata_hash,
                        classification["status"],
                        datetime.now(),
                    ]
                )
            try:
                store.connection.execute("BEGIN TRANSACTION")
                if announcement_values:
                    store.connection.executemany(
                        """
                        INSERT OR IGNORE INTO announcements VALUES
                        (?, ?, ?, ?, ?, ?, 'discovered', NULL, ?)
                        """,
                        announcement_values,
                    )
                if candidate_values:
                    store.connection.executemany(
                        """
                        INSERT OR IGNORE INTO event_candidates VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                        """,
                        candidate_values,
                    )
                store.connection.execute(
                    "INSERT OR REPLACE INTO announcement_collection_status VALUES (?, ?, ?, ?, ?, NULL, ?)",
                    [
                        company["symbol"],
                        start,
                        end,
                        len(rows),
                        "ok" if rows else "empty",
                        datetime.now(),
                    ],
                )
                store.connection.execute("COMMIT")
                discovered += len(announcement_values)
                candidates += len(candidate_values)
            except Exception as exc:
                store.connection.execute("ROLLBACK")
                failures += 1
                store.connection.execute(
                    "INSERT OR REPLACE INTO announcement_collection_status VALUES (?, ?, ?, 0, 'failed', ?, ?)",
                    [company["symbol"], start, end, str(exc)[:300], datetime.now()],
                )
        return {
            "queried_companies": len(companies),
            "query_failures": failures,
            "discovered_announcements": discovered,
            "event_candidates": candidates,
        }
    finally:
        client.close()


def collect_event_documents(
    store: EventReplayStore,
    *,
    max_new_documents: int | None = None,
) -> dict[str, Any]:
    if max_new_documents is not None and max_new_documents < 1:
        raise ValueError("max_new_documents must be positive")
    current_docs, current_bytes = store.connection.execute(
        "SELECT count(*), coalesce(sum(pdf_bytes), 0) FROM document_metrics"
    ).fetchone()
    per_symbol = dict(
        store.connection.execute(
            """
            SELECT a.symbol, count(*) FROM announcements a
            JOIN document_metrics d USING (announcement_id)
            GROUP BY a.symbol
            """
        ).fetchall()
    )
    candidates = store.connection.execute(
        """
        SELECT e.event_id, e.symbol, a.pdf_url
        FROM event_candidates e JOIN announcements a USING (announcement_id)
        WHERE e.polarity IN ('LONG_CANDIDATE', 'MIXED_REVIEW')
        ORDER BY e.published_at, e.symbol, e.event_id
        """
    ).fetchall()
    client = CninfoClient()
    downloaded = downloaded_bytes = failed = capped = 0
    try:
        for event_id, symbol, pdf_url in candidates:
            if max_new_documents is not None and downloaded >= max_new_documents:
                break
            if store.connection.execute(
                "SELECT 1 FROM document_metrics WHERE announcement_id=?", [event_id]
            ).fetchone():
                continue
            if current_docs >= DEFAULT_MAX_EVENT_DOCUMENTS or current_bytes >= DEFAULT_MAX_EVENT_RAW_BYTES:
                capped += 1
                store.connection.execute(
                    "UPDATE announcements SET status='capped', error='event replay storage cap reached' WHERE announcement_id=?",
                    [event_id],
                )
                continue
            if int(per_symbol.get(symbol, 0)) >= DEFAULT_MAX_EVENT_DOCUMENTS_PER_COMPANY:
                capped += 1
                store.connection.execute(
                    "UPDATE announcements SET status='capped', error='event replay company cap reached' WHERE announcement_id=?",
                    [event_id],
                )
                continue
            pdf_path = store.documents_dir / f"{event_id}.pdf"
            text_path = store.text_dir / f"{event_id}.txt"
            try:
                size = client.download_pdf(pdf_url, pdf_path, DEFAULT_MAX_DOCUMENT_BYTES)
                if current_bytes + size > DEFAULT_MAX_EVENT_RAW_BYTES:
                    pdf_path.unlink(missing_ok=True)
                    capped += 1
                    store.connection.execute(
                        "UPDATE announcements SET status='capped', error='event replay storage cap reached' WHERE announcement_id=?",
                        [event_id],
                    )
                    continue
                metrics, facts = analyze_pdf(pdf_path, text_path, event_id)
                store.connection.execute("BEGIN TRANSACTION")
                try:
                    store.connection.execute(
                        "INSERT INTO document_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            metrics["announcement_id"], metrics["sha256"], metrics["pages"],
                            metrics["pdf_bytes"], metrics["embedded_text_bytes"],
                            metrics["extracted_text_bytes"], metrics["ocr_text_bytes"],
                            metrics["ocr_pages"], metrics["low_text_pages"],
                            metrics["rendered_png_bytes"], metrics["persistent_inflation_pct"],
                            metrics["ocr_render_multiplier"], metrics["fact_count"],
                            metrics["parse_status"], metrics["measured_at"],
                        ],
                    )
                    if facts:
                        store.connection.executemany(
                            "INSERT OR IGNORE INTO evidence_facts VALUES (?, ?, ?, ?, ?, ?)",
                            [
                                [
                                    fact["fact_id"],
                                    fact["announcement_id"],
                                    fact["page_number"],
                                    fact["category"],
                                    fact["evidence_sentence"],
                                    fact["review_status"],
                                ]
                                for fact in facts
                            ],
                        )
                    store.connection.execute(
                        "UPDATE announcements SET status='measured', error=NULL WHERE announcement_id=?",
                        [event_id],
                    )
                    store.connection.execute("COMMIT")
                except Exception:
                    store.connection.execute("ROLLBACK")
                    raise
                current_docs += 1
                current_bytes += size
                per_symbol[symbol] = int(per_symbol.get(symbol, 0)) + 1
                downloaded += 1
                downloaded_bytes += size
            except Exception as exc:
                failed += 1
                pdf_path.unlink(missing_ok=True)
                text_path.unlink(missing_ok=True)
                store.connection.execute(
                    "UPDATE announcements SET status='failed', error=? WHERE announcement_id=?",
                    [str(exc)[:300], event_id],
                )
        return {
            "downloaded_documents": downloaded,
            "downloaded_bytes": downloaded_bytes,
            "failed": failed,
            "capped": capped,
        }
    finally:
        client.close()


def materialize_event_dates(store: EventReplayStore) -> dict[str, int]:
    trading_dates = [
        row[0]
        for row in store.connection.execute(
            "SELECT date FROM research_daily_prices WHERE symbol=? ORDER BY date",
            [DEFAULT_BENCHMARK],
        ).fetchall()
    ]
    if not trading_dates:
        raise RuntimeError("benchmark trading calendar is missing")
    updated = unresolved = 0
    for event_id, published_at in store.connection.execute(
        "SELECT event_id, published_at FROM event_candidates ORDER BY event_id"
    ).fetchall():
        if published_at is None:
            unresolved += 1
            store.connection.execute(
                "UPDATE event_candidates SET status='MISSING_PUBLISHED_AT' WHERE event_id=?",
                [event_id],
            )
            continue
        published_date = published_at.date()
        decision_index = next(
            (index for index, value in enumerate(trading_dates) if value >= published_date),
            None,
        )
        if decision_index is None or decision_index + 1 >= len(trading_dates):
            unresolved += 1
            store.connection.execute(
                "UPDATE event_candidates SET status='OUTSIDE_PRICE_WINDOW' WHERE event_id=?",
                [event_id],
            )
            continue
        store.connection.execute(
            """
            UPDATE event_candidates SET decision_date=?, entry_date=?,
                status='DISCOVERY_ONLY_REVIEW_REQUIRED' WHERE event_id=?
            """,
            [trading_dates[decision_index], trading_dates[decision_index + 1], event_id],
        )
        updated += 1
    return {"updated": updated, "unresolved": unresolved}


def _price_map(store: EventReplayStore) -> dict[str, dict[date, dict[str, float | None]]]:
    result: dict[str, dict[date, dict[str, float | None]]] = defaultdict(dict)
    rows = store.connection.execute(
        """
        SELECT symbol, date, open, high, low, close, pre_close, volume, adj_factor
        FROM research_daily_prices ORDER BY symbol, date
        """
    ).fetchall()
    for symbol, day, open_, high, low, close, pre_close, volume, adj_factor in rows:
        result[symbol][day] = {
            "open": float(open_), "high": float(high), "low": float(low),
            "close": float(close),
            "pre_close": float(pre_close) if pre_close is not None else None,
            "volume": float(volume) if volume is not None else None,
            "adj_factor": float(adj_factor) if adj_factor is not None else None,
        }
    return result


def _adjusted_price(row: dict[str, float | None], field: str) -> float | None:
    value = row.get(field)
    factor = row.get("adj_factor")
    if value is None or factor is None or factor <= 0:
        return None
    return float(value) * float(factor)


def _is_locked_one_price(symbol: str, row: dict[str, float | None]) -> bool:
    high = row.get("high")
    low = row.get("low")
    open_ = row.get("open")
    pre_close = row.get("pre_close")
    if high is None or low is None or open_ is None or pre_close in (None, 0):
        return False
    if float(high) != float(low):
        return False
    code = symbol.split(".", 1)[0]
    if code.startswith(("300", "301", "688", "689")):
        limit_threshold = 0.195
    elif code.startswith(("4", "8", "920")):
        limit_threshold = 0.295
    else:
        # A 5% threshold also rejects ST one-price limit boards conservatively.
        limit_threshold = 0.048
    return abs(float(open_) / float(pre_close) - 1) >= limit_threshold


def settle_discovery_outcomes(store: EventReplayStore) -> dict[str, int]:
    """Settle title-discovery events; these rows are explicitly not trade signals."""
    prices = _price_map(store)
    calendar = sorted(prices.get(DEFAULT_BENCHMARK, {}))
    if not calendar:
        raise RuntimeError("benchmark prices are missing")
    universe = store.universe()
    chain_by_symbol = {row["symbol"]: row["chain_id"] for row in universe}
    peers: dict[str, list[str]] = defaultdict(list)
    for row in universe:
        peers[row["chain_id"]].append(row["symbol"])
    events = store.connection.execute(
        """
        SELECT e.event_id, e.symbol, e.primary_event_type, e.decision_date, e.entry_date
        FROM event_candidates e
        JOIN document_metrics d ON d.announcement_id=e.announcement_id
        WHERE e.polarity='LONG_CANDIDATE' AND e.decision_date IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM event_candidates veto
              WHERE veto.symbol=e.symbol
                AND veto.decision_date=e.decision_date
                AND veto.polarity IN ('NEGATIVE_VETO', 'MIXED_REVIEW')
          )
        QUALIFY row_number() OVER (
            PARTITION BY e.symbol, e.decision_date
            ORDER BY e.published_at, e.event_id
        ) = 1
        ORDER BY e.decision_date, e.symbol, e.event_id
        """
    ).fetchall()
    settled = pending = unexecutable = 0
    now = datetime.now()
    for event_id, symbol, event_type, decision_date, entry_date in events:
        symbol_prices = prices.get(symbol, {})
        entry = symbol_prices.get(entry_date)
        if entry is None or not entry.get("volume") or _is_locked_one_price(symbol, entry):
            for horizon in EVENT_HORIZONS:
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                    (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     'UNEXECUTABLE_NEXT_OPEN', ?)
                    """,
                    [event_id, symbol, event_type, horizon, decision_date, entry_date, now],
                )
                unexecutable += 1
            continue
        entry_index = calendar.index(entry_date)
        entry_price = _adjusted_price(entry, "open")
        if entry_price is None:
            for horizon in EVENT_HORIZONS:
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                    (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     'MISSING_ADJ_FACTOR', ?)
                    """,
                    [event_id, symbol, event_type, horizon, decision_date, entry_date, now],
                )
            continue
        for horizon in EVENT_HORIZONS:
            exit_index = entry_index + horizon - 1
            if exit_index >= len(calendar):
                status = "PENDING_OUTCOME"
                exit_date = None
                pending += 1
            else:
                exit_date = calendar[exit_index]
                exit_row = symbol_prices.get(exit_date)
                status = "SETTLED" if exit_row is not None else "MISSING_STOCK_BAR"
            if status != "SETTLED":
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                    (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
                    """,
                    [
                        event_id, symbol, event_type, horizon, decision_date, entry_date,
                        exit_date, status, now,
                    ],
                )
                continue
            window_dates = calendar[entry_index : exit_index + 1]
            window = [symbol_prices.get(value) for value in window_dates]
            if any(row is None for row in window):
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                    (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL,
                     'MISSING_INTERMEDIATE_BAR', ?)
                    """,
                    [
                        event_id,
                        symbol,
                        event_type,
                        horizon,
                        decision_date,
                        entry_date,
                        exit_date,
                        now,
                    ],
                )
                continue
            exit_row = symbol_prices[exit_date]
            exit_price = _adjusted_price(exit_row, "close")
            adjusted_lows = [_adjusted_price(row, "low") for row in window if row]
            adjusted_highs = [_adjusted_price(row, "high") for row in window if row]
            if (
                exit_price is None
                or any(value is None for value in adjusted_lows)
                or any(value is None for value in adjusted_highs)
            ):
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                    (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL,
                     'MISSING_ADJ_FACTOR', ?)
                    """,
                    [
                        event_id,
                        symbol,
                        event_type,
                        horizon,
                        decision_date,
                        entry_date,
                        exit_date,
                        now,
                    ],
                )
                continue
            gross = exit_price / entry_price - 1
            net = gross - DEFAULT_EVENT_COST_BPS / 10_000
            benchmark_prices = prices[DEFAULT_BENCHMARK]
            benchmark_entry = benchmark_prices.get(entry_date)
            benchmark_exit = benchmark_prices.get(exit_date)
            benchmark = (
                float(benchmark_exit["close"]) / float(benchmark_entry["open"]) - 1
                if benchmark_entry and benchmark_exit
                else None
            )
            chain_returns: list[float] = []
            for peer in peers.get(chain_by_symbol.get(symbol, ""), []):
                if peer == symbol:
                    continue
                peer_entry = prices.get(peer, {}).get(entry_date)
                peer_exit = prices.get(peer, {}).get(exit_date)
                if peer_entry and peer_exit and float(peer_entry["open"]):
                    chain_returns.append(
                        float(peer_exit["close"]) / float(peer_entry["open"]) - 1
                    )
            chain_return = sum(chain_returns) / len(chain_returns) if chain_returns else None
            store.connection.execute(
                """
                INSERT OR REPLACE INTO event_discovery_outcomes VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SETTLED', ?)
                """,
                [
                    event_id, symbol, event_type, horizon, decision_date, entry_date, exit_date,
                    gross, net, benchmark, chain_return,
                    min(float(value) / entry_price - 1 for value in adjusted_lows),
                    max(float(value) / entry_price - 1 for value in adjusted_highs),
                    now,
                ],
            )
            settled += 1
    return {"events": len(events), "settled": settled, "pending": pending, "unexecutable": unexecutable}


def persist_model_review(
    store: EventReplayStore,
    *,
    event_id: str,
    input_hash: str,
    model: str,
    raw_output: dict[str, Any],
    review_status: str,
) -> dict[str, Any]:
    """Persist a paid semantic result idempotently before downstream processing."""
    raw_json = json.dumps(raw_output, ensure_ascii=False, sort_keys=True)
    output_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    existing = store.connection.execute(
        "SELECT output_hash, raw_output_json FROM event_model_reviews WHERE event_id=? AND input_hash=?",
        [event_id, input_hash],
    ).fetchone()
    if existing:
        if existing[0] != output_hash:
            raise RuntimeError("immutable event model review conflicts with cached output")
        return {"status": "cached", "output_hash": existing[0], "raw_output": json.loads(existing[1])}
    store.connection.execute(
        "INSERT INTO event_model_reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
        [event_id, input_hash, model, raw_json, output_hash, review_status, datetime.now()],
    )
    return {"status": "stored", "output_hash": output_hash, "raw_output": raw_output}


def build_event_report(store: EventReplayStore) -> dict[str, Any]:
    manifest = store.get_meta("event_manifest")
    validation_split = store.get_meta("validation_split")
    if not validation_split:
        benchmark_rows = store.connection.execute(
            "SELECT count(*) FROM research_daily_prices WHERE symbol=?",
            [DEFAULT_BENCHMARK],
        ).fetchone()[0]
        validation_split = (
            freeze_validation_split(store)
            if benchmark_rows >= 3
            else {"status": "PENDING_PRICE_COLLECTION"}
        )
    totals = store.connection.execute(
        """
        SELECT count(*),
               count(*) FILTER (WHERE polarity='LONG_CANDIDATE'),
               count(*) FILTER (WHERE polarity='NEGATIVE_VETO'),
               count(*) FILTER (WHERE polarity='MIXED_REVIEW')
        FROM event_candidates
        """
    ).fetchone()
    documents = store.connection.execute(
        "SELECT count(*), coalesce(sum(pdf_bytes), 0) FROM document_metrics"
    ).fetchone()
    price_status = store.connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE status='ok'), coalesce(sum(row_count), 0) FROM price_collection_status"
    ).fetchone()
    stock_coverage = store.connection.execute(
        """
        SELECT min(row_count), median(row_count), max(row_count)
        FROM price_collection_status WHERE asset_type='stock' AND status='ok'
        """
    ).fetchone()
    announcement_status = store.connection.execute(
        """
        SELECT count(*),
               count(*) FILTER (WHERE status IN ('ok', 'empty')),
               count(*) FILTER (WHERE status='failed')
        FROM announcement_collection_status
        """
    ).fetchone()
    estimated_candidate_sizes = [
        float(row[0]) * 1024
        for row in store.connection.execute(
            """
            WITH ranked AS (
                SELECT a.announced_size_kb,
                       row_number() OVER (
                           PARTITION BY e.symbol
                           ORDER BY e.published_at, e.event_id
                       ) AS company_rank
                FROM event_candidates e
                JOIN announcements a USING (announcement_id)
                WHERE e.polarity IN ('LONG_CANDIDATE', 'MIXED_REVIEW')
                  AND a.announced_size_kb IS NOT NULL
            )
            SELECT announced_size_kb FROM ranked
            WHERE company_rank <= ?
            ORDER BY company_rank
            LIMIT ?
            """,
            [DEFAULT_MAX_EVENT_DOCUMENTS_PER_COMPANY, DEFAULT_MAX_EVENT_DOCUMENTS],
        ).fetchall()
    ]
    outcome_rows = store.connection.execute(
        """
        SELECT primary_event_type, horizon, count(*), avg(net_return), median(net_return),
               stddev_samp(net_return), avg(CASE WHEN net_return > 0 THEN 1.0 ELSE 0.0 END),
               avg(net_return-benchmark_return), avg(net_return-chain_return),
               min(mae), max(mfe),
               CASE WHEN count(*) > 1 AND stddev_samp(net_return) > 0
                    THEN avg(net_return) / stddev_samp(net_return) * sqrt(count(*))
                    ELSE NULL END
        FROM event_discovery_outcomes WHERE status='SETTLED'
        GROUP BY primary_event_type, horizon ORDER BY primary_event_type, horizon
        """
    ).fetchall()
    split_outcome_rows = []
    if validation_split.get("validation_start"):
        validation_start = date.fromisoformat(validation_split["validation_start"])
        split_outcome_rows = store.connection.execute(
            """
            SELECT CASE WHEN decision_date < ? THEN 'TRAIN' ELSE 'VALIDATION' END AS sample,
                   primary_event_type, horizon, count(*), avg(net_return), median(net_return),
                   stddev_samp(net_return), avg(CASE WHEN net_return > 0 THEN 1.0 ELSE 0.0 END),
                   avg(net_return-benchmark_return), avg(net_return-chain_return),
                   min(mae), max(mfe),
                   CASE WHEN count(*) > 1 AND stddev_samp(net_return) > 0
                        THEN avg(net_return) / stddev_samp(net_return) * sqrt(count(*))
                        ELSE NULL END
            FROM event_discovery_outcomes WHERE status='SETTLED'
            GROUP BY sample, primary_event_type, horizon
            ORDER BY sample, primary_event_type, horizon
            """,
            [validation_start],
        ).fetchall()
    report = {
        "replay_id": manifest.get("replay_id"),
        "period": {"start": manifest.get("start_date"), "end": manifest.get("end_date")},
        "validation_split": validation_split,
        "universe_size": manifest.get("universe_size"),
        "events": {
            "total": int(totals[0]),
            "long_discovery": int(totals[1]),
            "risk_veto": int(totals[2]),
            "mixed_review": int(totals[3]),
        },
        "documents": {"measured": int(documents[0]), "raw_bytes": int(documents[1])},
        "document_plan": {
            "sized_candidates_within_caps": len(estimated_candidate_sizes),
            "announced_bytes_estimate": int(sum(estimated_candidate_sizes)),
            "hard_document_cap": DEFAULT_MAX_EVENT_DOCUMENTS,
            "hard_raw_bytes_cap": DEFAULT_MAX_EVENT_RAW_BYTES,
        },
        "prices": {
            "targets": int(price_status[0]),
            "successful_targets": int(price_status[1]),
            "rows": int(price_status[2]),
            "stock_rows_min": stock_coverage[0],
            "stock_rows_median": stock_coverage[1],
            "stock_rows_max": stock_coverage[2],
        },
        "announcement_collection": {
            "targets": int(announcement_status[0]),
            "completed_targets": int(announcement_status[1]),
            "failed_targets": int(announcement_status[2]),
        },
        "discovery_event_study": [
            {
                "event_type": row[0], "horizon": int(row[1]), "observations": int(row[2]),
                "mean_net_return": row[3], "median_net_return": row[4],
                "standard_deviation": row[5], "win_rate": row[6],
                "mean_alpha_vs_csi300": row[7], "mean_alpha_vs_chain": row[8],
                "worst_mae": row[9], "best_mfe": row[10],
                "mean_return_t_stat": row[11],
            }
            for row in outcome_rows
        ],
        "discovery_event_study_by_split": [
            {
                "sample": row[0], "event_type": row[1], "horizon": int(row[2]),
                "observations": int(row[3]), "mean_net_return": row[4],
                "median_net_return": row[5], "standard_deviation": row[6],
                "win_rate": row[7], "mean_alpha_vs_csi300": row[8],
                "mean_alpha_vs_chain": row[9], "worst_mae": row[10],
                "best_mfe": row[11], "mean_return_t_stat": row[12],
            }
            for row in split_outcome_rows
        ],
        "trade_signals": store.connection.execute("SELECT count(*) FROM event_signals").fetchone()[0],
        "model_reviews": store.connection.execute("SELECT count(*) FROM event_model_reviews").fetchone()[0],
        "alpha_status": "UNVERIFIED_ALPHA",
        "qualification": manifest.get("qualification"),
        "warning": "公告标题事件研究不是买入信号; 只有原文核验、财务传导和未定价门槛完成后才能生成事件信号",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_json(store.root / "event-report.json", report)
    return report


def run_event_replay(
    root: Path,
    source_root: Path,
    data_dir: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    initialize_event_replay(root, source_root, start_date=start, end_date=end)
    store = EventReplayStore(root)
    try:
        result = {
            "prices": collect_research_prices(store, data_dir),
            "metadata": collect_event_metadata(store),
            "documents": collect_event_documents(store, max_new_documents=100),
            "dates": materialize_event_dates(store),
            "outcomes": settle_discovery_outcomes(store),
        }
        result["report"] = build_event_report(store)
        _atomic_json(root / "event-status.json", result)
        return result
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "collect-prices", "collect-events", "download", "settle", "run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--max-new-documents",
        type=int,
        help="bound one download run without changing the immutable total storage cap",
    )
    args = parser.parse_args(argv)
    with _pilot_lock(args.root):
        if args.command in {"init", "run"} and args.source_root is None:
            parser.error("--source-root is required for init/run")
        if args.command in {"init", "run"}:
            start, end = resolve_local_price_window(
                args.data_dir,
                date.fromisoformat(args.start_date) if args.start_date else None,
                date.fromisoformat(args.end_date) if args.end_date else None,
            )
        if args.command == "init":
            payload = initialize_event_replay(args.root, args.source_root, start_date=start, end_date=end)
        elif args.command == "run":
            payload = run_event_replay(
                args.root,
                args.source_root,
                args.data_dir,
                start,
                end,
            )
        else:
            store = EventReplayStore(args.root)
            try:
                if not store.get_meta("event_manifest"):
                    raise RuntimeError("event replay is not initialized")
                if args.command == "collect-prices":
                    payload = collect_research_prices(store, args.data_dir)
                elif args.command == "collect-events":
                    payload = collect_event_metadata(store)
                elif args.command == "download":
                    payload = collect_event_documents(
                        store,
                        max_new_documents=args.max_new_documents,
                    )
                elif args.command == "settle":
                    payload = {
                        "dates": materialize_event_dates(store),
                        "outcomes": settle_discovery_outcomes(store),
                        "report": build_event_report(store),
                    }
                else:
                    payload = build_event_report(store)
            finally:
                store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
