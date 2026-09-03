"""Collect exact official forecast PDFs for event rows missing reason text.

The collector is bounded, resumable, and outcome-blind.  It queries CNINFO only
for the event's announcement date, stores document provenance in DuckDB, and
persists extracted evidence before the fact audit can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))

from audit_p0_short_horizon_event_facts import classify_reason  # noqa: E402

from app.services.serenity_pilot import (  # noqa: E402
    DEFAULT_MAX_DOCUMENT_BYTES,
    CninfoClient,
    analyze_pdf,
)

SCHEMA_VERSION = "p0-short-horizon-event-documents-v1"
SELECTION_VERSION = "V3"
MAX_TOTAL_PDF_BYTES = 512_000_000
FORECAST_TITLE_PATTERN = re.compile(r"业绩(?:预告|预增|预减|预亏|预盈|扭亏)")
REVISED_TITLE_PATTERN = re.compile(r"修正|更正|补充")
REASON_HEADING_PATTERN = re.compile(
    r"(?:业绩变动(?:的)?(?:主要)?原因(?:说明|分析)?|"
    r"本期业绩(?:预增|变化|变动)的(?:主要)?原因)\s*[:\uff1a]?",
    re.MULTILINE,
)
NEXT_HEADING_PATTERN = re.compile(
    r"\n\s*(?:(?:[一二三四五六七八九十]+)[、.]|(?:\d+)[、.])\s*"
    r"(?:其他|风险|说明|提示|相关|备查|审计|会计师)",
    re.MULTILINE,
)
EVENT_KEY_FIELDS = (
    "symbol",
    "ann_date",
    "period_end",
    "type",
    "p_change_min",
    "p_change_max",
    "net_profit_min",
    "net_profit_max",
)


def _canonical(value: Any) -> str | float | int | None:
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (str, int)):
        return value
    return str(value)


def event_key(event: dict[str, Any]) -> str:
    payload = {field: _canonical(event.get(field)) for field in EVENT_KEY_FIELDS}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _period_title_fragments(period_end: date) -> tuple[str, ...]:
    year = period_end.year
    month_day = (period_end.month, period_end.day)
    if month_day == (3, 31):
        return (f"{year}年第一季度", f"{year}年一季度")
    if month_day == (6, 30):
        return (f"{year}年半年度", f"{year}年上半年")
    if month_day == (9, 30):
        return (f"{year}年前三季度", f"{year}年第三季度")
    if month_day == (12, 31):
        return (f"{year}年度",)
    return (str(year),)


def _source_report_fragments(period_end: date) -> tuple[str, ...]:
    year = period_end.year
    month_day = (period_end.month, period_end.day)
    if month_day == (3, 31):
        return (f"{year - 1}年年度报告",)
    if month_day == (6, 30):
        return (f"{year}年第一季度报告", f"{year}年一季度报告")
    if month_day == (9, 30):
        return (f"{year}年半年度报告", f"{year}年上半年报告")
    if month_day == (12, 31):
        return (f"{year}年第三季度报告", f"{year}年三季度报告")
    return ()


def select_forecast_document(rows: list[dict[str, Any]], period_end: date) -> dict[str, Any] | None:
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    fragments = _period_title_fragments(period_end)
    source_fragments = _source_report_fragments(period_end)
    for row in rows:
        title = re.sub(r"\s+", "", str(row.get("title") or ""))
        is_forecast = FORECAST_TITLE_PATTERN.search(title) is not None
        is_source_report = any(fragment in title for fragment in source_fragments)
        if not is_forecast and not is_source_report:
            continue
        score = 30 if is_forecast else 10
        if any(fragment in title for fragment in fragments):
            score += 20
        if "首次" in title:
            score += 2
        if REVISED_TITLE_PATTERN.search(title):
            score -= 10
        if is_source_report and "摘要" in title:
            score += 5
        if "审计报告" in title or "社会责任报告" in title:
            score -= 20
        candidates.append((score, str(row.get("announcement_id") or ""), row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]


def extract_reason_section(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    candidates: list[str] = []
    for match in REASON_HEADING_PATTERN.finditer(cleaned):
        tail = cleaned[match.end() : match.end() + 4000]
        boundary = NEXT_HEADING_PATTERN.search(tail)
        section = tail[: boundary.start()] if boundary else tail[:2000]
        section = re.sub(r"--- PAGE \d+ ---", " ", section)
        section = re.sub(r"[ \t]+", " ", section)
        section = re.sub(r"\n{3,}", "\n\n", section).strip()
        if len(re.sub(r"\s+", "", section)) >= 12:
            candidates.append(section)
    if candidates:
        return max(candidates, key=lambda value: len(re.sub(r"\s+", "", value)))[:3000]

    fallback_sentences = []
    for sentence in re.split(r"[。\uff1b;]\s*", cleaned):
        compact = re.sub(r"\s+", " ", sentence).strip()
        if 12 <= len(compact) <= 800 and classify_reason(compact) != "UNCLASSIFIED":
            fallback_sentences.append(compact)
    return "。".join(fallback_sentences[:8])[:3000]


def _initialize(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS event_document_targets (
            event_key VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            ann_date DATE NOT NULL,
            period_end DATE NOT NULL,
            forecast_type VARCHAR,
            query_status VARCHAR NOT NULL,
            error VARCHAR,
            updated_at TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS event_document_announcements (
            event_key VARCHAR NOT NULL,
            announcement_id VARCHAR NOT NULL,
            announce_time TIMESTAMP,
            title VARCHAR NOT NULL,
            pdf_url VARCHAR NOT NULL,
            announced_size_kb DOUBLE,
            selected BOOLEAN NOT NULL,
            discovered_at TIMESTAMP NOT NULL,
            PRIMARY KEY (event_key, announcement_id)
        );
        CREATE TABLE IF NOT EXISTS event_document_metrics (
            announcement_id VARCHAR PRIMARY KEY,
            pdf_sha256 VARCHAR NOT NULL,
            pdf_path VARCHAR NOT NULL,
            text_path VARCHAR NOT NULL,
            pdf_bytes BIGINT NOT NULL,
            pages INTEGER NOT NULL,
            ocr_pages INTEGER NOT NULL,
            parse_status VARCHAR NOT NULL,
            measured_at TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS event_document_evidence (
            event_key VARCHAR PRIMARY KEY,
            announcement_id VARCHAR NOT NULL,
            reason_text VARCHAR NOT NULL,
            reason_class VARCHAR NOT NULL,
            extraction_method VARCHAR NOT NULL,
            evidence_sha256 VARCHAR NOT NULL,
            persisted_at TIMESTAMP NOT NULL
        );
        """
    )


def _register_targets(
    connection: duckdb.DuckDBPyConnection, events: pl.DataFrame
) -> list[dict[str, Any]]:
    targets = events.filter(pl.col("reason_class") == "MISSING").sort(["ann_date", "symbol"])
    rows = targets.to_dicts()
    now = datetime.now()
    for row in rows:
        connection.execute(
            """
            INSERT OR IGNORE INTO event_document_targets VALUES
            (?, ?, ?, ?, ?, 'PENDING', NULL, ?)
            """,
            [
                event_key(row),
                row["symbol"],
                row["ann_date"],
                row["period_end"],
                row.get("type"),
                now,
            ],
        )
    return rows


def _discover(
    connection: duckdb.DuckDBPyConnection,
    client: CninfoClient,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    key = event_key(row)
    target_status = connection.execute(
        "SELECT query_status FROM event_document_targets WHERE event_key=?", [key]
    ).fetchone()
    if target_status and target_status[0] == f"NO_MATCH_{SELECTION_VERSION}":
        return None
    existing = connection.execute(
        """
        SELECT a.announcement_id, a.announce_time, a.title, a.pdf_url,
               a.announced_size_kb
        FROM event_document_announcements a
        WHERE a.event_key=? AND a.selected=true
        """,
        [key],
    ).fetchone()
    if existing:
        return {
            "announcement_id": existing[0],
            "announce_time": existing[1],
            "title": existing[2],
            "pdf_url": existing[3],
            "announced_size_kb": existing[4],
        }

    code = str(row["symbol"]).split(".", maxsplit=1)[0]
    ann_date = row["ann_date"]
    try:
        announcements = client.announcements(code, ann_date, ann_date)
        selected = select_forecast_document(announcements, row["period_end"])
        if selected is None:
            announcements = client.announcements(
                code,
                ann_date - timedelta(days=1),
                ann_date + timedelta(days=1),
            )
            selected = select_forecast_document(announcements, row["period_end"])
        now = datetime.now()
        for announcement in announcements:
            connection.execute(
                """
                INSERT OR REPLACE INTO event_document_announcements VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    key,
                    announcement["announcement_id"],
                    announcement.get("announce_time"),
                    announcement["title"],
                    announcement["pdf_url"],
                    announcement.get("announced_size_kb"),
                    bool(
                        selected and announcement["announcement_id"] == selected["announcement_id"]
                    ),
                    now,
                ],
            )
        status = "DISCOVERED" if selected else f"NO_MATCH_{SELECTION_VERSION}"
        connection.execute(
            "UPDATE event_document_targets SET query_status=?, error=NULL, updated_at=? WHERE event_key=?",
            [status, now, key],
        )
        return selected
    except Exception as exc:
        connection.execute(
            "UPDATE event_document_targets SET query_status='FAILED', error=?, updated_at=? WHERE event_key=?",
            [str(exc)[:500], datetime.now(), key],
        )
        return None


def _measure_document(
    connection: duckdb.DuckDBPyConnection,
    client: CninfoClient,
    root: Path,
    row: dict[str, Any],
    announcement: dict[str, Any],
) -> bool:
    key = event_key(row)
    announcement_id = str(announcement["announcement_id"])
    existing = connection.execute(
        "SELECT 1 FROM event_document_evidence WHERE event_key=?", [key]
    ).fetchone()
    if existing:
        return False
    pdf_path = root / "documents" / f"{announcement_id}.pdf"
    text_path = root / "text" / f"{announcement_id}.txt"
    if not pdf_path.is_file():
        client.download_pdf(
            str(announcement["pdf_url"]),
            pdf_path,
            DEFAULT_MAX_DOCUMENT_BYTES,
        )
    metrics, _facts = analyze_pdf(pdf_path, text_path, announcement_id)
    reason_text = extract_reason_section(text_path.read_text(encoding="utf-8"))
    reason_class = classify_reason(reason_text)
    evidence_hash = hashlib.sha256(reason_text.encode("utf-8")).hexdigest()
    now = datetime.now()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO event_document_metrics VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                announcement_id,
                metrics["sha256"],
                str(pdf_path),
                str(text_path),
                metrics["pdf_bytes"],
                metrics["pages"],
                metrics["ocr_pages"],
                metrics["parse_status"],
                metrics["measured_at"],
            ],
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO event_document_evidence VALUES
            (?, ?, ?, ?, 'DETERMINISTIC_REASON_SECTION_V1', ?, ?)
            """,
            [key, announcement_id, reason_text, reason_class, evidence_hash, now],
        )
        connection.execute(
            "UPDATE event_document_targets SET query_status='MEASURED', error=NULL, updated_at=? WHERE event_key=?",
            [now, key],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return True


def _materialize(connection: duckdb.DuckDBPyConnection, root: Path) -> dict[str, Any]:
    supplement_path = root.parent / "p0_short_horizon_event_documents_v1.parquet"
    rows = connection.execute(
        """
        SELECT t.event_key, t.symbol, t.ann_date, t.period_end,
               t.query_status AS document_status, a.announcement_id,
               a.title AS document_title, m.pdf_sha256, e.reason_text AS document_reason,
               e.reason_class AS document_reason_class, e.evidence_sha256
        FROM event_document_targets t
        LEFT JOIN event_document_announcements a
          ON a.event_key=t.event_key AND a.selected=true
        LEFT JOIN event_document_metrics m
          ON m.announcement_id=a.announcement_id
        LEFT JOIN event_document_evidence e
          ON e.event_key=t.event_key AND e.announcement_id=a.announcement_id
        ORDER BY t.ann_date, t.symbol
        """
    ).pl()
    rows.write_parquet(supplement_path, compression="zstd", statistics=True)
    status_rows = connection.execute(
        "SELECT query_status, count(*) FROM event_document_targets GROUP BY 1 ORDER BY 1"
    ).fetchall()
    class_rows = connection.execute(
        "SELECT reason_class, count(*) FROM event_document_evidence GROUP BY 1 ORDER BY 1"
    ).fetchall()
    totals = connection.execute(
        "SELECT count(*), coalesce(sum(pdf_bytes), 0) FROM event_document_metrics"
    ).fetchone()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "market_outcomes_read": False,
        "deepseek_used": False,
        "targets": rows.height,
        "status": {str(key): int(value) for key, value in status_rows},
        "reason_classification": {str(key): int(value) for key, value in class_rows},
        "documents": int(totals[0]),
        "pdf_bytes": int(totals[1]),
        "supplement_path": str(supplement_path),
        "supplement_sha256": hashlib.sha256(supplement_path.read_bytes()).hexdigest(),
    }
    summary_path = root.parent / "p0_short_horizon_event_documents_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run(
    event_table: Path,
    root: Path,
    *,
    max_new_documents: int,
    min_interval_s: float,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(root / "event_documents.duckdb"))
    _initialize(connection)
    events = pl.read_parquet(event_table)
    targets = _register_targets(connection, events)
    existing_bytes = int(
        connection.execute(
            "SELECT coalesce(sum(pdf_bytes), 0) FROM event_document_metrics"
        ).fetchone()[0]
    )
    downloaded = 0
    client = CninfoClient(min_interval_s=min_interval_s)
    try:
        for index, row in enumerate(targets, start=1):
            announcement = _discover(connection, client, row)
            if announcement is None or downloaded >= max_new_documents:
                continue
            announced_bytes = int(float(announcement.get("announced_size_kb") or 0) * 1024)
            if existing_bytes + announced_bytes > MAX_TOTAL_PDF_BYTES:
                break
            try:
                if _measure_document(connection, client, root, row, announcement):
                    downloaded += 1
                    existing_bytes = int(
                        connection.execute(
                            "SELECT coalesce(sum(pdf_bytes), 0) FROM event_document_metrics"
                        ).fetchone()[0]
                    )
            except Exception as exc:
                connection.execute(
                    "UPDATE event_document_targets SET query_status='FAILED', error=?, updated_at=? WHERE event_key=?",
                    [str(exc)[:500], datetime.now(), event_key(row)],
                )
            if index % 20 == 0:
                print(
                    json.dumps(
                        {
                            "processed_targets": index,
                            "total_targets": len(targets),
                            "new_documents": downloaded,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        client.close()
    summary = _materialize(connection, root)
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-new-documents", type=int, default=200)
    parser.add_argument("--min-interval-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.max_new_documents < 0:
        raise ValueError("max-new-documents must be non-negative")
    run(
        args.event_table,
        args.root,
        max_new_documents=args.max_new_documents,
        min_interval_s=args.min_interval_s,
    )


if __name__ == "__main__":
    main()
