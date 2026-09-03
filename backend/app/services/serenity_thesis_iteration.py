# ruff: noqa: RUF001
"""Evidence-repair iteration for the one-year Serenity chokepoint study.

This module is deliberately separate from the frozen first two semantic rounds.
It repairs one red-team finding: the explored price rule did not require the
five PDF-backed 64-point dimensions.  Selection of supplemental documents never
uses returns, and every paid response is cached in ``semantic_model_calls``
before the derived thesis score is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.services.serenity_event_replay import EventReplayStore
from app.services.serenity_pdf_scoring import (
    DIMENSION_WEIGHTS,
    DocumentPacket,
    ModelCallSpec,
    ScoreState,
    _adapter_runner,
    _file_hash,
    _freeze_json,
    _parse_pages,
    execute_cached_call,
    sanitize_score_evidence,
    validate_score_output,
)
from app.services.serenity_pilot import (
    DEFAULT_MAX_DOCUMENT_BYTES,
    CninfoClient,
    _atomic_json,
    _pilot_lock,
    _stable_hash,
    analyze_pdf,
)
from app.services.serenity_strategy_optimizer import (
    EVENT_ENRICHED_SCORE_STAGE,
    FOLDS,
    MAX_REQUEST_BYTES,
    OPTIMIZATION_ID,
    STAGE_COST_CAP_MICROS_CNY,
    EventScoreState,
    _dimension_score,
    _event_review_is_bound,
    _mask_identity,
    _preflight_stage,
    _select_support_pages,
    _trial_summary,
    build_event_score_prompt,
    initialize_optimization_tables,
    write_optimization_contract,
)

THESIS_SUBSTAGE = "THESIS_EVIDENCE_REPAIR"
THESIS_POLICY_ID = "P7_PDF64_HARD_GATE"
THESIS_OVERLAY_POLICY_ID = "P8_PDF64_DUAL_ENTRY"
ANNUAL_LOOKBACK_DAYS = 550
MAX_THESIS_DOCUMENTS = 16
MAX_THESIS_RAW_BYTES = 200_000_000
MAX_THESIS_PAGES_PER_DOCUMENT = 12
MAX_THESIS_CONTEXTS = 14
THESIS_WORST_CASE_CAP_MICROS_CNY = 13_000_000
MIN_COMPLETE_SCORE = 38.4
MIN_DIMENSION_RATING = 3

_FULL_ANNUAL_RE = re.compile(r"年度报告(?:（更正后）)?$")
_ANNUAL_EXCLUDE_RE = re.compile(r"半年度报告|摘要|披露提示|英文版")
_SUPPLIER_SUBJECT_RE = re.compile(r"供应商|厂商|供给|供应来源|供应格局|市场份额|市占率")
_SUPPLIER_CONSTRAINT_RE = re.compile(
    r"独家|独供|唯一|仅有|只有|少数|高度集中|垄断|寡头|"
    r"主导|不超过[\u4e00\u4e8c\u4e09123]家|[\u4e00\u4e8c\u4e09123]家(?:合格)?(?:供应商|厂商)|"
    r"份额.{0,12}(?:[5-9]\d|100)%|市占率.{0,12}(?:[5-9]\d|100)%|"
    r"替代.{0,12}(?:认证|受限|困难|周期)"
)
_IRRELEVANT_INPUT_SUPPLIER_RE = re.compile(
    r"前五(?:名|大)供应商.{0,20}(?:采购|占比)|"
    r"采购额.{0,20}年度采购总额|第一大供应商"
)


def is_full_annual_report(title: str) -> bool:
    compact = re.sub(r"\s+", "", str(title or ""))
    return bool(_FULL_ANNUAL_RE.search(compact)) and not bool(_ANNUAL_EXCLUDE_RE.search(compact))


def supplier_concentration_evidence_is_bound(
    dimensions: list[dict[str, Any]],
) -> bool:
    """Reject citations that exist verbatim but do not prove supply concentration.

    The model previously treated an issuer's upstream self-sufficiency statement as
    evidence that the issuer was one of only a few suppliers of the bottleneck
    product.  Exact quote matching cannot catch that construct error.  Ratings of
    three or above therefore need a quote that names the supply market and an
    explicit scarcity, concentration, limited-source, or substitution constraint.
    Ordinary top-five *input procurement* concentration is a different exposure and
    is deliberately rejected here.
    """

    item = next(
        (value for value in dimensions if value.get("dimension_id") == "supplier_concentration"),
        None,
    )
    if not item or item.get("status") == "UNKNOWN" or item.get("rating") is None:
        return False
    if int(item["rating"]) < MIN_DIMENSION_RATING:
        return False
    for citation in item.get("evidence") or []:
        quote = re.sub(r"\s+", "", str(citation.get("quote") or ""))
        if not quote or _IRRELEVANT_INPUT_SUPPLIER_RE.search(quote):
            continue
        if _SUPPLIER_SUBJECT_RE.search(quote) and _SUPPLIER_CONSTRAINT_RE.search(quote):
            return True
    return False


def thesis_dimension_gate(dimensions: list[dict[str, Any]], complete_score: float | None) -> bool:
    if complete_score is None or float(complete_score) < MIN_COMPLETE_SCORE:
        return False
    ratings = {
        str(item.get("dimension_id")): item.get("rating")
        for item in dimensions
        if item.get("status") != "UNKNOWN"
    }
    required = {
        "architecture_coupling",
        "chokepoint_severity",
        "supplier_concentration",
        "expansion_difficulty",
        "evidence_quality",
    }
    ratings_pass = required.issubset(ratings) and all(
        ratings[key] is not None and int(ratings[key]) >= MIN_DIMENSION_RATING for key in required
    )
    return ratings_pass and supplier_concentration_evidence_is_bound(dimensions)


def conservative_dimension_consensus(
    *dimension_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge independently validated calls without choosing the higher rating."""
    result: list[dict[str, Any]] = []
    for dimension_id in DIMENSION_WEIGHTS:
        candidates = [
            item
            for dimensions in dimension_sets
            for item in dimensions
            if item.get("dimension_id") == dimension_id
            and item.get("status") != "UNKNOWN"
            and item.get("rating") is not None
        ]
        if not candidates:
            result.append(
                {
                    "dimension_id": dimension_id,
                    "status": "UNKNOWN",
                    "rating": None,
                    "reason": "all independently validated calls remain unknown",
                    "evidence": [],
                }
            )
            continue
        minimum = min(int(item["rating"]) for item in candidates)
        retained = [item for item in candidates if int(item["rating"]) == minimum]
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in retained:
            for citation in item.get("evidence") or []:
                key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    evidence.append(citation)
        result.append(
            {
                "dimension_id": dimension_id,
                "status": "EVIDENCED",
                "rating": minimum,
                "reason": (
                    "conservative cross-call consensus; minimum evidenced rating retained; "
                    "UNKNOWN is absence, not a contradictory score"
                ),
                "evidence": evidence,
            }
        )
    return result


def initialize_thesis_tables(store: EventReplayStore) -> None:
    initialize_optimization_tables(store)
    store.connection.execute(
        """
        CREATE TABLE IF NOT EXISTS serenity_thesis_support_documents (
            optimization_id VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            announcement_id VARCHAR NOT NULL,
            announce_time TIMESTAMP NOT NULL,
            title VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            error VARCHAR,
            planned_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, event_id, announcement_id)
        );
        CREATE TABLE IF NOT EXISTS serenity_thesis_scores (
            optimization_id VARCHAR NOT NULL,
            substage VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            entity_id VARCHAR NOT NULL,
            event_gate VARCHAR NOT NULL,
            event_stage VARCHAR NOT NULL,
            newness VARCHAR NOT NULL,
            economic_bridge VARCHAR NOT NULL,
            known_dimension_weight DOUBLE NOT NULL,
            known_dimension_points DOUBLE NOT NULL,
            complete_dimension_score DOUBLE,
            dimension_json VARCHAR NOT NULL,
            event_review_json VARCHAR NOT NULL,
            raw_output_json VARCHAR NOT NULL,
            model_input_sha256 VARCHAR NOT NULL,
            scored_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, substage, event_id)
        );
        CREATE TABLE IF NOT EXISTS serenity_thesis_trials (
            optimization_id VARCHAR NOT NULL,
            round_id VARCHAR NOT NULL,
            policy_id VARCHAR NOT NULL,
            fold_id VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            observation_count INTEGER NOT NULL,
            mean_net_return DOUBLE,
            median_net_return DOUBLE,
            win_rate DOUBLE,
            alpha_csi300 DOUBLE,
            alpha_chain DOUBLE,
            mean_mae DOUBLE,
            mean_mfe DOUBLE,
            status VARCHAR NOT NULL,
            evaluated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, round_id, policy_id, fold_id, horizon)
        );
        CREATE TABLE IF NOT EXISTS serenity_thesis_consensus_scores (
            optimization_id VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            event_gate VARCHAR NOT NULL,
            complete_dimension_score DOUBLE,
            dimension_json VARCHAR NOT NULL,
            hard_gate_pass BOOLEAN NOT NULL,
            derivation_rule VARCHAR NOT NULL,
            derived_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, event_id)
        );
        """
    )


def _optimization_root(store: EventReplayStore) -> Path:
    return store.root / "optimization" / OPTIMIZATION_ID


def _qualified_events(store: EventReplayStore) -> list[tuple[Any, ...]]:
    return store.connection.execute(
        """
        SELECT s.event_id, s.symbol, u.code, s.decision_date, e.announcement_id
        FROM serenity_event_semantic_scores s
        JOIN universe u USING (symbol)
        JOIN event_candidates e USING (event_id)
        WHERE s.optimization_id=? AND s.stage=?
          AND s.event_gate='PASS' AND s.newness='NEW_INFORMATION'
          AND s.economic_bridge='PASS' AND s.event_stage!='ROUTINE_ADMIN'
        ORDER BY s.decision_date, s.symbol, s.event_id
        """,
        [OPTIMIZATION_ID, EVENT_ENRICHED_SCORE_STAGE],
    ).fetchall()


def discover_and_plan(store: EventReplayStore) -> dict[str, Any]:
    """Discover official annual reports and freeze a return-blind manifest."""
    initialize_thesis_tables(store)
    events = _qualified_events(store)
    if not events:
        raise RuntimeError("no enriched PASS events are available")
    ranges: dict[str, tuple[str, date, date]] = {}
    for _event_id, symbol, code, decision_day, _announcement_id in events:
        current = ranges.get(str(symbol))
        start = decision_day - timedelta(days=ANNUAL_LOOKBACK_DAYS)
        if current is None:
            ranges[str(symbol)] = (str(code), start, decision_day)
        else:
            ranges[str(symbol)] = (
                str(code),
                min(current[1], start),
                max(current[2], decision_day),
            )

    client = CninfoClient()
    discovery_failures: list[dict[str, str]] = []
    try:
        for symbol, (code, start, end) in sorted(ranges.items()):
            try:
                for item in client.announcements(code, start, end):
                    store.connection.execute(
                        """
                        INSERT OR IGNORE INTO announcements VALUES
                        (?, ?, ?, ?, ?, ?, 'discovered', NULL, ?)
                        """,
                        [
                            item["announcement_id"],
                            symbol,
                            item["announce_time"],
                            item["title"],
                            item["pdf_url"],
                            item["announced_size_kb"],
                            datetime.now(),
                        ],
                    )
            except Exception as exc:
                discovery_failures.append({"symbol": symbol, "error": str(exc)[:240]})
    finally:
        client.close()

    selected: list[dict[str, Any]] = []
    for event_id, symbol, _code, decision_day, _event_announcement_id in events:
        candidates = store.connection.execute(
            """
            SELECT announcement_id, announce_time, title, announced_size_kb
            FROM announcements
            WHERE symbol=? AND announce_time IS NOT NULL
              AND CAST(announce_time AS DATE)<=?
              AND CAST(announce_time AS DATE)>=? - INTERVAL 550 DAY
            ORDER BY announce_time DESC, announcement_id DESC
            """,
            [symbol, decision_day, decision_day],
        ).fetchall()
        annual = next((row for row in candidates if is_full_annual_report(row[2])), None)
        if annual is None:
            continue
        selected.append(
            {
                "event_id": str(event_id),
                "symbol": str(symbol),
                "decision_date": decision_day.isoformat(),
                "announcement_id": str(annual[0]),
                "announce_time": annual[1].isoformat(),
                "title": str(annual[2]),
                "announced_size_kb": (float(annual[3]) if annual[3] is not None else None),
            }
        )

    unique = {row["announcement_id"]: row for row in selected}
    if len(unique) > MAX_THESIS_DOCUMENTS:
        raise RuntimeError("thesis annual-report plan exceeds document cap")
    estimated_bytes = round(
        sum(max(0.0, float(row["announced_size_kb"] or 0.0)) * 1024 for row in unique.values())
    )
    if estimated_bytes > MAX_THESIS_RAW_BYTES:
        raise RuntimeError("thesis annual-report plan exceeds raw-byte cap")
    manifest = {
        "optimization_id": OPTIMIZATION_ID,
        "substage": THESIS_SUBSTAGE,
        "selection_uses_outcomes": False,
        "qualification_rule": (
            "round2 event gate PASS + NEW_INFORMATION + economic bridge PASS + non-admin"
        ),
        "document_rule": (
            "latest complete official annual report available at T0; exclude half-year, summary, "
            "disclosure notice and English edition"
        ),
        "lookback_days": ANNUAL_LOOKBACK_DAYS,
        "qualified_event_count": len(events),
        "selected_event_count": len(selected),
        "unique_document_count": len(unique),
        "max_documents": MAX_THESIS_DOCUMENTS,
        "max_raw_bytes": MAX_THESIS_RAW_BYTES,
        "estimated_raw_bytes": estimated_bytes,
        "records": selected,
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = _optimization_root(store) / "thesis-support-document-manifest.json"
    _freeze_json(path, manifest, "thesis support document manifest")
    store.connection.execute(
        "DELETE FROM serenity_thesis_support_documents WHERE optimization_id=?",
        [OPTIMIZATION_ID],
    )
    if selected:
        store.connection.executemany(
            """
            INSERT INTO serenity_thesis_support_documents VALUES
            (?, ?, ?, ?, ?, ?, ?, 'PLANNED', NULL, ?)
            """,
            [
                [
                    OPTIMIZATION_ID,
                    row["event_id"],
                    row["symbol"],
                    date.fromisoformat(row["decision_date"]),
                    row["announcement_id"],
                    datetime.fromisoformat(row["announce_time"]),
                    row["title"],
                    datetime.now(),
                ]
                for row in selected
            ],
        )
    return {
        "qualified_events": len(events),
        "selected_events": len(selected),
        "unique_documents": len(unique),
        "estimated_raw_bytes": estimated_bytes,
        "discovery_failures": discovery_failures,
        "manifest_path": str(path),
    }


def collect_documents(store: EventReplayStore) -> dict[str, Any]:
    """Download and commit each frozen annual report independently."""
    initialize_thesis_tables(store)
    path = _optimization_root(store) / "thesis-support-document-manifest.json"
    if not path.is_file():
        raise RuntimeError("thesis support manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    unique = {str(row["announcement_id"]): row for row in manifest.get("records") or []}
    downloaded = downloaded_bytes = reused = failed = 0
    client = CninfoClient()
    try:
        for announcement_id in sorted(unique):
            existing = store.connection.execute(
                "SELECT pdf_bytes FROM document_metrics WHERE announcement_id=?",
                [announcement_id],
            ).fetchone()
            if existing:
                reused += 1
                store.connection.execute(
                    """
                    UPDATE serenity_thesis_support_documents SET status='READY', error=NULL
                    WHERE optimization_id=? AND announcement_id=?
                    """,
                    [OPTIMIZATION_ID, announcement_id],
                )
                continue
            announcement = store.connection.execute(
                "SELECT pdf_url FROM announcements WHERE announcement_id=?",
                [announcement_id],
            ).fetchone()
            if not announcement:
                raise RuntimeError(f"thesis announcement disappeared: {announcement_id}")
            pdf_path = store.documents_dir / f"{announcement_id}.pdf"
            text_path = store.text_dir / f"{announcement_id}.txt"
            try:
                size = client.download_pdf(
                    str(announcement[0]), pdf_path, DEFAULT_MAX_DOCUMENT_BYTES
                )
                if downloaded_bytes + size > MAX_THESIS_RAW_BYTES:
                    pdf_path.unlink(missing_ok=True)
                    raise RuntimeError("thesis download would exceed frozen raw-byte cap")
                metrics, facts = analyze_pdf(pdf_path, text_path, announcement_id)
                store.connection.execute("BEGIN TRANSACTION")
                try:
                    store.connection.execute(
                        "INSERT INTO document_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            metrics["announcement_id"],
                            metrics["sha256"],
                            metrics["pages"],
                            metrics["pdf_bytes"],
                            metrics["embedded_text_bytes"],
                            metrics["extracted_text_bytes"],
                            metrics["ocr_text_bytes"],
                            metrics["ocr_pages"],
                            metrics["low_text_pages"],
                            metrics["rendered_png_bytes"],
                            metrics["persistent_inflation_pct"],
                            metrics["ocr_render_multiplier"],
                            metrics["fact_count"],
                            metrics["parse_status"],
                            metrics["measured_at"],
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
                        [announcement_id],
                    )
                    store.connection.execute(
                        """
                        UPDATE serenity_thesis_support_documents SET status='READY', error=NULL
                        WHERE optimization_id=? AND announcement_id=?
                        """,
                        [OPTIMIZATION_ID, announcement_id],
                    )
                    store.connection.execute("COMMIT")
                except Exception:
                    store.connection.execute("ROLLBACK")
                    raise
                downloaded += 1
                downloaded_bytes += size
            except Exception as exc:
                failed += 1
                pdf_path.unlink(missing_ok=True)
                text_path.unlink(missing_ok=True)
                store.connection.execute(
                    """
                    UPDATE serenity_thesis_support_documents SET status='FAILED', error=?
                    WHERE optimization_id=? AND announcement_id=?
                    """,
                    [str(exc)[:300], OPTIMIZATION_ID, announcement_id],
                )
        return {
            "planned_unique_documents": len(unique),
            "downloaded_documents": downloaded,
            "downloaded_bytes": downloaded_bytes,
            "reused_documents": reused,
            "failed_documents": failed,
            "ready_links": store.connection.execute(
                """
                SELECT count(*) FROM serenity_thesis_support_documents
                WHERE optimization_id=? AND status='READY'
                """,
                [OPTIMIZATION_ID],
            ).fetchone()[0],
        }
    finally:
        client.close()


def _select_thesis_pages(pages: dict[int, str]) -> dict[int, str]:
    selected = _select_support_pages(pages)
    if len(selected) >= MAX_THESIS_PAGES_PER_DOCUMENT:
        return selected
    keywords = (
        "核心技术",
        "技术壁垒",
        "竞争优势",
        "供应商",
        "采购",
        "进口",
        "国产替代",
        "客户认证",
        "研发投入",
        "在建工程",
        "产能",
        "建设周期",
    )
    ranked = sorted(
        (
            (sum(text.count(word) for word in keywords), page_number, text)
            for page_number, text in pages.items()
            if page_number not in selected
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for score, page_number, text in ranked:
        if score <= 0 or len(selected) >= MAX_THESIS_PAGES_PER_DOCUMENT:
            break
        selected[page_number] = text
    return dict(sorted(selected.items()))


def load_states(store: EventReplayStore) -> tuple[list[EventScoreState], list[dict[str, Any]]]:
    initialize_thesis_tables(store)
    universe = {row["symbol"]: row for row in store.universe()}
    states: list[EventScoreState] = []
    blocked: list[dict[str, Any]] = []
    for event_id, symbol, _code, decision_day, _event_announcement_id in _qualified_events(store):
        company = universe[str(symbol)]
        event_type = store.connection.execute(
            "SELECT primary_event_type FROM event_candidates WHERE event_id=?", [event_id]
        ).fetchone()[0]
        documents = store.connection.execute(
            """
            SELECT e.announcement_id, e.published_at, e.title, d.sha256,
                   'EVENT' AS kind, 1 AS kind_priority
            FROM event_candidates e
            JOIN document_metrics d ON d.announcement_id=e.announcement_id
            WHERE e.symbol=? AND e.decision_date=?
              AND e.polarity IN ('LONG_CANDIDATE', 'MIXED_REVIEW')
            UNION ALL
            SELECT m.announcement_id, a.announce_time, a.title, d.sha256,
                   m.selection_kind, 2 AS kind_priority
            FROM serenity_event_support_documents m
            JOIN announcements a USING (announcement_id)
            JOIN document_metrics d USING (announcement_id)
            WHERE m.optimization_id=? AND m.event_id=? AND m.status='READY'
              AND CAST(a.announce_time AS DATE)<=m.decision_date
            UNION ALL
            SELECT m.announcement_id, m.announce_time, m.title, d.sha256,
                   'THESIS_ANNUAL_REPORT' AS kind, 0 AS kind_priority
            FROM serenity_thesis_support_documents m
            JOIN document_metrics d USING (announcement_id)
            WHERE m.optimization_id=? AND m.event_id=? AND m.status='READY'
              AND CAST(m.announce_time AS DATE)<=m.decision_date
            ORDER BY 2, 1, 6
            """,
            [
                symbol,
                decision_day,
                OPTIMIZATION_ID,
                event_id,
                OPTIMIZATION_ID,
                event_id,
            ],
        ).fetchall()
        packets: list[DocumentPacket] = []
        document_map: dict[str, str] = {}
        seen: set[str] = set()
        thesis_ready = False
        for index, (document_id, published_at, title, sha256, kind, _priority) in enumerate(
            documents, start=1
        ):
            document_id = str(document_id)
            if document_id in seen or published_at is None:
                continue
            seen.add(document_id)
            text_path = store.text_dir / f"{document_id}.txt"
            if not text_path.is_file():
                continue
            pages = _parse_pages(text_path)
            if str(kind) != "EVENT":
                pages = (
                    _select_thesis_pages(pages)
                    if str(kind) == "THESIS_ANNUAL_REPORT"
                    else _select_support_pages(pages)
                )
            pages = {
                page: _mask_identity(
                    text,
                    symbol=str(symbol),
                    code=str(company["code"]),
                    name=str(company["name"]),
                )
                for page, text in pages.items()
            }
            if not pages:
                continue
            opaque_id = (
                "DOC-"
                + _stable_hash(OPTIMIZATION_ID, THESIS_SUBSTAGE, str(event_id), str(index))[:16]
            )
            packets.append(
                DocumentPacket(
                    document_id=opaque_id,
                    announcement_time=published_at,
                    title=_mask_identity(
                        f"[{kind}] {title}",
                        symbol=str(symbol),
                        code=str(company["code"]),
                        name=str(company["name"]),
                    ),
                    pages=pages,
                    sha256=str(sha256),
                )
            )
            document_map[opaque_id] = document_id
            thesis_ready = thesis_ready or str(kind) == "THESIS_ANNUAL_REPORT"
        entity_id = "EVENT-" + _stable_hash(OPTIMIZATION_ID, str(event_id))[:20]
        score_state = ScoreState(
            symbol=str(symbol),
            name="[发行人]",
            chain_id=str(company["chain_id"]),
            cutoff_date=decision_day,
            entity_id=entity_id,
            documents=tuple(packets),
        )
        prompt = build_event_score_prompt(score_state, str(event_type))
        if not thesis_ready:
            blocked.append({"event_id": str(event_id), "status": "NO_READY_FULL_ANNUAL_REPORT"})
        elif len(prompt.encode("utf-8")) > MAX_REQUEST_BYTES - 30_000:
            blocked.append(
                {
                    "event_id": str(event_id),
                    "status": "OVERSIZE_THESIS_CONTEXT",
                    "prompt_bytes": len(prompt.encode("utf-8")),
                }
            )
        else:
            states.append(
                EventScoreState(
                    event_id=str(event_id),
                    symbol=str(symbol),
                    decision_date=decision_day,
                    event_type=str(event_type),
                    entity_id=entity_id,
                    score_state=score_state,
                    document_map=document_map,
                )
            )
    ranked = sorted(
        states,
        key=lambda state: (
            -len(build_event_score_prompt(state.score_state, state.event_type).encode("utf-8")),
            state.decision_date,
            state.event_id,
        ),
    )
    selected = ranked[:MAX_THESIS_CONTEXTS]
    for state in ranked[MAX_THESIS_CONTEXTS:]:
        blocked.append(
            {
                "event_id": state.event_id,
                "status": "DEFERRED_THESIS_CNY13_CAP",
                "selection_rule": (
                    "14 largest return-blind T0 PDF contexts; ties by decision date and event id"
                ),
            }
        )
    selected.sort(key=lambda state: (state.decision_date, state.event_id))
    return selected, blocked


def run_scores(store: EventReplayStore, *, execute: bool) -> dict[str, Any]:
    initialize_thesis_tables(store)
    paths = write_optimization_contract(store)
    states, blocked = load_states(store)
    completed = {
        row[0]
        for row in store.connection.execute(
            """
            SELECT event_id FROM serenity_thesis_scores
            WHERE optimization_id=? AND substage=?
            """,
            [OPTIMIZATION_ID, THESIS_SUBSTAGE],
        ).fetchall()
    }
    pending = [state for state in states if state.event_id not in completed]
    policy_hash = _file_hash(paths["policy"])
    specs = [
        ModelCallSpec(
            OPTIMIZATION_ID,
            EVENT_ENRICHED_SCORE_STAGE,
            "SLOT-THESIS-" + _stable_hash(OPTIMIZATION_ID, state.event_id)[:28],
            state.decision_date,
            build_event_score_prompt(state.score_state, state.event_type),
            paths["score_schema"],
            policy_hash,
        )
        for state in pending
    ]
    preflight = (
        _preflight_stage(
            paths,
            stage=EVENT_ENRICHED_SCORE_STAGE,
            trade_date=specs[0].trade_date,
            prompts=[spec.prompt for spec in specs],
        )
        if specs
        else {"status": "NO_PENDING_CALLS"}
    )
    worst_case = int(
        (preflight.get("preflight") or {})
        .get("worst_case_increment", {})
        .get("charged_cost_micros_cny", 0)
    )
    if worst_case > THESIS_WORST_CASE_CAP_MICROS_CNY:
        raise RuntimeError("thesis evidence repair exceeds its CNY13 worst-case cap")
    current_stage_cost = int(
        store.connection.execute(
            """
            SELECT coalesce(sum(cost_micros_cny), 0)
            FROM semantic_model_calls WHERE replay_id=? AND stage=?
            """,
            [OPTIMIZATION_ID, EVENT_ENRICHED_SCORE_STAGE],
        ).fetchone()[0]
    )
    if current_stage_cost + worst_case > STAGE_COST_CAP_MICROS_CNY[EVENT_ENRICHED_SCORE_STAGE]:
        raise RuntimeError("enriched stage plus thesis repair exceeds frozen CNY25 cap")
    amendment = {
        "optimization_id": OPTIMIZATION_ID,
        "substage": THESIS_SUBSTAGE,
        "parent_stage": EVENT_ENRICHED_SCORE_STAGE,
        "selection_uses_outcomes": False,
        "selection_rule": (
            "14 largest return-blind T0 PDF contexts; ties by decision date and event id"
        ),
        "reason": (
            "red team rejected P4 because the PDF-backed 64-point thesis was not a hard gate; "
            "repair evidence coverage before any further price-rule selection"
        ),
        "threshold_policy": {
            "complete_dimension_score_min": MIN_COMPLETE_SCORE,
            "each_dimension_rating_min": MIN_DIMENSION_RATING,
            "horizons": [2, 3, 5, 10],
            "primary_entry": "next_session_open_without_momentum_or_gap_filter",
        },
        "base_enriched_population_sha256": _file_hash(
            paths["run_root"] / f"{EVENT_ENRICHED_SCORE_STAGE.lower()}-paid-population.json"
        ),
        "slots": [
            {
                "event_id": state.event_id,
                "slot_id": spec.slot_id,
                "input_sha256": spec.input_sha256,
            }
            for state, spec in zip(pending, specs, strict=True)
        ],
    }
    amendment_path = paths["run_root"] / "thesis-evidence-paid-population-amendment.json"
    if amendment_path.is_file():
        existing = json.loads(amendment_path.read_text(encoding="utf-8"))
        if existing != amendment:
            paid_calls = store.connection.execute(
                """
                SELECT count(*) FROM semantic_model_calls
                WHERE replay_id=? AND stage=? AND slot_id LIKE 'SLOT-THESIS-%'
                """,
                [OPTIMIZATION_ID, EVENT_ENRICHED_SCORE_STAGE],
            ).fetchone()[0]
            if paid_calls:
                raise RuntimeError("thesis paid population cannot change after a model call")
            correction = {
                **amendment,
                "supersedes_sha256": _file_hash(amendment_path),
                "correction_reason": (
                    "preflight exposed unstable duplicate-document ordering; no paid call occurred"
                ),
            }
            correction_path = (
                paths["run_root"] / "thesis-evidence-paid-population-amendment-correction-01.json"
            )
            _freeze_json(
                correction_path,
                correction,
                "corrected thesis evidence paid population amendment",
            )
            amendment_path = correction_path
    else:
        _freeze_json(amendment_path, amendment, "thesis evidence paid population amendment")
    plan = {
        "substage": THESIS_SUBSTAGE,
        "states": len(states),
        "completed": len(completed),
        "pending": len(specs),
        "blocked": blocked,
        "preflight": preflight,
        "current_parent_stage_cost_micros_cny": current_stage_cost,
        "execute": execute,
        "amendment_path": str(amendment_path),
    }
    _atomic_json(paths["run_root"] / "thesis-evidence-plan.json", plan)
    if not execute or not specs:
        return plan
    runner = _adapter_runner(paths)
    for index, (state, spec) in enumerate(zip(pending, specs, strict=True), start=1):
        result = execute_cached_call(store.connection, spec, runner)
        raw_output = json.loads(result.raw_output)
        documents = {
            document.document_id: document.pages for document in state.score_state.documents
        }
        sanitized, _adjustments = sanitize_score_evidence(raw_output, documents)
        sanitized, event_citation_failures = _event_review_is_bound(sanitized, documents)
        errors = validate_score_output(sanitized, entity_id=state.entity_id, documents=documents)
        if errors:
            raise RuntimeError(
                f"thesis score validation failed for {state.event_id}: {'; '.join(errors[:5])}"
            )
        review = sanitized["event_review"]
        score = _dimension_score(sanitized)
        store.connection.execute(
            """
            INSERT OR REPLACE INTO serenity_thesis_scores VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                OPTIMIZATION_ID,
                THESIS_SUBSTAGE,
                state.event_id,
                state.symbol,
                state.decision_date,
                state.entity_id,
                review["event_gate"],
                review["event_stage"],
                review["newness"],
                review["economic_bridge"],
                score["known_weight"],
                score["known_points"],
                score["complete"],
                json.dumps(sanitized["dimensions"], ensure_ascii=False, sort_keys=True),
                json.dumps(
                    {
                        **review,
                        "event_citation_failures": event_citation_failures,
                        "document_map": state.document_map,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                result.raw_output,
                spec.input_sha256,
                datetime.now(),
            ],
        )
        print(
            json.dumps(
                {
                    "event": "serenity_thesis_score_progress",
                    "completed": len(completed) + index,
                    "total": len(states),
                    "event_gate": review["event_gate"],
                    "known_dimension_weight": score["known_weight"],
                    "complete_dimension_score": score["complete"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {**plan, "completed_after_run": len(states)}


def _non_overlapping(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_exit: date | None = None
    for row in sorted(rows, key=lambda item: (item["entry_date"], item["event_id"])):
        if last_exit is not None and row["entry_date"] <= last_exit:
            continue
        selected.append(row)
        last_exit = row["exit_date"]
    return selected


def evaluate(store: EventReplayStore) -> dict[str, Any]:
    """Evaluate a theory-first gate; this remains post-hoc descriptive research."""
    initialize_thesis_tables(store)
    rows = store.connection.execute(
        """
        SELECT o.event_id, o.decision_date, o.entry_date, o.exit_date, o.horizon,
               o.net_return, o.benchmark_return, o.chain_return, o.mae, o.mfe,
               f.relative_momentum_20, f.next_open_gap,
               s.event_gate, s.newness, s.economic_bridge, s.event_stage,
               s.complete_dimension_score, s.dimension_json,
               b.event_gate, b.newness, b.economic_bridge, b.event_stage,
               b.complete_dimension_score, b.dimension_json, s.symbol
        FROM event_discovery_outcomes o
        JOIN serenity_event_features f
          ON f.optimization_id=? AND f.event_id=o.event_id
        JOIN serenity_thesis_scores s
          ON s.optimization_id=? AND s.substage=? AND s.event_id=o.event_id
        JOIN serenity_event_semantic_scores b
          ON b.optimization_id=? AND b.stage=? AND b.event_id=o.event_id
        WHERE o.status='SETTLED'
        ORDER BY o.decision_date, o.event_id, o.horizon
        """,
        [
            OPTIMIZATION_ID,
            OPTIMIZATION_ID,
            THESIS_SUBSTAGE,
            OPTIMIZATION_ID,
            EVENT_ENRICHED_SCORE_STAGE,
        ],
    ).fetchall()
    records = [
        {
            "event_id": str(row[0]),
            "decision_date": row[1],
            "entry_date": row[2],
            "exit_date": row[3],
            "horizon": int(row[4]),
            "net_return": float(row[5]),
            "benchmark_return": float(row[6]),
            "chain_return": float(row[7]),
            "mae": float(row[8]),
            "mfe": float(row[9]),
            "relative_momentum_20": row[10],
            "next_open_gap": row[11],
            "event_gate": row[12],
            "newness": row[13],
            "economic_bridge": row[14],
            "event_stage": row[15],
            "complete_score": row[16],
            "dimensions": conservative_dimension_consensus(
                json.loads(row[17]), json.loads(row[23])
            ),
            "base_event_gate": row[18],
            "base_newness": row[19],
            "base_economic_bridge": row[20],
            "base_event_stage": row[21],
            "base_complete_score": row[22],
            "symbol": row[24],
        }
        for row in rows
    ]
    store.connection.execute(
        "DELETE FROM serenity_thesis_consensus_scores WHERE optimization_id=?",
        [OPTIMIZATION_ID],
    )
    for row in records:
        consensus_score = _dimension_score({"dimensions": row["dimensions"]})["complete"]
        row["complete_score"] = consensus_score
        gates_agree = (
            row["event_gate"] == "PASS"
            and row["newness"] == "NEW_INFORMATION"
            and row["economic_bridge"] == "PASS"
            and row["event_stage"] != "ROUTINE_ADMIN"
            and row["base_event_gate"] == "PASS"
            and row["base_newness"] == "NEW_INFORMATION"
            and row["base_economic_bridge"] == "PASS"
            and row["base_event_stage"] != "ROUTINE_ADMIN"
        )
        hard_gate = gates_agree and thesis_dimension_gate(row["dimensions"], consensus_score)
        row["consensus_event_gate"] = "PASS" if gates_agree else "FAIL_CROSS_CALL"
        store.connection.execute(
            """
            INSERT OR REPLACE INTO serenity_thesis_consensus_scores VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                OPTIMIZATION_ID,
                row["event_id"],
                row["symbol"],
                row["decision_date"],
                row["consensus_event_gate"],
                consensus_score,
                json.dumps(row["dimensions"], ensure_ascii=False, sort_keys=True),
                hard_gate,
                "minimum evidenced rating across Round2 and Round4; UNKNOWN is not contradiction",
                datetime.now(),
            ],
        )

    def pure(row: dict[str, Any]) -> bool:
        return row["consensus_event_gate"] == "PASS" and thesis_dimension_gate(
            row["dimensions"], row["complete_score"]
        )

    def overlay(row: dict[str, Any]) -> bool:
        return (
            pure(row)
            and row["relative_momentum_20"] is not None
            and row["relative_momentum_20"] >= 0
            and row["next_open_gap"] is not None
            and (row["next_open_gap"] <= 0 or row["next_open_gap"] > 0.02)
        )

    round_id = "ROUND04_THESIS_EVIDENCE"
    policies = {THESIS_POLICY_ID: pure, THESIS_OVERLAY_POLICY_ID: overlay}
    results: list[dict[str, Any]] = []
    store.connection.execute(
        "DELETE FROM serenity_thesis_trials WHERE optimization_id=? AND round_id=?",
        [OPTIMIZATION_ID, round_id],
    )
    for policy_id, predicate in policies.items():
        for fold_id, start, end in (*FOLDS, ("ALL", date(2025, 8, 26), date(2026, 8, 27))):
            for horizon in (2, 3, 5, 10):
                selected = _non_overlapping(
                    [
                        row
                        for row in records
                        if row["horizon"] == horizon
                        and start <= row["decision_date"] <= end
                        and predicate(row)
                    ]
                )
                summary = _trial_summary(selected)
                status = "OBSERVED" if selected else "NO_OBSERVATIONS"
                result = {
                    "policy_id": policy_id,
                    "fold_id": fold_id,
                    "horizon": horizon,
                    **summary,
                    "status": status,
                }
                results.append(result)
                store.connection.execute(
                    """
                    INSERT INTO serenity_thesis_trials VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        OPTIMIZATION_ID,
                        round_id,
                        policy_id,
                        fold_id,
                        horizon,
                        summary["observation_count"],
                        summary["mean_net_return"],
                        summary["median_net_return"],
                        summary["win_rate"],
                        summary["alpha_csi300"],
                        summary["alpha_chain"],
                        summary["mean_mae"],
                        summary["mean_mfe"],
                        status,
                        datetime.now(),
                    ],
                )
    report = {
        "optimization_id": OPTIMIZATION_ID,
        "round_id": round_id,
        "data_window": {"start": "2025-08-26", "end": "2026-08-27"},
        "claim_boundary": "POSTHOC_DESCRIPTIVE_UNVERIFIED_ALPHA_NO_CAPITAL_AUTHORITY",
        "primary_policy": {
            "policy_id": THESIS_POLICY_ID,
            "entry": "next trading session open",
            "exit_horizons": [2, 3, 5, 10],
            "event_gate": "PASS + NEW_INFORMATION + economic bridge PASS + non-admin",
            "pdf64_gate": {
                "complete_score_min": MIN_COMPLETE_SCORE,
                "each_dimension_rating_min": MIN_DIMENSION_RATING,
                "consensus": (
                    "minimum evidenced rating across Round2 and Round4; "
                    "UNKNOWN is absence rather than contradiction"
                ),
            },
            "momentum_or_gap_filter": "NONE",
        },
        "secondary_overlay": (
            "P8 applies the previously explored momentum/gap timing only after the PDF64 gate; "
            "it is diagnostic and cannot rescue a failed primary thesis gate"
        ),
        "results": results,
    }
    _atomic_json(_optimization_root(store) / "round04-thesis-report.json", report)
    return report


def status(store: EventReplayStore) -> dict[str, Any]:
    initialize_thesis_tables(store)
    return {
        "optimization_id": OPTIMIZATION_ID,
        "support": store.connection.execute(
            """
            SELECT status, count(*), count(DISTINCT announcement_id)
            FROM serenity_thesis_support_documents
            WHERE optimization_id=? GROUP BY status ORDER BY status
            """,
            [OPTIMIZATION_ID],
        ).fetchall(),
        "scores": store.connection.execute(
            """
            SELECT count(*),
                   count(*) FILTER (WHERE event_gate='PASS'),
                   count(*) FILTER (WHERE complete_dimension_score IS NOT NULL),
                   avg(known_dimension_weight)
            FROM serenity_thesis_scores
            WHERE optimization_id=? AND substage=?
            """,
            [OPTIMIZATION_ID, THESIS_SUBSTAGE],
        ).fetchone(),
        "hard_gate_events": store.connection.execute(
            """
            SELECT event_id, symbol, decision_date, complete_dimension_score, dimension_json
            FROM serenity_thesis_consensus_scores
            WHERE optimization_id=? AND hard_gate_pass=true
            ORDER BY decision_date, event_id
            """,
            [OPTIMIZATION_ID],
        ).fetchall(),
        "parent_stage_cost_micros_cny": store.connection.execute(
            """
            SELECT coalesce(sum(cost_micros_cny), 0)
            FROM semantic_model_calls WHERE replay_id=? AND stage=?
            """,
            [OPTIMIZATION_ID, EVENT_ENRICHED_SCORE_STAGE],
        ).fetchone()[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("discover", "collect", "plan-score", "score", "evaluate", "status"),
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PYTHON"):
        os.environ["PYTHON"] = os.sys.executable
    with _pilot_lock(args.root):
        store = EventReplayStore(args.root)
        try:
            if args.command == "discover":
                payload = discover_and_plan(store)
            elif args.command == "collect":
                payload = collect_documents(store)
            elif args.command == "plan-score":
                payload = run_scores(store, execute=False)
            elif args.command == "score":
                payload = run_scores(store, execute=True)
            elif args.command == "evaluate":
                payload = evaluate(store)
            else:
                payload = status(store)
        finally:
            store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
