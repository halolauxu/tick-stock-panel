# ruff: noqa: RUF001
"""Budgeted, reproducible optimization for the one-year Serenity event study.

The module keeps the research boundary deliberately narrow:

* exactly the frozen 100-company / three-chain sample;
* exactly the locally available one-year price window;
* announcement outcomes are never placed in a DeepSeek prompt;
* every paid response is written to ``semantic_model_calls`` before use;
* missing PDF evidence remains unknown and never becomes a neutral score;
* the fixed CNY 60 owner authorization is shared by every optimization round.

This remains research-only.  It neither registers a public strategy nor authorizes
capital or broker activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.services.serenity_event_replay import EventReplayStore
from app.services.serenity_pdf_scoring import (
    DIMENSION_WEIGHTS,
    MODEL,
    PENALTY_GUIDANCE,
    RATING_ANCHORS,
    CachedModelResult,
    DocumentPacket,
    ModelCallSpec,
    ScoreState,
    _adapter_runner,
    _budget_environment,
    _canonical_page_quote,
    _file_hash,
    _freeze_json,
    _parse_pages,
    _runtime_inventory,
    _runtime_scripts,
    _score_schema,
    execute_cached_call,
    initialize_semantic_tables,
    sanitize_score_evidence,
    validate_score_output,
)
from app.services.serenity_pilot import _atomic_json, _json_default, _pilot_lock, _stable_hash

OPTIMIZATION_ID = "serenity-event-1y-opt-v1"
OPTIMIZATION_VERSION = "1.0.0"
EVENT_SCORE_STAGE = "SERENITY_EVENT_SCORE"
EVENT_ENRICHED_SCORE_STAGE = "SERENITY_EVENT_ENRICHED_SCORE"
STAGE_COST_CAP_MICROS_CNY = {
    EVENT_SCORE_STAGE: 15_000_000,
    EVENT_ENRICHED_SCORE_STAGE: 25_000_000,
}
MAX_COST_MICROS_CNY = 60_000_000
MAX_REQUESTS = 170
MAX_PROMPT_TOKENS = 8_000_000
MAX_COMPLETION_TOKENS = 700_000
MAX_TOTAL_TOKENS = 8_700_000
MAX_REQUEST_BYTES = 820_000
FROZEN_START = date(2025, 8, 26)
FROZEN_END = date(2026, 8, 27)
FROZEN_UNIVERSE_SIZE = 100
FROZEN_CHAIN_COUNT = 3

FOLDS = (
    ("F1", date(2025, 8, 26), date(2026, 1, 31)),
    ("F2", date(2026, 2, 1), date(2026, 4, 30)),
    ("F3", date(2026, 5, 1), date(2026, 6, 30)),
    ("F4", date(2026, 7, 1), date(2026, 8, 27)),
)


@dataclass(frozen=True)
class EventScoreState:
    event_id: str
    symbol: str
    decision_date: date
    event_type: str
    entity_id: str
    score_state: ScoreState
    document_map: dict[str, str]


def _canonical_bytes(payload: Any, *, exclude_hash: bool = False) -> bytes:
    if exclude_hash and isinstance(payload, dict):
        payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _with_content_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = hashlib.sha256(
        _canonical_bytes(result, exclude_hash=True)
    ).hexdigest()
    return result


def _optimization_root(store: EventReplayStore) -> Path:
    return store.root / "optimization" / OPTIMIZATION_ID


def initialize_optimization_tables(store: EventReplayStore) -> None:
    initialize_semantic_tables(store.connection)
    store.connection.execute(
        """
        CREATE TABLE IF NOT EXISTS serenity_event_features (
            optimization_id VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            event_type VARCHAR NOT NULL,
            deterministic_subtype VARCHAR NOT NULL,
            chain_id VARCHAR NOT NULL,
            fact_count INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            momentum_5 DOUBLE,
            momentum_20 DOUBLE,
            benchmark_momentum_20 DOUBLE,
            relative_momentum_20 DOUBLE,
            next_open_gap DOUBLE,
            feature_json VARCHAR NOT NULL,
            feature_hash VARCHAR NOT NULL,
            frozen_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, event_id)
        );
        CREATE TABLE IF NOT EXISTS serenity_event_semantic_scores (
            optimization_id VARCHAR NOT NULL,
            stage VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            entity_id VARCHAR NOT NULL,
            event_gate VARCHAR NOT NULL,
            event_stage VARCHAR NOT NULL,
            newness VARCHAR NOT NULL,
            economic_bridge VARCHAR NOT NULL,
            repricing_horizon VARCHAR NOT NULL,
            known_dimension_weight DOUBLE NOT NULL,
            known_dimension_points DOUBLE NOT NULL,
            dimension_score_lower_bound DOUBLE NOT NULL,
            dimension_score_upper_bound DOUBLE NOT NULL,
            complete_dimension_score DOUBLE,
            dimension_json VARCHAR NOT NULL,
            penalty_json VARCHAR NOT NULL,
            event_review_json VARCHAR NOT NULL,
            raw_output_json VARCHAR NOT NULL,
            model_input_sha256 VARCHAR NOT NULL,
            scored_at TIMESTAMP NOT NULL,
            PRIMARY KEY (optimization_id, stage, event_id)
        );
        CREATE TABLE IF NOT EXISTS serenity_optimization_trials (
            optimization_id VARCHAR NOT NULL,
            protocol_hash VARCHAR NOT NULL,
            trial_id VARCHAR NOT NULL,
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
            PRIMARY KEY (optimization_id, trial_id, fold_id, horizon)
        );
        """
    )


def classify_capacity_subtype(title: str) -> str:
    compact = re.sub(r"\s+", "", str(title or ""))
    if re.search(
        r"结项|节余募集|永久补充流动资金|募集资金等额置换|置换预先|变更募投|"
        r"调整.*募集资金|注销募集资金|核查意见",
        compact,
    ):
        return "FINANCING_ADMIN"
    if re.search(r"正式投产|投产|量产|试生产|产能释放|达产|竣工", compact):
        return "OPERATING_MILESTONE"
    if re.search(r"开工|进展|竞得土地|购买土地", compact):
        return "IMPLEMENTATION_PROGRESS"
    if re.search(r"拟投资|投资建设|建设项目|扩产|产能建设|募投项目", compact):
        return "CAPEX_PLAN"
    return "OTHER_CAPACITY"


def deterministic_subtype(event_type: str, title: str) -> str:
    if event_type == "CAPACITY_MILESTONE":
        return classify_capacity_subtype(title)
    return event_type


def _price_rows(store: EventReplayStore) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for symbol, day, open_, close, amount in store.connection.execute(
        """
        SELECT symbol, date, open, close, amount
        FROM research_daily_prices ORDER BY symbol, date
        """
    ).fetchall():
        grouped[str(symbol)].append(
            {
                "date": day,
                "open": float(open_),
                "close": float(close),
                "amount": float(amount) if amount is not None else None,
            }
        )
    return grouped


def _backward_return(rows: list[dict[str, Any]], index: int, sessions: int) -> float | None:
    if index < sessions or not rows[index - sessions]["close"]:
        return None
    return float(rows[index]["close"]) / float(rows[index - sessions]["close"]) - 1


def materialize_event_features(store: EventReplayStore) -> dict[str, Any]:
    initialize_optimization_tables(store)
    prices = _price_rows(store)
    indexes = {
        symbol: {row["date"]: index for index, row in enumerate(rows)}
        for symbol, rows in prices.items()
    }
    benchmark = prices.get("000300.SH", [])
    benchmark_index = indexes.get("000300.SH", {})
    rows = store.connection.execute(
        """
        SELECT e.event_id, e.symbol, e.decision_date, e.entry_date, e.title,
               e.primary_event_type, u.chain_id, coalesce(d.fact_count, 0),
               coalesce(d.pages, 0)
        FROM event_candidates e
        JOIN universe u USING (symbol)
        JOIN document_metrics d ON d.announcement_id=e.announcement_id
        WHERE EXISTS (
            SELECT 1 FROM event_discovery_outcomes o
            WHERE o.event_id=e.event_id AND o.status='SETTLED'
        )
        ORDER BY e.decision_date, e.symbol, e.event_id
        """
    ).fetchall()
    values: list[list[Any]] = []
    blocked = 0
    for (
        event_id,
        symbol,
        decision_day,
        entry_day,
        title,
        event_type,
        chain_id,
        facts,
        pages,
    ) in rows:
        symbol_rows = prices.get(str(symbol), [])
        index = indexes.get(str(symbol), {}).get(decision_day)
        entry_index = indexes.get(str(symbol), {}).get(entry_day)
        b_index = benchmark_index.get(decision_day)
        if index is None or entry_index is None or b_index is None:
            blocked += 1
            continue
        momentum_5 = _backward_return(symbol_rows, index, 5)
        momentum_20 = _backward_return(symbol_rows, index, 20)
        benchmark_momentum_20 = _backward_return(benchmark, b_index, 20)
        relative_momentum_20 = (
            momentum_20 - benchmark_momentum_20
            if momentum_20 is not None and benchmark_momentum_20 is not None
            else None
        )
        next_open_gap = (
            float(symbol_rows[entry_index]["open"]) / float(symbol_rows[index]["close"]) - 1
        )
        subtype = deterministic_subtype(str(event_type), str(title))
        feature = {
            "event_type": str(event_type),
            "deterministic_subtype": subtype,
            "fact_count": int(facts),
            "page_count": int(pages),
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "benchmark_momentum_20": benchmark_momentum_20,
            "relative_momentum_20": relative_momentum_20,
            "next_open_gap": next_open_gap,
            "point_in_time": True,
            "next_open_gap_is_execution_condition": True,
        }
        feature_json = json.dumps(feature, ensure_ascii=False, sort_keys=True)
        values.append(
            [
                OPTIMIZATION_ID,
                str(event_id),
                str(symbol),
                decision_day,
                str(event_type),
                subtype,
                str(chain_id),
                int(facts),
                int(pages),
                momentum_5,
                momentum_20,
                benchmark_momentum_20,
                relative_momentum_20,
                next_open_gap,
                feature_json,
                hashlib.sha256(feature_json.encode()).hexdigest(),
                datetime.now(),
            ]
        )
    store.connection.execute("BEGIN TRANSACTION")
    try:
        if values:
            store.connection.executemany(
                "INSERT OR REPLACE INTO serenity_event_features VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        store.connection.execute("COMMIT")
    except Exception:
        store.connection.execute("ROLLBACK")
        raise
    return {"materialized": len(values), "blocked": blocked}


def _event_score_schema() -> dict[str, Any]:
    schema = json.loads(json.dumps(_score_schema()))
    citation = json.loads(
        json.dumps(schema["properties"]["dimensions"]["items"]["properties"]["evidence"]["items"])
    )
    schema["required"].append("event_review")
    schema["properties"]["event_review"] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "event_gate",
            "event_stage",
            "newness",
            "economic_bridge",
            "repricing_horizon",
            "reason",
            "evidence",
        ],
        "properties": {
            "event_gate": {"enum": ["PASS", "FAIL", "DATA_INSUFFICIENT"]},
            "event_stage": {
                "enum": [
                    "ROUTINE_ADMIN",
                    "CAPEX_PLAN",
                    "IMPLEMENTATION",
                    "OPERATING_MILESTONE",
                    "CUSTOMER_OR_ORDER_VALIDATION",
                    "SUPPLY_OR_PRICE_SHOCK",
                    "EARNINGS_REVISION",
                    "OTHER",
                ]
            },
            "newness": {"enum": ["NEW_INFORMATION", "ROUTINE_UPDATE", "UNKNOWN"]},
            "economic_bridge": {"enum": ["PASS", "FAIL", "DATA_INSUFFICIENT"]},
            "repricing_horizon": {"enum": ["D2_D3", "D4_D5", "D6_D10", "NONE", "UNKNOWN"]},
            "reason": {"type": "string", "minLength": 2, "maxLength": 240},
            "evidence": {"type": "array", "maxItems": 6, "items": citation},
        },
    }
    return schema


def _mask_identity(text: str, *, symbol: str, code: str, name: str) -> str:
    masked = str(text or "")
    candidates = {
        symbol,
        code,
        name,
        name.replace("股份有限公司", ""),
        name.replace("有限公司", ""),
    }
    for candidate in sorted(
        (value for value in candidates if len(value) >= 3), key=len, reverse=True
    ):
        masked = masked.replace(candidate, "[发行人]")
    masked = re.sub(
        r"(?<!\d)(?:19|20)\d{2}\s*[年./-]\s*\d{1,2}\s*[月./-]\s*\d{1,2}\s*日?", "[历史日期]", masked
    )
    masked = re.sub(r"(?<!\d)(?:19|20)\d{2}\s*年", "[历史年份]", masked)
    return masked


def load_event_score_states(
    store: EventReplayStore, *, stage: str = EVENT_SCORE_STAGE
) -> tuple[list[EventScoreState], list[dict[str, Any]]]:
    universe = {row["symbol"]: row for row in store.universe()}
    anchors = store.connection.execute(
        """
        SELECT DISTINCT e.event_id, e.symbol, e.decision_date, e.primary_event_type,
               f.deterministic_subtype, f.fact_count
        FROM event_candidates e
        JOIN document_metrics d ON d.announcement_id=e.announcement_id
        JOIN serenity_event_features f
          ON f.optimization_id=? AND f.event_id=e.event_id
        WHERE EXISTS (
            SELECT 1 FROM event_discovery_outcomes o
            WHERE o.event_id=e.event_id AND o.status='SETTLED'
        )
        ORDER BY e.decision_date, e.symbol, e.event_id
        """,
        [OPTIMIZATION_ID],
    ).fetchall()
    states: list[EventScoreState] = []
    blocked: list[dict[str, Any]] = []
    for event_id, symbol, decision_day, event_type, subtype, fact_count in anchors:
        event_only_eligible = bool(
            (event_type == "CAPACITY_MILESTONE" and subtype != "FINANCING_ADMIN")
            or int(fact_count) > 0
        )
        if stage == EVENT_SCORE_STAGE and not event_only_eligible:
            blocked.append(
                {
                    "event_id": str(event_id),
                    "status": "DEFERRED_ZERO_COST_LOW_INFORMATION",
                    "selection_rule": (
                        "non-financing-admin capacity events OR regex fact_count>0"
                    ),
                }
            )
            continue
        company = universe[str(symbol)]
        documents = store.connection.execute(
            """
            SELECT e.announcement_id, e.published_at, e.title, d.sha256
            FROM event_candidates e
            JOIN document_metrics d ON d.announcement_id=e.announcement_id
            WHERE e.symbol=? AND e.decision_date=?
              AND e.polarity IN ('LONG_CANDIDATE', 'MIXED_REVIEW')
            ORDER BY e.published_at, e.announcement_id
            """,
            [symbol, decision_day],
        ).fetchall()
        packets: list[DocumentPacket] = []
        document_map: dict[str, str] = {}
        for index, (document_id, published_at, title, sha256) in enumerate(documents, start=1):
            text_path = store.text_dir / f"{document_id}.txt"
            if not text_path.is_file() or published_at is None:
                continue
            opaque_id = "DOC-" + _stable_hash(OPTIMIZATION_ID, str(event_id), str(index))[:16]
            original_pages = _parse_pages(text_path)
            pages = {
                page: _mask_identity(
                    value,
                    symbol=str(symbol),
                    code=str(company["code"]),
                    name=str(company["name"]),
                )
                for page, value in original_pages.items()
            }
            packets.append(
                DocumentPacket(
                    document_id=opaque_id,
                    announcement_time=published_at,
                    title=_mask_identity(
                        str(title),
                        symbol=str(symbol),
                        code=str(company["code"]),
                        name=str(company["name"]),
                    ),
                    pages=pages,
                    sha256=str(sha256),
                )
            )
            document_map[opaque_id] = str(document_id)
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
        if not packets:
            blocked.append({"event_id": str(event_id), "status": "DATA_INSUFFICIENT_NO_PDF"})
        elif len(prompt.encode("utf-8")) > MAX_REQUEST_BYTES - 30_000:
            blocked.append(
                {
                    "event_id": str(event_id),
                    "status": "DATA_INSUFFICIENT_OVERSIZE_CONTEXT",
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
    return states, blocked


def build_event_score_prompt(state: ScoreState, event_type: str) -> str:
    policy = {
        "task": "仅依据T0前给定PDF原文，核验事件并评定卡脖子五维64分和八项风险",
        "event_type_is_discovery_only": event_type,
        "rules": [
            "公告标题分类不是事实，必须以PDF正文判断事件阶段、新增性和经济传导",
            "ROUTINE_ADMIN包括纯募投结项、置换、补流、账户或保荐程序，除非正文有新增经营事实",
            "经济传导PASS必须有产品、产能、订单、收入、价格、成本、交付或客户验证的具体桥梁",
            "2到10交易日重定价窗口必须由正文中的新增催化支持，不得依据未来股价",
            "五个维度只能引用给定PDF逐字原句；未覆盖必须UNKNOWN/null，不能把缺失当0分",
            "0分也必须有明确反证；发行人自身PDF通常不能构成独立交叉验证",
            "评分只输出0到5整数，由本地程序换算为64分",
            "只输出Schema规定的JSON；quote为足以证明结论的最短连续原句",
        ],
        "rating_anchors": RATING_ANCHORS,
        "penalty_guidance": PENALTY_GUIDANCE,
    }
    parts = [
        "这是不连接券商的一年期历史研究。你看不到证券身份、绝对日期和任何事后收益。",
        "严禁使用外部知识、记忆中的公司信息或猜测，只输出满足Schema的单个JSON对象。",
        "评分政策：" + json.dumps(policy, ensure_ascii=False, sort_keys=True),
        f"entity_id={state.entity_id}; cutoff=T0; chain={state.chain_id}",
    ]
    for document in state.documents:
        parts.append(
            f"\n<DOCUMENT id={document.document_id} sha256={document.sha256} "
            f"published=T{(document.announcement_time.date() - state.cutoff_date).days:+d} "
            f"title={json.dumps(document.title, ensure_ascii=False)}>"
        )
        for page_number, page_text in sorted(document.pages.items()):
            parts.append(f"\n<PAGE number={page_number}>\n{page_text}\n</PAGE>")
        parts.append("\n</DOCUMENT>")
    return "\n".join(parts)


def _protocol_snapshot(store: EventReplayStore) -> dict[str, Any]:
    manifest = store.get_meta("event_manifest")
    if not manifest:
        raise RuntimeError("event replay manifest is missing")
    start = date.fromisoformat(str(manifest.get("start_date")))
    end = date.fromisoformat(str(manifest.get("end_date")))
    if (start, end) != (FROZEN_START, FROZEN_END):
        raise RuntimeError("optimizer is bound to the only available one-year window")
    universe = store.connection.execute(
        "SELECT count(*), count(DISTINCT chain_id) FROM universe"
    ).fetchone()
    if tuple(int(value) for value in universe) != (FROZEN_UNIVERSE_SIZE, FROZEN_CHAIN_COUNT):
        raise RuntimeError("optimizer requires the frozen 100-company / three-chain sample")
    calendar = [
        row[0].isoformat()
        for row in store.connection.execute(
            "SELECT date FROM research_daily_prices WHERE symbol='000300.SH' ORDER BY date"
        ).fetchall()
    ]
    event_rows = store.connection.execute(
        """
        SELECT e.event_id, e.metadata_hash, d.sha256
        FROM event_candidates e JOIN document_metrics d ON d.announcement_id=e.announcement_id
        WHERE EXISTS (
            SELECT 1 FROM event_discovery_outcomes o
            WHERE o.event_id=e.event_id AND o.status='SETTLED'
        ) ORDER BY e.event_id
        """
    ).fetchall()
    protocol = _with_content_hash(
        {
            "schema_version": OPTIMIZATION_VERSION,
            "optimization_id": OPTIMIZATION_ID,
            "data_window": {"start": start.isoformat(), "end": end.isoformat()},
            "data_window_rule": "ONLY_LOCALLY_AVAILABLE_ONE_YEAR_NO_THREE_YEAR_CLAIM",
            "universe_size": int(universe[0]),
            "chain_count": int(universe[1]),
            "calendar_sessions": len(calendar),
            "calendar_sha256": hashlib.sha256("\n".join(calendar).encode()).hexdigest(),
            "event_input_sha256": hashlib.sha256(_canonical_bytes(event_rows)).hexdigest(),
            "horizons": [2, 3, 5, 10],
            "folds": [
                {"fold_id": fold_id, "start": start_day.isoformat(), "end": end_day.isoformat()}
                for fold_id, start_day, end_day in FOLDS
            ],
            "trial_policy": "FIXED_SMALL_RULE_FAMILY_NO_PARAMETER_SWEEP",
            "seen_validation_rule": (
                "prior two-thirds/one-third result is exploratory; folds are walk-forward diagnostics, "
                "future daily shadow observations remain the real unseen confirmation"
            ),
            "paid_model": MODEL,
            "owner_budget_cny": 60,
            "paid_rounds": [
                "event_only_semantic_score",
                "evidence_enriched_rescore_if_round1_qualifies",
                "blind_red_team_only_for_surviving_policy",
            ],
            "budget_allocation_cny": {
                "event_only_semantic_score_cap": 15,
                "evidence_enriched_rescore_cap": 25,
                "blind_red_team_cap": 10,
                "unresolved_request_reserve": 10,
                "total_cap": 60,
            },
            "claim_boundary": "RETROSPECTIVE_OPTIMIZATION_UNVERIFIED_ALPHA_NO_CAPITAL_AUTHORITY",
        }
    )
    return protocol


def write_optimization_contract(store: EventReplayStore) -> dict[str, Path]:
    initialize_optimization_tables(store)
    root = _optimization_root(store)
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "event-score.schema.json"
    policy_path = root / "model-execution-policy.json"
    authorization_path = root / "model-budget-authorization.json"
    state_path = root / "model-budget-state.json"
    protocol_path = root / "optimization-protocol.json"
    _freeze_json(schema_path, _event_score_schema(), "event score schema")
    protocol = _protocol_snapshot(store)
    _freeze_json(protocol_path, protocol, "optimization protocol")
    policy = {
        "schema_version": "1.0.0",
        "policy_id": "SERENITY-EVENT-ONE-YEAR-DEEPSEEK-V4-FLASH-V1",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/chat/completions",
        "model": MODEL,
        "paid_execution_default": "DISABLED_WITHOUT_ASSET_OWNER_BUDGET",
        "network_execution_status": "ACTIVE_EXPLICIT_ASSET_OWNER_AUTHORIZATION",
        "thinking": "disabled",
        "http_timeout_seconds": 1100,
        "semantic_worker_limit": 1,
        "max_attempts_per_context_cap": 2,
        "stage_profiles": {
            stage: {
                "thinking_type": "disabled",
                "reasoning_effort": "high",
                "max_output_tokens": 4096,
                "max_request_bytes": MAX_REQUEST_BYTES,
                "max_attempts_per_context": 2,
                "max_retry_contexts": 3,
            }
            for stage in (EVENT_SCORE_STAGE, EVENT_ENRICHED_SCORE_STAGE)
        },
        "response_format": "json_object",
        "tools_allowed": False,
        "fallback_model_allowed": False,
        "mixed_provider_run_allowed": False,
        "credential_sources": ["PROCESS_ENV:DEEPSEEK_API_KEY"],
        "credential_persistence_allowed": False,
        "required_audit_fields": [
            "provider",
            "model",
            "request_sha256",
            "response_sha256",
            "finish_reason",
            "usage",
            "context_id",
        ],
    }
    _freeze_json(policy_path, policy, "optimization model policy")
    event_dates = [
        row[0].isoformat()
        for row in store.connection.execute(
            """
            SELECT DISTINCT decision_date FROM event_discovery_outcomes
            WHERE status='SETTLED' ORDER BY decision_date
            """
        ).fetchall()
    ]
    authorization = _with_content_hash(
        {
            "schema_version": "1.0.0",
            "budget_id": "SERENITY-EVENT-ONE-YEAR-CNY60-V1",
            "authorization_status": "APPROVED",
            "authorized_by": "ASSET_OWNER",
            "approval_evidence": "用户在Codex任务中明确授权本轮多轮策略优化总预算60元",
            "replay_id": OPTIMIZATION_ID,
            "allowed_trade_dates": event_dates,
            "provider": "deepseek",
            "model": MODEL,
            "policy_sha256": _file_hash(policy_path),
            "effective_at": "2026-08-28T00:00:00+08:00",
            "expires_at": "2026-09-04T23:59:00+08:00",
            "budget_state_path": str(state_path.resolve()),
            "limits": {
                "max_requests": MAX_REQUESTS,
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "max_total_tokens": MAX_TOTAL_TOKENS,
                "max_cost_micros_cny": MAX_COST_MICROS_CNY,
                "max_attempts_per_context": 2,
            },
            "tariff": {
                "input_cache_hit_micros_cny_per_million": 112000,
                "input_cache_miss_micros_cny_per_million": 3520000,
                "output_micros_cny_per_million": 10560000,
                "evidence": "https://api-docs.deepseek.com/quick_start/pricing; conservative 8 CNY/USD cap basis",
            },
        }
    )
    _freeze_json(authorization_path, authorization, "optimization model budget")
    runtime_manifest = _with_content_hash(
        {
            "optimization_id": OPTIMIZATION_ID,
            "protocol_sha256": _file_hash(protocol_path),
            "policy_sha256": _file_hash(policy_path),
            "authorization_sha256": _file_hash(authorization_path),
            "runtime_sha256": _runtime_inventory(),
        }
    )
    _freeze_json(root / "runtime-manifest.json", runtime_manifest, "optimization runtime")
    return {
        "run_root": root,
        "score_schema": schema_path,
        "policy": policy_path,
        "authorization": authorization_path,
        "state": state_path,
        "protocol": protocol_path,
    }


def _usage_path(paths: dict[str, Path], trade_date: date) -> Path:
    return paths["run_root"] / f"model-usage-{trade_date.isoformat()}.jsonl"


def _preflight_stage(
    paths: dict[str, Path], *, stage: str, trade_date: date, prompts: list[str]
) -> dict[str, Any]:
    prompts_path = paths["run_root"] / f"{stage.lower()}-pending-prompts.json"
    _atomic_json(prompts_path, prompts)
    helper = (
        "import json,os,sys; from pathlib import Path; "
        "sys.path.insert(0, os.environ['RUNTIME_SCRIPTS']); "
        "from deepseek_payload import build_request_payload; "
        "from model_budget import preflight_stage,stage_profile; "
        "schema=json.loads(Path(os.environ['SCHEMA_PATH']).read_text()); "
        "prompts=json.loads(Path(os.environ['PROMPTS_PATH']).read_text()); "
        "profile=stage_profile(os.environ['ASHARE_LLM_STAGE']); "
        "thinking=profile.get('thinking_type','disabled'); "
        "total=sum(len(json.dumps(build_request_payload(p,schema,profile,thinking_type=thinking),"
        "ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()) for p in prompts); "
        "print(json.dumps({'request_bytes':total,'preflight':preflight_stage("
        "stage=os.environ['ASHARE_LLM_STAGE'],context_count=len(prompts),request_byte_count_total=total)},sort_keys=True))"
    )
    dummy = ModelCallSpec(
        OPTIMIZATION_ID,
        stage,
        "SLOT-PREFLIGHT-ONLY",
        trade_date,
        "preflight",
        paths["score_schema"],
        _file_hash(paths["policy"]),
    )
    env = _budget_environment(paths, dummy)
    env.update(
        {
            "RUNTIME_SCRIPTS": str(_runtime_scripts()),
            "SCHEMA_PATH": str(paths["score_schema"]),
            "PROMPTS_PATH": str(prompts_path),
        }
    )
    completed = subprocess.run(
        [os.environ.get("PYTHON", sys.executable), "-c", helper],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"whole-stage paid budget preflight failed: {completed.stderr[:500]}")
    return json.loads(completed.stdout)


def _event_review_is_bound(
    output: dict[str, Any], documents: dict[str, dict[int, str]]
) -> tuple[dict[str, Any], int]:
    sanitized = json.loads(json.dumps(output, ensure_ascii=False))
    review = sanitized.get("event_review") or {}
    evidence = review.get("evidence") or []
    invalid = 0
    for citation in evidence:
        canonical = _canonical_page_quote(citation, documents)
        if canonical is None:
            invalid += 1
        else:
            citation["quote"] = canonical
    if invalid or (review.get("event_gate") == "PASS" and not evidence):
        review.update(
            {
                "event_gate": "DATA_INSUFFICIENT",
                "newness": "UNKNOWN",
                "economic_bridge": "DATA_INSUFFICIENT",
                "repricing_horizon": "UNKNOWN",
                "reason": "事件引用无法逐字绑定给定PDF，已按证据不足处理",
                "evidence": [],
            }
        )
    return sanitized, invalid


def _dimension_score(output: dict[str, Any]) -> dict[str, float | None]:
    dimensions = {item["dimension_id"]: item for item in output.get("dimensions", [])}
    known_weight = 0.0
    known_points = 0.0
    for dimension_id, weight in DIMENSION_WEIGHTS.items():
        item = dimensions.get(dimension_id, {})
        if item.get("status") == "UNKNOWN" or item.get("rating") is None:
            continue
        known_weight += float(weight)
        known_points += int(item["rating"]) / 5 * float(weight)
    total_weight = float(sum(DIMENSION_WEIGHTS.values()))
    complete = round(known_points, 4) if known_weight == total_weight else None
    return {
        "known_weight": known_weight,
        "known_points": round(known_points, 4),
        "lower_bound": round(known_points, 4),
        "upper_bound": round(known_points + total_weight - known_weight, 4),
        "complete": complete,
    }


def run_event_scores(
    store: EventReplayStore,
    paths: dict[str, Path],
    *,
    execute: bool,
    stage: str = EVENT_SCORE_STAGE,
    runner: Callable[[ModelCallSpec], CachedModelResult] | None = None,
) -> dict[str, Any]:
    initialize_optimization_tables(store)
    states, blocked = load_event_score_states(store, stage=stage)
    completed = {
        row[0]
        for row in store.connection.execute(
            """
            SELECT event_id FROM serenity_event_semantic_scores
            WHERE optimization_id=? AND stage=?
            """,
            [OPTIMIZATION_ID, stage],
        ).fetchall()
    }
    policy_hash = _file_hash(paths["policy"])
    specs = [
        ModelCallSpec(
            OPTIMIZATION_ID,
            stage,
            "SLOT-EVENT-" + _stable_hash(OPTIMIZATION_ID, stage, state.event_id)[:28],
            state.decision_date,
            build_event_score_prompt(state.score_state, state.event_type),
            paths["score_schema"],
            policy_hash,
        )
        for state in states
        if state.event_id not in completed
    ]
    pending_states = [state for state in states if state.event_id not in completed]
    preflight = (
        _preflight_stage(
            paths,
            stage=stage,
            trade_date=specs[0].trade_date,
            prompts=[spec.prompt for spec in specs],
        )
        if specs
        else {"status": "NO_PENDING_CALLS"}
    )
    worst_case_cost = int(
        (preflight.get("preflight") or {})
        .get("worst_case_increment", {})
        .get("charged_cost_micros_cny", 0)
    )
    stage_cost_cap = STAGE_COST_CAP_MICROS_CNY[stage]
    if worst_case_cost > stage_cost_cap:
        raise RuntimeError(
            f"{stage} worst-case cost {worst_case_cost} exceeds frozen stage cap {stage_cost_cap}"
        )
    plan = {
        "stage": stage,
        "total_states": len(states),
        "already_complete": len(completed),
        "pending": len(specs),
        "blocked": blocked,
        "preflight": preflight,
        "execute": execute,
    }
    paid_population = _with_content_hash(
        {
            "optimization_id": OPTIMIZATION_ID,
            "stage": stage,
            "selection_rule": (
                "(CAPACITY_MILESTONE AND subtype!=FINANCING_ADMIN) OR fact_count>0"
            ),
            "selection_uses_outcomes": False,
            "pending_slots": [
                {
                    "event_id": state.event_id,
                    "slot_id": spec.slot_id,
                    "input_sha256": spec.input_sha256,
                }
                for state, spec in zip(pending_states, specs, strict=True)
            ],
        }
    )
    population_path = paths["run_root"] / f"{stage.lower()}-paid-population.json"
    if population_path.is_file():
        existing = json.loads(population_path.read_text(encoding="utf-8"))
        if existing != paid_population:
            paid_calls = store.connection.execute(
                """
                SELECT count(*) FROM semantic_model_calls
                WHERE replay_id=? AND stage=?
                """,
                [OPTIMIZATION_ID, stage],
            ).fetchone()[0]
            if paid_calls:
                raise RuntimeError("paid population cannot change after the first model call")
            correction = _with_content_hash(
                {
                    **{key: value for key, value in paid_population.items() if key != "content_sha256"},
                    "supersedes_sha256": _file_hash(population_path),
                    "correction_reason": (
                        "preflight-only population exceeded the frozen stage cap; no paid call occurred"
                    ),
                }
            )
            correction_path = (
                paths["run_root"] / f"{stage.lower()}-paid-population-correction-01.json"
            )
            _freeze_json(correction_path, correction, f"{stage} corrected paid population")
            plan["paid_population_path"] = str(correction_path)
        else:
            plan["paid_population_path"] = str(population_path)
    else:
        _freeze_json(population_path, paid_population, f"{stage} paid population")
        plan["paid_population_path"] = str(population_path)
    _atomic_json(paths["run_root"] / f"{stage.lower()}-plan.json", plan)
    if not execute or not specs:
        return plan
    paid_runner = runner or _adapter_runner(paths)
    for index, (state, spec) in enumerate(zip(pending_states, specs, strict=True), start=1):
        result = execute_cached_call(store.connection, spec, paid_runner)
        raw_output = json.loads(result.raw_output)
        documents = {
            document.document_id: document.pages for document in state.score_state.documents
        }
        sanitized, _adjustments = sanitize_score_evidence(raw_output, documents)
        sanitized, event_citation_failures = _event_review_is_bound(sanitized, documents)
        errors = validate_score_output(sanitized, entity_id=state.entity_id, documents=documents)
        if errors:
            raise RuntimeError(
                f"semantic score validation failed for {state.event_id}: {'; '.join(errors[:5])}"
            )
        review = sanitized["event_review"]
        dimension_score = _dimension_score(sanitized)
        store.connection.execute(
            """
            INSERT OR REPLACE INTO serenity_event_semantic_scores VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                OPTIMIZATION_ID,
                stage,
                state.event_id,
                state.symbol,
                state.decision_date,
                state.entity_id,
                review["event_gate"],
                review["event_stage"],
                review["newness"],
                review["economic_bridge"],
                review["repricing_horizon"],
                dimension_score["known_weight"],
                dimension_score["known_points"],
                dimension_score["lower_bound"],
                dimension_score["upper_bound"],
                dimension_score["complete"],
                json.dumps(sanitized["dimensions"], ensure_ascii=False, sort_keys=True),
                json.dumps(sanitized["penalties"], ensure_ascii=False, sort_keys=True),
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
                    "event": "serenity_event_score_progress",
                    "completed": len(completed) + index,
                    "total": len(states),
                    "event_gate": review["event_gate"],
                    "event_stage": review["event_stage"],
                    "known_dimension_weight": dimension_score["known_weight"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {**plan, "completed_after_run": len(states)}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _trial_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observation_count": len(rows),
        "mean_net_return": _mean([row["net_return"] for row in rows]),
        "median_net_return": _median([row["net_return"] for row in rows]),
        "win_rate": _mean([1.0 if row["net_return"] > 0 else 0.0 for row in rows]),
        "alpha_csi300": _mean([row["net_return"] - row["benchmark_return"] for row in rows]),
        "alpha_chain": _mean([row["net_return"] - row["chain_return"] for row in rows]),
        "mean_mae": _mean([row["mae"] for row in rows]),
        "mean_mfe": _mean([row["mfe"] for row in rows]),
    }


def evaluate_trials(store: EventReplayStore) -> dict[str, Any]:
    initialize_optimization_tables(store)
    protocol_path = _optimization_root(store) / "optimization-protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("frozen optimization protocol is missing")
    protocol_hash = _file_hash(protocol_path)
    rows = store.connection.execute(
        """
        SELECT o.event_id, o.horizon, o.decision_date, o.net_return,
               o.benchmark_return, o.chain_return, o.mae, o.mfe,
               f.deterministic_subtype, f.relative_momentum_20, f.next_open_gap,
               s.event_gate, s.event_stage, s.newness, s.economic_bridge,
               s.known_dimension_weight, s.known_dimension_points,
               s.complete_dimension_score, s.dimension_json
        FROM event_discovery_outcomes o
        JOIN serenity_event_features f
          ON f.optimization_id=? AND f.event_id=o.event_id
        LEFT JOIN serenity_event_semantic_scores s
          ON s.optimization_id=? AND s.stage=? AND s.event_id=o.event_id
        WHERE o.status='SETTLED'
        ORDER BY o.decision_date, o.event_id, o.horizon
        """,
        [OPTIMIZATION_ID, OPTIMIZATION_ID, EVENT_SCORE_STAGE],
    ).fetchall()
    records = [
        {
            "event_id": row[0],
            "horizon": int(row[1]),
            "decision_date": row[2],
            "net_return": float(row[3]),
            "benchmark_return": float(row[4]),
            "chain_return": float(row[5]),
            "mae": float(row[6]),
            "mfe": float(row[7]),
            "subtype": row[8],
            "relative_momentum_20": row[9],
            "next_open_gap": row[10],
            "event_gate": row[11],
            "event_stage": row[12],
            "newness": row[13],
            "economic_bridge": row[14],
            "known_dimension_weight": float(row[15]) if row[15] is not None else 0.0,
            "known_dimension_points": float(row[16]) if row[16] is not None else 0.0,
            "complete_dimension_score": row[17],
            "dimensions": json.loads(row[18]) if row[18] else [],
        }
        for row in rows
    ]

    def dimension_gates(row: dict[str, Any]) -> bool:
        dimensions = {item["dimension_id"]: item for item in row["dimensions"]}
        minimums = {
            "architecture_coupling": 3,
            "chokepoint_severity": 3,
            "supplier_concentration": 3,
            "expansion_difficulty": 3,
            "evidence_quality": 3,
        }
        return all(
            dimensions.get(key, {}).get("status") != "UNKNOWN"
            and int(dimensions[key]["rating"]) >= minimum
            for key, minimum in minimums.items()
        )

    trials: dict[str, Callable[[dict[str, Any]], bool]] = {
        "T0_TITLE_DISCOVERY_BASELINE": lambda row: True,
        "T1_DETERMINISTIC_NON_ADMIN": lambda row: row["subtype"] != "FINANCING_ADMIN",
        "T2_SEMANTIC_EVENT_GATE": lambda row: (
            row["event_gate"] == "PASS"
            and row["newness"] == "NEW_INFORMATION"
            and row["economic_bridge"] == "PASS"
            and row["event_stage"] != "ROUTINE_ADMIN"
        ),
        "T3_STRICT_FULL64_GATES": lambda row: (
            row["event_gate"] == "PASS"
            and row["newness"] == "NEW_INFORMATION"
            and row["economic_bridge"] == "PASS"
            and dimension_gates(row)
        ),
        "T4_RESEARCH_COVERAGE_GATE": lambda row: (
            row["event_gate"] == "PASS"
            and row["newness"] == "NEW_INFORMATION"
            and row["economic_bridge"] == "PASS"
            and row["known_dimension_weight"] >= 32
            and row["known_dimension_points"] / row["known_dimension_weight"] >= 0.6
        ),
    }
    store.connection.execute(
        "DELETE FROM serenity_optimization_trials WHERE optimization_id=?",
        [OPTIMIZATION_ID],
    )
    result_rows: list[dict[str, Any]] = []
    all_folds = (*FOLDS, ("ALL", FROZEN_START, FROZEN_END))
    for trial_id, predicate in trials.items():
        for fold_id, fold_start, fold_end in all_folds:
            for horizon in (2, 3, 5, 10):
                selected = [
                    row
                    for row in records
                    if row["horizon"] == horizon
                    and fold_start <= row["decision_date"] <= fold_end
                    and predicate(row)
                ]
                summary = _trial_summary(selected)
                status = "OBSERVED" if selected else "NO_OBSERVATIONS"
                result = {
                    "trial_id": trial_id,
                    "fold_id": fold_id,
                    "horizon": horizon,
                    **summary,
                    "status": status,
                }
                result_rows.append(result)
                store.connection.execute(
                    """
                    INSERT INTO serenity_optimization_trials VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        OPTIMIZATION_ID,
                        protocol_hash,
                        trial_id,
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
        "protocol_sha256": protocol_hash,
        "data_window": {"start": FROZEN_START.isoformat(), "end": FROZEN_END.isoformat()},
        "trials": result_rows,
        "semantic_coverage": store.connection.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE event_gate='PASS'),
                   avg(known_dimension_weight),
                   count(*) FILTER (WHERE complete_dimension_score IS NOT NULL)
            FROM serenity_event_semantic_scores
            WHERE optimization_id=? AND stage=?
            """,
            [OPTIMIZATION_ID, EVENT_SCORE_STAGE],
        ).fetchone(),
        "alpha_status": "UNVERIFIED_ALPHA",
        "selection_status": "NO_AUTOMATIC_WINNER; FUTURE_SHADOW_CONFIRMATION_REQUIRED",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_json(_optimization_root(store) / "trial-report.json", report)
    return report


def status_report(store: EventReplayStore) -> dict[str, Any]:
    initialize_optimization_tables(store)
    root = _optimization_root(store)
    budget_state = (
        json.loads((root / "model-budget-state.json").read_text(encoding="utf-8"))
        if (root / "model-budget-state.json").is_file()
        else None
    )
    return {
        "optimization_id": OPTIMIZATION_ID,
        "features": store.connection.execute(
            "SELECT count(*) FROM serenity_event_features WHERE optimization_id=?",
            [OPTIMIZATION_ID],
        ).fetchone()[0],
        "semantic_scores": store.connection.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE event_gate='PASS'),
                   count(*) FILTER (WHERE complete_dimension_score IS NOT NULL),
                   avg(known_dimension_weight)
            FROM serenity_event_semantic_scores WHERE optimization_id=?
            """,
            [OPTIMIZATION_ID],
        ).fetchone(),
        "semantic_calls": store.connection.execute(
            """
            SELECT count(*), coalesce(sum(cost_micros_cny),0)
            FROM semantic_model_calls WHERE replay_id=?
            """,
            [OPTIMIZATION_ID],
        ).fetchone(),
        "budget_usage": budget_state.get("usage") if budget_state else None,
        "trial_rows": store.connection.execute(
            "SELECT count(*) FROM serenity_optimization_trials WHERE optimization_id=?",
            [OPTIMIZATION_ID],
        ).fetchone()[0],
        "claim_boundary": "UNVERIFIED_ALPHA_NO_CAPITAL_AUTHORITY",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "plan-score", "score", "evaluate", "status"),
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    with _pilot_lock(args.root):
        store = EventReplayStore(args.root)
        try:
            if args.command == "prepare":
                payload = {
                    "features": materialize_event_features(store),
                    "contracts": {
                        key: str(value) for key, value in write_optimization_contract(store).items()
                    },
                }
            elif args.command in {"plan-score", "score"}:
                paths = write_optimization_contract(store)
                payload = run_event_scores(store, paths, execute=args.command == "score")
            elif args.command == "evaluate":
                payload = evaluate_trials(store)
            else:
                payload = status_report(store)
        finally:
            store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
