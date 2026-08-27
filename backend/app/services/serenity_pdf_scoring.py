# ruff: noqa: RUF001
"""Evidence-bound DeepSeek scoring for the bounded Serenity PDF pilot.

This module extends the existing seven-session engineering pilot without
changing its frozen 36-point decisions.  It evaluates the five PDF-dependent
dimensions, persists every paid response in the pilot DuckDB, and materializes
a separate research-only score/outcome ledger.  It is deliberately not a
clean-room Alpha replay and never authorizes capital or submits orders.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

import duckdb

from app import secrets_store
from app.config import settings
from app.services.ai_provider import current_ai_model, current_ai_provider
from app.services.serenity_pilot import (
    DEFAULT_COST_BPS,
    HORIZONS,
    PilotStore,
    _load_index_rows,
    _load_market_rows,
    _stable_hash,
)

REPLAY_ID = "serenity-historical-7td-20260826-full64-v1"
SCORING_VERSION = "1.0.0"
PROVIDER = "deepseek"
MODEL = "deepseek-v4-flash"
OFFICIAL_BASE_URL = "https://api.deepseek.com"
FACT_AUDIT_STAGE = "FACT_AUDIT"
SCORE_STAGE = "SERENITY_PDF_SCORE"
FACT_AUDIT_SAMPLE_SIZE = 50
FACT_AUDIT_BATCH_SIZE = 5
FACT_AUDIT_PASS_RATE = 0.80
MAX_SCORE_REQUEST_BYTES = 820_000
MAX_REQUESTS = 120
MAX_PROMPT_TOKENS = 25_000_000
MAX_COMPLETION_TOKENS = 1_000_000
MAX_TOTAL_TOKENS = 26_000_000
MAX_COST_MICROS_CNY = 120_000_000
SELECTION_THRESHOLD = 70.0
PENALTY_MULTIPLIER = 2.0
REQUIRED_RUNTIME_FILES = (
    "recommend-ashare-next-day/scripts/deepseek-codex-adapter.py",
    "recommend-ashare-next-day/scripts/deepseek_payload.py",
    "recommend-ashare-next-day/scripts/model_budget.py",
    "recommend-ashare-next-day/scripts/p0_common.py",
    "recommend-ashare-next-day/scripts/audit-model-budget.py",
    "schemas/model-budget-authorization.schema.json",
    "schemas/model-budget-state.schema.json",
)

DIMENSION_WEIGHTS = {
    "architecture_coupling": 10,
    "chokepoint_severity": 15,
    "supplier_concentration": 12,
    "expansion_difficulty": 12,
    "evidence_quality": 15,
}
PENALTY_IDS = (
    "dilution_financing",
    "governance",
    "geopolitics",
    "liquidity",
    "hype_risk",
    "accounting_quality",
    "cyclicality",
    "alternative_design_risk",
)

RATING_ANCHORS = {
    "architecture_coupling": {
        "0": "原文明确为标准化、即插即用且替换不需改设计",
        "1": "非核心部件，已有多个可互换方案",
        "2": "需要有限适配或短期认证，但不改变主要系统架构",
        "3": "与系统性能或工艺明显耦合，替换需较长验证或局部重构",
        "4": "替换会导致主要系统重构、良率或性能显著受损",
        "5": "架构定义型依赖；移除会阻断系统，且有多项量化一级证据",
    },
    "chokepoint_severity": {
        "0": "原文明确无约束且存在充足替代供给",
        "1": "只造成轻微成本或排期影响",
        "2": "造成局部延误、成本或效率损失，有成熟绕行方案",
        "3": "显著影响产量、良率、收入或交付，绕行代价较高",
        "4": "可导致停产、严重良率损失或长时间交付中断",
        "5": "不可绕过的系统级或行业级阻断，后果有量化一级证据",
    },
    "supplier_concentration": {
        "0": "原文明确至少10家合格供应商或无集中风险",
        "1": "6至9家可替代供应商",
        "2": "4至5家合格供应商",
        "3": "2至3家、单一区域高度集中或替代认证显著受限",
        "4": "1至2家且存在独供或近独供事实",
        "5": "有效垄断、唯一验证来源或份额超过80%的量化一级证据",
    },
    "expansion_difficulty": {
        "0": "原文明确3个月内可扩、资本低且设备通用",
        "1": "6个月内可扩，约束较少",
        "2": "6至12个月且有中等资本或认证要求",
        "3": "12至24个月并包含客户认证、良率爬坡或专用设备",
        "4": "超过24个月、高资本、许可或关键设备材料受限",
        "5": "超过36个月且同时受稀缺上游、专用设备、许可与良率壁垒约束",
    },
    "evidence_quality": {
        "0": "决定性主张不可追溯、相互矛盾或被原文否定",
        "1": "只有单一模糊自述，无量化事实",
        "2": "一份一级来源，事实明确但未量化或覆盖不完整",
        "3": "至少两项具体量化一级事实且内部一致，但仍为同一发行人来源",
        "4": "量化一级事实并有独立客户、监管、合同或第三方一级来源交叉验证",
        "5": "多项相互独立且可复核的审计、监管、客户或合同级证据，反证已闭合",
    },
}

PENALTY_GUIDANCE = {
    "dilution_financing": "再融资、可转债、股权稀释或持续融资依赖",
    "governance": "治理、控股股东、关联交易、内部控制或管理层诚信风险",
    "geopolitics": "出口管制、制裁、跨境供应或单一海外区域风险",
    "liquidity": "证券交易流动性或业务现金流动性风险",
    "hype_risk": "概念宣传与已实现收入、订单、交付明显不匹配",
    "accounting_quality": "审计、会计政策、应收、存货、减值或利润质量风险",
    "cyclicality": "价格、库存、资本开支或需求强周期风险",
    "alternative_design_risk": "替代技术、绕行架构或客户自研导致卡点失效",
}


@dataclass(frozen=True)
class ModelCallSpec:
    replay_id: str
    stage: str
    slot_id: str
    trade_date: date
    prompt: str
    schema_path: Path
    policy_sha256: str = ""

    @property
    def input_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.replay_id.encode())
        digest.update(b"\0")
        digest.update(self.stage.encode())
        digest.update(b"\0")
        digest.update(self.slot_id.encode())
        digest.update(b"\0")
        digest.update(self.trade_date.isoformat().encode())
        digest.update(b"\0")
        digest.update(self.policy_sha256.encode())
        digest.update(b"\0")
        digest.update(self.schema_path.read_bytes())
        digest.update(b"\0")
        digest.update(self.prompt.encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class CachedModelResult:
    raw_output: str
    response_sha256: str
    output_sha256: str
    context_id: str
    api_request_id: str | None
    system_fingerprint: str | None
    finish_reason: str
    usage: dict[str, int]
    cost_micros_cny: int
    adapter_request_sha256: str | None = None


@dataclass(frozen=True)
class DocumentPacket:
    document_id: str
    announcement_time: datetime
    title: str
    pages: dict[int, str]
    sha256: str


@dataclass(frozen=True)
class ScoreState:
    symbol: str
    name: str
    chain_id: str
    cutoff_date: date
    entity_id: str
    documents: tuple[DocumentPacket, ...]


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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def _freeze_json(path: Path, payload: Any, label: str) -> None:
    """Create one immutable JSON contract or verify an identical prior copy."""
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"frozen {label} already exists with different content")
        return
    _atomic_json(path, payload)


def initialize_semantic_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_model_calls (
            replay_id VARCHAR NOT NULL,
            stage VARCHAR NOT NULL,
            slot_id VARCHAR NOT NULL,
            input_sha256 VARCHAR NOT NULL,
            adapter_request_sha256 VARCHAR,
            trade_date DATE NOT NULL,
            provider VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            raw_output_json VARCHAR NOT NULL,
            response_sha256 VARCHAR NOT NULL,
            output_sha256 VARCHAR NOT NULL,
            context_id VARCHAR NOT NULL,
            api_request_id VARCHAR,
            system_fingerprint VARCHAR,
            finish_reason VARCHAR NOT NULL,
            usage_json VARCHAR NOT NULL,
            cost_micros_cny BIGINT NOT NULL,
            persisted_at TIMESTAMP NOT NULL,
            PRIMARY KEY (replay_id, stage, slot_id, input_sha256)
        );
        CREATE TABLE IF NOT EXISTS semantic_fact_audit (
            replay_id VARCHAR NOT NULL,
            fact_id VARCHAR NOT NULL,
            batch_slot_id VARCHAR NOT NULL,
            verdict VARCHAR NOT NULL,
            normalized_claim VARCHAR NOT NULL,
            quote VARCHAR NOT NULL,
            reason VARCHAR NOT NULL,
            model_input_sha256 VARCHAR NOT NULL,
            reviewed_at TIMESTAMP NOT NULL,
            PRIMARY KEY (replay_id, fact_id)
        );
        CREATE TABLE IF NOT EXISTS semantic_score_results (
            replay_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            cutoff_date DATE NOT NULL,
            entity_id VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            base_36_points DOUBLE,
            pdf_points DOUBLE,
            raw_factor_score DOUBLE,
            known_penalty_points DOUBLE NOT NULL,
            risk_adjusted_score_upper_bound DOUBLE,
            risk_adjusted_score_exact DOUBLE,
            dimension_json VARCHAR NOT NULL,
            penalty_json VARCHAR NOT NULL,
            kill_switch_json VARCHAR NOT NULL,
            raw_output_json VARCHAR NOT NULL,
            model_input_sha256 VARCHAR NOT NULL,
            scored_at TIMESTAMP NOT NULL,
            PRIMARY KEY (replay_id, symbol, cutoff_date)
        );
        CREATE TABLE IF NOT EXISTS serenity_full_decisions (
            replay_id VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            chain_id VARCHAR NOT NULL,
            source_cutoff_date DATE,
            base_36_points DOUBLE,
            pdf_points DOUBLE,
            raw_factor_score DOUBLE,
            known_penalty_points DOUBLE NOT NULL,
            research_score DOUBLE,
            status VARCHAR NOT NULL,
            research_selected BOOLEAN NOT NULL,
            capital_authorized BOOLEAN NOT NULL DEFAULT FALSE,
            frozen_at TIMESTAMP NOT NULL,
            PRIMARY KEY (replay_id, decision_date, symbol)
        );
        CREATE TABLE IF NOT EXISTS serenity_full_outcomes (
            replay_id VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            entry_date DATE,
            exit_date DATE,
            net_return DOUBLE,
            benchmark_return DOUBLE,
            mae DOUBLE,
            mfe DOUBLE,
            status VARCHAR NOT NULL,
            settled_at TIMESTAMP,
            PRIMARY KEY (replay_id, decision_date, symbol, horizon)
        );
        """
    )


def _result_from_row(row: tuple[Any, ...]) -> CachedModelResult:
    return CachedModelResult(
        raw_output=str(row[0]),
        response_sha256=str(row[1]),
        output_sha256=str(row[2]),
        context_id=str(row[3]),
        api_request_id=row[4],
        system_fingerprint=row[5],
        finish_reason=str(row[6]),
        usage=json.loads(row[7]),
        cost_micros_cny=int(row[8]),
        adapter_request_sha256=row[9],
    )


def execute_cached_call(
    connection: duckdb.DuckDBPyConnection,
    spec: ModelCallSpec,
    runner: Callable[[ModelCallSpec], CachedModelResult],
) -> CachedModelResult:
    """Return an exact cached result or persist one paid adapter result atomically."""
    initialize_semantic_tables(connection)
    row = connection.execute(
        """
        SELECT raw_output_json, response_sha256, output_sha256, context_id,
               api_request_id, system_fingerprint, finish_reason, usage_json,
               cost_micros_cny, adapter_request_sha256
        FROM semantic_model_calls
        WHERE replay_id=? AND stage=? AND slot_id=? AND input_sha256=? AND status='PASS'
        """,
        [spec.replay_id, spec.stage, spec.slot_id, spec.input_sha256],
    ).fetchone()
    if row:
        cached = _result_from_row(row)
        if hashlib.sha256(cached.raw_output.encode()).hexdigest() != cached.output_sha256:
            raise RuntimeError("persisted model output hash mismatch")
        return cached

    result = runner(spec)
    json.loads(result.raw_output)
    if hashlib.sha256(result.raw_output.encode()).hexdigest() != result.output_sha256:
        raise RuntimeError("paid model output hash does not match the adapter receipt")
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT INTO semantic_model_calls VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, 'PASS', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (replay_id, stage, slot_id, input_sha256) DO UPDATE SET
                adapter_request_sha256=excluded.adapter_request_sha256,
                trade_date=excluded.trade_date,
                provider=excluded.provider,
                model=excluded.model,
                status='PASS',
                raw_output_json=excluded.raw_output_json,
                response_sha256=excluded.response_sha256,
                output_sha256=excluded.output_sha256,
                context_id=excluded.context_id,
                api_request_id=excluded.api_request_id,
                system_fingerprint=excluded.system_fingerprint,
                finish_reason=excluded.finish_reason,
                usage_json=excluded.usage_json,
                cost_micros_cny=excluded.cost_micros_cny,
                persisted_at=excluded.persisted_at
            """,
            [
                spec.replay_id,
                spec.stage,
                spec.slot_id,
                spec.input_sha256,
                result.adapter_request_sha256,
                spec.trade_date,
                PROVIDER,
                MODEL,
                result.raw_output,
                result.response_sha256,
                result.output_sha256,
                result.context_id,
                result.api_request_id,
                result.system_fingerprint,
                result.finish_reason,
                json.dumps(result.usage, sort_keys=True),
                result.cost_micros_cny,
                datetime.now(),
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return result


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _citation_is_bound(citation: dict[str, Any], documents: dict[str, dict[int, str]]) -> bool:
    document_id = str(citation.get("document_id") or "")
    page_number = citation.get("page_number")
    quote = str(citation.get("quote") or "").strip()
    if document_id not in documents or not isinstance(page_number, int) or len(quote) < 6:
        return False
    page = documents[document_id].get(page_number, "")
    return _normalize_text(quote) in _normalize_text(page)


def validate_score_output(
    output: dict[str, Any], *, entity_id: str, documents: dict[str, dict[int, str]]
) -> list[str]:
    errors: list[str] = []
    if output.get("schema_version") != "1.0.0":
        errors.append("schema_version不正确")
    if output.get("entity_id") != entity_id:
        errors.append("entity_id不匹配")
    dimensions = output.get("dimensions")
    if not isinstance(dimensions, list):
        return [*errors, "dimensions不是数组"]
    dimension_map = {
        str(item.get("dimension_id")): item for item in dimensions if isinstance(item, dict)
    }
    if set(dimension_map) != set(DIMENSION_WEIGHTS):
        errors.append("五个评分维度不完整或重复")
    for dimension_id, item in dimension_map.items():
        status = item.get("status")
        rating = item.get("rating")
        evidence = item.get("evidence")
        if status not in {"EVIDENCED", "CONTRADICTED", "UNKNOWN"}:
            errors.append(f"{dimension_id}状态非法")
            continue
        if status == "UNKNOWN":
            if rating is not None or evidence not in ([], None):
                errors.append(f"{dimension_id}未知状态不得伪造评分或证据")
            continue
        if not isinstance(rating, int) or isinstance(rating, bool) or not 0 <= rating <= 5:
            errors.append(f"{dimension_id}评分必须是0到5整数")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{dimension_id}有评分但没有证据")
            continue
        directions = {
            citation.get("direction") for citation in evidence if isinstance(citation, dict)
        }
        if rating == 0 and "COUNTER" not in directions:
            errors.append(f"{dimension_id}的0分必须有明确反证")
        if isinstance(rating, int) and rating > 0 and "SUPPORT" not in directions:
            errors.append(f"{dimension_id}的正评分必须有支持证据")
        if status == "CONTRADICTED" and "COUNTER" not in directions:
            errors.append(f"{dimension_id}的反证状态缺少反向引用")
        for citation in evidence:
            if not isinstance(citation, dict) or not _citation_is_bound(citation, documents):
                errors.append(f"{dimension_id}引用无法在PDF原文中逐字定位")
    penalties = output.get("penalties")
    if not isinstance(penalties, list):
        return [*errors, "penalties不是数组"]
    penalty_map = {
        str(item.get("penalty_id")): item for item in penalties if isinstance(item, dict)
    }
    if set(penalty_map) != set(PENALTY_IDS):
        errors.append("八个风险扣分维度不完整或重复")
    for penalty_id, item in penalty_map.items():
        status = item.get("status")
        rating = item.get("rating")
        evidence = item.get("evidence")
        if status == "UNKNOWN":
            if rating is not None or evidence not in ([], None):
                errors.append(f"{penalty_id}未知状态不得伪造扣分")
            continue
        if status not in {"EVIDENCED", "CONTRADICTED"}:
            errors.append(f"{penalty_id}状态非法")
            continue
        if not isinstance(rating, int) or isinstance(rating, bool) or not 0 <= rating <= 5:
            errors.append(f"{penalty_id}扣分等级必须是0到5整数")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{penalty_id}有扣分但没有证据")
            continue
        directions = {
            citation.get("direction") for citation in evidence if isinstance(citation, dict)
        }
        if isinstance(rating, int) and rating > 0 and "SUPPORT" not in directions:
            errors.append(f"{penalty_id}的正扣分必须有支持证据")
        if rating == 0 and "COUNTER" not in directions:
            errors.append(f"{penalty_id}的0分必须有明确反证")
        if status == "CONTRADICTED" and "COUNTER" not in directions:
            errors.append(f"{penalty_id}的反证状态缺少反向引用")
        for citation in evidence:
            if not isinstance(citation, dict) or not _citation_is_bound(citation, documents):
                errors.append(f"{penalty_id}引用无法在PDF原文中逐字定位")
    return errors


def compute_full_score(base_36_points: float | None, output: dict[str, Any]) -> dict[str, Any]:
    dimensions = {item["dimension_id"]: item for item in output.get("dimensions", [])}
    missing_dimensions = [
        key
        for key in DIMENSION_WEIGHTS
        if key not in dimensions or dimensions[key].get("status") == "UNKNOWN"
    ]
    pdf_points: float | None = None
    raw_factor_score: float | None = None
    if base_36_points is not None and not missing_dimensions:
        pdf_points = round(
            sum(
                int(dimensions[key]["rating"]) / 5 * weight
                for key, weight in DIMENSION_WEIGHTS.items()
            ),
            2,
        )
        raw_factor_score = round(min(100.0, max(0.0, base_36_points + pdf_points)), 2)

    penalties = {item["penalty_id"]: item for item in output.get("penalties", [])}
    known_penalty_points = round(
        sum(
            int(item["rating"]) * PENALTY_MULTIPLIER
            for item in penalties.values()
            if item.get("status") != "UNKNOWN" and item.get("rating") is not None
        ),
        2,
    )
    unknown_penalties = [
        key
        for key in PENALTY_IDS
        if key not in penalties or penalties[key].get("status") == "UNKNOWN"
    ]
    upper_bound = (
        round(max(0.0, raw_factor_score - known_penalty_points), 2)
        if raw_factor_score is not None
        else None
    )
    exact = upper_bound if not unknown_penalties else None
    if raw_factor_score is None:
        status = "DATA_INSUFFICIENT"
    elif unknown_penalties:
        status = "FACTOR_SCORE_COMPLETE_RISK_INCOMPLETE"
    else:
        status = "COMPLETE"
    return {
        "base_36_points": round(base_36_points, 2) if base_36_points is not None else None,
        "pdf_points": pdf_points,
        "raw_factor_score": raw_factor_score,
        "known_penalty_points": known_penalty_points,
        "risk_adjusted_score_upper_bound": upper_bound,
        "risk_adjusted_score_exact": exact,
        "missing_dimensions": missing_dimensions,
        "unknown_penalties": unknown_penalties,
        "status": status,
    }


def _score_schema() -> dict[str, Any]:
    citation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["document_id", "page_number", "quote", "direction"],
        "properties": {
            "document_id": {"type": "string", "minLength": 1},
            "page_number": {"type": "integer", "minimum": 1},
            "quote": {"type": "string", "minLength": 6},
            "direction": {"enum": ["SUPPORT", "COUNTER"]},
        },
    }
    dimension = {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimension_id", "status", "rating", "reason", "evidence"],
        "properties": {
            "dimension_id": {"enum": list(DIMENSION_WEIGHTS)},
            "status": {"enum": ["EVIDENCED", "CONTRADICTED", "UNKNOWN"]},
            "rating": {"type": ["integer", "null"], "minimum": 0, "maximum": 5},
            "reason": {"type": "string", "minLength": 2},
            "evidence": {"type": "array", "maxItems": 6, "items": citation},
        },
    }
    penalty = {
        "type": "object",
        "additionalProperties": False,
        "required": ["penalty_id", "status", "rating", "reason", "evidence"],
        "properties": {
            "penalty_id": {"enum": list(PENALTY_IDS)},
            "status": {"enum": ["EVIDENCED", "CONTRADICTED", "UNKNOWN"]},
            "rating": {"type": ["integer", "null"], "minimum": 0, "maximum": 5},
            "reason": {"type": "string", "minLength": 2},
            "evidence": {"type": "array", "maxItems": 4, "items": citation},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "entity_id",
            "cutoff",
            "dimensions",
            "penalties",
            "kill_switches",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "entity_id": {"type": "string", "minLength": 8},
            "cutoff": {"const": "T0"},
            "dimensions": {"type": "array", "minItems": 5, "maxItems": 5, "items": dimension},
            "penalties": {"type": "array", "minItems": 8, "maxItems": 8, "items": penalty},
            "kill_switches": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
        },
    }


def _fact_audit_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "reviews"],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "reviews": {
                "type": "array",
                "minItems": FACT_AUDIT_BATCH_SIZE,
                "maxItems": FACT_AUDIT_BATCH_SIZE,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["fact_id", "verdict", "normalized_claim", "quote", "reason"],
                    "properties": {
                        "fact_id": {"type": "string", "minLength": 8},
                        "verdict": {"enum": ["SUPPORTED", "MISCLASSIFIED", "AMBIGUOUS"]},
                        "normalized_claim": {"type": "string"},
                        "quote": {"type": "string", "minLength": 6},
                        "reason": {"type": "string", "minLength": 2},
                    },
                },
            },
        },
    }


def _parse_pages(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^--- PAGE (\d+) ---\n", text))
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages[int(match.group(1))] = text[start:end].strip()
    return pages


def load_score_states(store: PilotStore) -> tuple[list[ScoreState], list[dict[str, Any]]]:
    manifest = store.get_meta("manifest", {})
    decision_dates = [date.fromisoformat(value) for value in manifest.get("decision_dates", [])]
    if len(decision_dates) != 7:
        raise RuntimeError("full64 scoring is bound to the frozen seven decision dates")
    universe = {row["symbol"]: row for row in store.universe()}
    rows = store.connection.execute(
        """
        SELECT a.symbol, a.announcement_id, a.announce_time, a.title, d.sha256
        FROM announcements a JOIN document_metrics d USING (announcement_id)
        WHERE a.status='measured'
        ORDER BY a.symbol, a.announce_time, a.announcement_id
        """
    ).fetchall()
    by_symbol: dict[str, list[DocumentPacket]] = defaultdict(list)
    for symbol, announcement_id, announce_time, title, sha256 in rows:
        if announce_time is None:
            continue
        text_path = store.text_dir / f"{announcement_id}.txt"
        if not text_path.is_file():
            raise RuntimeError(f"measured document text is missing: {announcement_id}")
        by_symbol[str(symbol)].append(
            DocumentPacket(
                document_id=str(announcement_id),
                announcement_time=announce_time,
                title=str(title),
                pages=_parse_pages(text_path),
                sha256=str(sha256),
            )
        )

    states: list[ScoreState] = []
    blocked: list[dict[str, Any]] = []
    for symbol, company in sorted(universe.items()):
        documents = by_symbol.get(symbol, [])
        prior_ids: tuple[str, ...] = ()
        if not documents:
            blocked.append({"symbol": symbol, "status": "DATA_INSUFFICIENT_NO_PDF"})
            continue
        for cutoff in decision_dates:
            cutoff_at = datetime.combine(cutoff, datetime_time(21, 30))
            visible = tuple(doc for doc in documents if doc.announcement_time <= cutoff_at)
            visible_ids = tuple(doc.document_id for doc in visible)
            if not visible or visible_ids == prior_ids:
                continue
            entity_id = "ENTITY-" + _stable_hash(REPLAY_ID, symbol, cutoff.isoformat())[:20]
            state = ScoreState(
                symbol=symbol,
                name=str(company["name"]),
                chain_id=str(company["chain_id"]),
                cutoff_date=cutoff,
                entity_id=entity_id,
                documents=visible,
            )
            prompt = build_score_prompt(state)
            if len(prompt.encode("utf-8")) > MAX_SCORE_REQUEST_BYTES - 30_000:
                blocked.append(
                    {
                        "symbol": symbol,
                        "cutoff_date": cutoff.isoformat(),
                        "status": "DATA_INSUFFICIENT_OVERSIZE_CONTEXT",
                        "prompt_bytes": len(prompt.encode("utf-8")),
                    }
                )
            else:
                states.append(state)
            prior_ids = visible_ids
    return states, blocked


def build_score_prompt(state: ScoreState) -> str:
    policy = {
        "task": "仅依据以下T0前已经公开的PDF原文，评定五个维度和八项风险；不得使用外部知识",
        "rules": [
            "只引用给定PDF中的逐字原句；页码和document_id必须准确",
            "原文没有覆盖时必须输出UNKNOWN和null，不得把缺失当0分",
            "0分也必须有明确反向证据，不能由沉默推断",
            "发行人自身PDF通常不能构成独立交叉验证，证据质量据此受限",
            "评分只输出0到5整数；分值换算由本地确定性程序完成",
            "区分支持证据和反证，遇到冲突优先降低评级或标CONTRADICTED",
        ],
        "rating_anchors": RATING_ANCHORS,
        "penalty_guidance": PENALTY_GUIDANCE,
    }
    parts = [
        "这是一个不连接券商的历史研究样本。公司身份仅用于同一PDF集合归并。",
        "输出必须是满足Schema的单个JSON对象。",
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


def _sample_facts(store: PilotStore) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        """
        SELECT f.fact_id, f.category, f.evidence_sentence, f.page_number,
               a.announcement_id, u.chain_id
        FROM evidence_facts f
        JOIN announcements a USING (announcement_id)
        JOIN universe u USING (symbol)
        ORDER BY f.category, u.chain_id, f.fact_id
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact_id, category, sentence, page_number, document_id, chain_id in rows:
        text_path = store.text_dir / f"{document_id}.txt"
        pages = _parse_pages(text_path) if text_path.is_file() else {}
        page_text = pages.get(int(page_number), "")
        if _normalize_text(str(sentence)) not in _normalize_text(page_text):
            raise RuntimeError(f"fact evidence is not bound to its PDF page: {fact_id}")
        key = (str(category), str(chain_id))
        grouped[key].append(
            {
                "fact_id": str(fact_id),
                "category": str(category),
                "sentence": str(sentence),
                "page_number": int(page_number),
                "document_id": str(document_id),
                "chain_id": str(chain_id),
            }
        )
    selected: list[dict[str, Any]] = []
    strata = sorted(grouped)
    cursor = {key: 0 for key in strata}
    while len(selected) < FACT_AUDIT_SAMPLE_SIZE:
        progressed = False
        for stratum in strata:
            values = grouped[stratum]
            index = cursor[stratum]
            if index >= len(values):
                continue
            selected.append(values[index])
            cursor[stratum] += 1
            progressed = True
            if len(selected) == FACT_AUDIT_SAMPLE_SIZE:
                break
        if not progressed:
            break
    if len(selected) != FACT_AUDIT_SAMPLE_SIZE:
        raise RuntimeError("fact audit cannot form the frozen 50-fact sample")
    return selected


def _fact_audit_prompt(batch: list[dict[str, Any]]) -> str:
    return (
        "逐条判断规则抽取的句子是否真实支持其category。只能依据句子本身；不得补充外部事实。"
        "SUPPORTED表示句子直接支持，MISCLASSIFIED表示类别错误，AMBIGUOUS表示语义不足。"
        "quote必须逐字复制candidate_sentence，fact_id必须原样返回。\n"
        + json.dumps(
            [
                {
                    "fact_id": row["fact_id"],
                    "category": row["category"],
                    "candidate_sentence": row["sentence"],
                }
                for row in batch
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _runtime_root() -> Path:
    raw = os.environ.get("SERENITY_ASHARE_RUNTIME_ROOT", "").strip()
    if raw:
        return Path(raw).resolve()
    return (settings.data_dir / "research_runtime" / "ashare-cio").resolve()


def _runtime_inventory() -> dict[str, str]:
    runtime_root = _runtime_root()
    inventory: dict[str, str] = {}
    for relative in REQUIRED_RUNTIME_FILES:
        path = runtime_root / relative
        if not path.is_file():
            raise RuntimeError(f"approved DeepSeek runtime file is missing: {path}")
        inventory[relative] = _file_hash(path)
    return inventory


def _runtime_scripts() -> Path:
    return _runtime_root() / "recommend-ashare-next-day" / "scripts"


def _semantic_input_snapshot(store: PilotStore) -> dict[str, Any]:
    universe_rows = store.connection.execute("SELECT * FROM universe ORDER BY symbol").fetchall()
    document_rows = store.connection.execute(
        """
        SELECT a.announcement_id, a.symbol, a.announce_time, a.title, d.sha256
        FROM announcements a JOIN document_metrics d USING (announcement_id)
        WHERE a.status='measured'
        ORDER BY a.announcement_id
        """
    ).fetchall()
    fact_rows = store.connection.execute(
        """
        SELECT fact_id, announcement_id, page_number, category, evidence_sentence
        FROM evidence_facts ORDER BY fact_id
        """
    ).fetchall()
    decision_rows = store.connection.execute(
        """
        SELECT decision_date, symbol, score, input_hash
        FROM decisions WHERE model='serenity'
        ORDER BY decision_date, symbol
        """
    ).fetchall()
    payload = {
        "universe": [
            [str(value) if value is not None else None for value in row] for row in universe_rows
        ],
        "documents": [
            [str(value) if value is not None else None for value in row] for row in document_rows
        ],
        "facts": [
            [str(value) if value is not None else None for value in row] for row in fact_rows
        ],
        "base_decisions": [
            [str(value) if value is not None else None for value in row] for row in decision_rows
        ],
    }
    return {
        "universe_count": len(universe_rows),
        "document_count": len(document_rows),
        "fact_count": len(fact_rows),
        "base_decision_count": len(decision_rows),
        "content_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }


def _run_root(store: PilotStore) -> Path:
    return store.root / "full64"


def _write_run_contract(store: PilotStore) -> dict[str, Path]:
    run_root = _run_root(store)
    run_root.mkdir(parents=True, exist_ok=True)
    score_schema = run_root / "serenity-pdf-score.schema.json"
    audit_schema = run_root / "serenity-fact-audit.schema.json"
    policy_path = run_root / "model-execution-policy.json"
    authorization_path = run_root / "model-budget-authorization.json"
    state_path = run_root / "model-budget-state.json"
    _freeze_json(score_schema, _score_schema(), "score schema")
    _freeze_json(audit_schema, _fact_audit_schema(), "fact-audit schema")
    policy = {
        "schema_version": "1.0.0",
        "policy_id": "SERENITY-PDF-FULL64-DEEPSEEK-V4-FLASH-V1",
        "provider": PROVIDER,
        "base_url": "https://api.deepseek.com/chat/completions",
        "model": MODEL,
        "paid_execution_default": "DISABLED_WITHOUT_ASSET_OWNER_BUDGET",
        "network_execution_status": "ACTIVE_EXPLICIT_ASSET_OWNER_AUTHORIZATION",
        "thinking": "enabled",
        "http_timeout_seconds": 1100,
        "semantic_worker_limit": 1,
        "max_attempts_per_context_cap": 2,
        "stage_profiles": {
            FACT_AUDIT_STAGE: {
                "thinking_type": "disabled",
                "reasoning_effort": "high",
                "max_output_tokens": 4096,
                "max_request_bytes": 120000,
                "max_attempts_per_context": 2,
                "max_retry_contexts": 2,
            },
            SCORE_STAGE: {
                "thinking_type": "enabled",
                "reasoning_effort": "high",
                "max_output_tokens": 8192,
                "max_request_bytes": MAX_SCORE_REQUEST_BYTES,
                "max_attempts_per_context": 2,
                "max_retry_contexts": 4,
            },
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
    _freeze_json(policy_path, policy, "model execution policy")
    policy_hash = _file_hash(policy_path)
    manifest = store.get_meta("manifest", {})
    decision_dates = list(manifest.get("decision_dates", []))
    if len(decision_dates) != 7:
        raise RuntimeError("budget cannot bind a non-seven-day pilot")
    input_snapshot = _semantic_input_snapshot(store)
    if input_snapshot["universe_count"] != 100:
        raise RuntimeError("full64 scoring requires the frozen 100-company sample")
    if input_snapshot["document_count"] != 360:
        raise RuntimeError("full64 scoring requires the frozen 360 measured PDFs")
    if input_snapshot["fact_count"] < FACT_AUDIT_SAMPLE_SIZE:
        raise RuntimeError("full64 scoring has too few PDF facts for the frozen audit")
    authorization = _with_content_hash(
        {
            "schema_version": "1.0.0",
            "budget_id": "SERENITY-FULL64-CNY120-V1",
            "authorization_status": "APPROVED",
            "authorized_by": "ASSET_OWNER",
            "approval_evidence": "用户在Codex任务中明确回复确认120元上限",
            "replay_id": REPLAY_ID,
            "allowed_trade_dates": decision_dates,
            "provider": PROVIDER,
            "model": MODEL,
            "policy_sha256": policy_hash,
            "effective_at": "2026-08-27T00:00:00+08:00",
            "expires_at": "2026-08-29T23:59:00+08:00",
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
                "evidence": "https://api-docs.deepseek.com/quick_start/pricing; peak USD price converted at conservative 8 CNY/USD cap basis",
            },
        }
    )
    _freeze_json(authorization_path, authorization, "model budget authorization")
    run_manifest = _with_content_hash(
        {
            "schema_version": SCORING_VERSION,
            "replay_id": REPLAY_ID,
            "pilot_id": manifest.get("pilot_id"),
            "sample_hash": manifest.get("sample_hash"),
            "decision_dates": decision_dates,
            "documents_expected": 360,
            "fact_audit_sample_size": FACT_AUDIT_SAMPLE_SIZE,
            "fact_audit_pass_rate": FACT_AUDIT_PASS_RATE,
            "dimension_weights": DIMENSION_WEIGHTS,
            "penalty_multiplier": PENALTY_MULTIPLIER,
            "selection_threshold": SELECTION_THRESHOLD,
            "claim_boundary": "RETROSPECTIVE_ENGINEERING_SAMPLE_NOT_CLEAN_ROOM_UNVERIFIED_ALPHA",
            "model": MODEL,
            "semantic_input_snapshot": input_snapshot,
            "runtime_sha256": _runtime_inventory(),
            "policy_sha256": policy_hash,
            "authorization_sha256": _file_hash(authorization_path),
        }
    )
    manifest_path = run_root / "run-manifest.json"
    _freeze_json(manifest_path, run_manifest, "full64 run manifest")
    return {
        "run_root": run_root,
        "score_schema": score_schema,
        "audit_schema": audit_schema,
        "policy": policy_path,
        "authorization": authorization_path,
        "state": state_path,
        "manifest": manifest_path,
    }


def _usage_path(paths: dict[str, Path], trade_date: date) -> Path:
    return paths["run_root"] / f"model-usage-{trade_date.isoformat()}.jsonl"


def _verified_output_hash(raw_output: str, ledger_row: dict[str, Any]) -> str:
    actual = hashlib.sha256(raw_output.encode()).hexdigest()
    if actual != ledger_row.get("output_sha256"):
        raise RuntimeError("DeepSeek adapter output does not match its immutable receipt")
    return actual


def _budget_environment(paths: dict[str, Path], spec: ModelCallSpec) -> dict[str, str]:
    if current_ai_provider() != "openai_compat" or current_ai_model() != MODEL:
        raise RuntimeError("server AI configuration must use openai_compat/deepseek-v4-flash")
    base_url = secrets_store.get_ai_config("ai_base_url", "").rstrip("/")
    if base_url not in {OFFICIAL_BASE_URL, OFFICIAL_BASE_URL + "/v1"}:
        raise RuntimeError("DeepSeek must use the official api.deepseek.com endpoint")
    env = dict(os.environ)
    env.update(
        {
            "ASHARE_LLM_BUDGET_AUTHORIZATION": str(paths["authorization"]),
            "ASHARE_LLM_BUDGET_POLICY": str(paths["policy"]),
            "ASHARE_LLM_REPLAY_ID": spec.replay_id,
            "ASHARE_LLM_TRADE_DATE": spec.trade_date.isoformat(),
            "ASHARE_LLM_STAGE": spec.stage,
            "ASHARE_LLM_CALL_SLOT_ID": spec.slot_id,
            "ASHARE_LLM_USAGE_LOG": str(_usage_path(paths, spec.trade_date)),
            "DEEPSEEK_HTTP_TIMEOUT_SECONDS": "1100",
        }
    )
    return env


def _adapter_environment(paths: dict[str, Path], spec: ModelCallSpec) -> dict[str, str]:
    key = secrets_store.get_ai_key().strip()
    if not key:
        raise RuntimeError("DeepSeek credential is not configured")
    env = _budget_environment(paths, spec)
    env["DEEPSEEK_API_KEY"] = key
    return env


def _adapter_runner(paths: dict[str, Path]) -> Callable[[ModelCallSpec], CachedModelResult]:
    adapter = _runtime_scripts() / "deepseek-codex-adapter.py"
    if not adapter.is_file():
        raise RuntimeError(f"approved DeepSeek adapter is missing: {adapter}")

    def run(spec: ModelCallSpec) -> CachedModelResult:
        outputs = paths["run_root"] / "adapter-outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        output_path = outputs / f"{spec.slot_id}-{spec.input_sha256}.json"
        completed = subprocess.run(
            [
                os.environ.get("PYTHON", sys.executable),
                str(adapter),
                "exec",
                "--json",
                "--output-schema",
                str(spec.schema_path),
                "--output-last-message",
                str(output_path),
            ],
            input=spec.prompt,
            text=True,
            capture_output=True,
            env=_adapter_environment(paths, spec),
            timeout=1200,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"DeepSeek adapter failed closed: {completed.stderr.strip()[:500]}")
        if not output_path.is_file():
            raise RuntimeError("DeepSeek adapter did not persist its structured output")
        rows = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.strip().startswith("{")
        ]
        turn = next((row for row in rows if row.get("type") == "turn.completed"), None)
        thread = next((row for row in rows if row.get("type") == "thread.started"), None)
        if not turn or not thread:
            raise RuntimeError("DeepSeek adapter audit stream is incomplete")
        raw_output = output_path.read_text(encoding="utf-8")
        json.loads(raw_output)
        ledger_row: dict[str, Any] | None = None
        usage_path = _usage_path(paths, spec.trade_date)
        if usage_path.is_file():
            for line in usage_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                candidate = json.loads(line)
                if candidate.get("status") == "PASS" and candidate.get(
                    "reservation_id"
                ) == turn.get("reservation_id"):
                    ledger_row = candidate
                    break
        if ledger_row is None:
            raise RuntimeError("DeepSeek paid call is missing from the immutable usage ledger")
        raw_output_sha256 = _verified_output_hash(raw_output, ledger_row)
        return CachedModelResult(
            raw_output=raw_output,
            response_sha256=str(ledger_row["response_sha256"]),
            output_sha256=raw_output_sha256,
            context_id=str(thread["thread_id"]),
            api_request_id=ledger_row.get("api_request_id"),
            system_fingerprint=ledger_row.get("system_fingerprint"),
            finish_reason=str(turn["finish_reason"]),
            usage={key: int(value) for key, value in turn["usage"].items()},
            cost_micros_cny=int(turn["actual_cost_micros_cny"]),
            adapter_request_sha256=str(turn["request_sha256"]),
        )

    return run


def _preflight_stage(
    paths: dict[str, Path], *, stage: str, trade_date: date, prompts: list[str], schema: Path
) -> dict[str, Any]:
    runtime_scripts = _runtime_scripts()
    helper = (
        "import json,os,sys; from pathlib import Path; "
        "sys.path.insert(0, os.environ['RUNTIME_SCRIPTS']); "
        "from deepseek_payload import build_request_payload; "
        "from model_budget import preflight_stage,stage_profile; "
        "schema=json.loads(Path(os.environ['SCHEMA_PATH']).read_text()); "
        "prompts=json.loads(Path(os.environ['PROMPTS_PATH']).read_text()); "
        "profile=stage_profile(os.environ['ASHARE_LLM_STAGE']); "
        "thinking=profile.get('thinking_type','enabled'); "
        "total=sum(len(json.dumps(build_request_payload(p,schema,profile,thinking_type=thinking),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()) for p in prompts); "
        "print(json.dumps(preflight_stage(stage=os.environ['ASHARE_LLM_STAGE'],context_count=len(prompts),request_byte_count_total=total),sort_keys=True))"
    )
    prompts_path = paths["run_root"] / f"{stage.lower()}-pending-prompts.json"
    _atomic_json(prompts_path, prompts)
    dummy_spec = ModelCallSpec(REPLAY_ID, stage, "SLOT-PREFLIGHT-ONLY", trade_date, "x", schema)
    env = _budget_environment(paths, dummy_spec)
    env.update(
        {
            "RUNTIME_SCRIPTS": str(runtime_scripts),
            "SCHEMA_PATH": str(schema),
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
        raise RuntimeError(
            f"whole-stage paid budget preflight failed: {completed.stderr.strip()[:500]}"
        )
    return json.loads(completed.stdout)


def audit_budget_ledgers(paths: dict[str, Path]) -> dict[str, Any]:
    """Reconcile every per-date model ledger against the cumulative budget state."""
    if not paths["state"].is_file():
        return {"status": "NOT_STARTED", "reports": []}
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    trade_dates = sorted(
        {
            str(row.get("trade_date"))
            for row in state.get("reservations", [])
            if row.get("trade_date")
        }
    )
    runtime_script = _runtime_scripts() / "audit-model-budget.py"
    reports: list[dict[str, Any]] = []
    for trade_date_value in trade_dates:
        trade_date = date.fromisoformat(trade_date_value)
        output_path = paths["run_root"] / f"budget-audit-{trade_date_value}.json"
        spec = ModelCallSpec(
            REPLAY_ID,
            SCORE_STAGE,
            "SLOT-BUDGET-AUDIT",
            trade_date,
            "budget-audit",
            paths["score_schema"],
        )
        completed = subprocess.run(
            [
                os.environ.get("PYTHON", sys.executable),
                str(runtime_script),
                "--authorization",
                str(paths["authorization"]),
                "--usage-ledger",
                str(_usage_path(paths, trade_date)),
                "--output",
                str(output_path),
            ],
            text=True,
            capture_output=True,
            env=_budget_environment(paths, spec),
            timeout=60,
            check=False,
        )
        if not output_path.is_file():
            raise RuntimeError(
                f"budget audit did not create an output for {trade_date_value}: "
                f"{completed.stderr.strip()[:300]}"
            )
        report = json.loads(output_path.read_text(encoding="utf-8"))
        reports.append(report)
        if completed.returncode != 0 or report.get("status") != "PASS":
            raise RuntimeError(
                f"model budget audit failed for {trade_date_value}: "
                f"{'; '.join(report.get('errors') or [])}"
            )
    return {"status": "PASS", "reports": reports}


def run_fact_audit(store: PilotStore, paths: dict[str, Path]) -> dict[str, Any]:
    initialize_semantic_tables(store.connection)
    sample = _sample_facts(store)
    batches = [
        sample[index : index + FACT_AUDIT_BATCH_SIZE]
        for index in range(0, len(sample), FACT_AUDIT_BATCH_SIZE)
    ]
    specs = [
        ModelCallSpec(
            REPLAY_ID,
            FACT_AUDIT_STAGE,
            f"SLOT-FACT-AUDIT-{index:02d}",
            date(2026, 8, 26),
            _fact_audit_prompt(batch),
            paths["audit_schema"],
            _file_hash(paths["policy"]),
        )
        for index, batch in enumerate(batches, start=1)
    ]
    pending = [
        spec
        for spec in specs
        if not store.connection.execute(
            "SELECT 1 FROM semantic_model_calls WHERE replay_id=? AND stage=? AND slot_id=? AND input_sha256=? AND status='PASS'",
            [spec.replay_id, spec.stage, spec.slot_id, spec.input_sha256],
        ).fetchone()
    ]
    if pending:
        _preflight_stage(
            paths,
            stage=FACT_AUDIT_STAGE,
            trade_date=date(2026, 8, 26),
            prompts=[spec.prompt for spec in pending],
            schema=paths["audit_schema"],
        )
    runner = _adapter_runner(paths)
    for spec, batch in zip(specs, batches, strict=True):
        result = execute_cached_call(store.connection, spec, runner)
        output = json.loads(result.raw_output)
        reviews = output.get("reviews")
        expected = {row["fact_id"]: row for row in batch}
        if not isinstance(reviews, list) or {
            row.get("fact_id") for row in reviews if isinstance(row, dict)
        } != set(expected):
            raise RuntimeError(f"fact audit output does not cover its exact batch: {spec.slot_id}")
        values = []
        for review in reviews:
            source = expected[review["fact_id"]]
            quote = str(review.get("quote") or "")
            if _normalize_text(quote) != _normalize_text(source["sentence"]):
                raise RuntimeError(
                    f"fact audit quote is not exact source text: {review['fact_id']}"
                )
            values.append(
                [
                    REPLAY_ID,
                    review["fact_id"],
                    spec.slot_id,
                    review["verdict"],
                    review.get("normalized_claim", ""),
                    quote,
                    review["reason"],
                    spec.input_sha256,
                    datetime.now(),
                ]
            )
        store.connection.executemany(
            "INSERT OR REPLACE INTO semantic_fact_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
    counts = dict(
        store.connection.execute(
            "SELECT verdict, count(*) FROM semantic_fact_audit WHERE replay_id=? GROUP BY verdict",
            [REPLAY_ID],
        ).fetchall()
    )
    total = sum(int(value) for value in counts.values())
    supported = int(counts.get("SUPPORTED", 0))
    pass_rate = supported / total if total else 0.0
    report = {
        "sample_size": total,
        "verdict_counts": counts,
        "supported_rate": pass_rate,
        "threshold": FACT_AUDIT_PASS_RATE,
        "status": "PASS"
        if total == FACT_AUDIT_SAMPLE_SIZE and pass_rate >= FACT_AUDIT_PASS_RATE
        else "FAIL",
        "claim_boundary": "MODEL_GROUNDED_FACT_AUDIT_NOT_HUMAN_GROUND_TRUTH",
    }
    _atomic_json(paths["run_root"] / "fact-audit-report.json", report)
    return report


def _document_lookup(state: ScoreState) -> dict[str, dict[int, str]]:
    return {document.document_id: document.pages for document in state.documents}


def _base_score(store: PilotStore, symbol: str, cutoff: date) -> float | None:
    row = store.connection.execute(
        "SELECT score FROM decisions WHERE decision_date=? AND symbol=? AND model='serenity'",
        [cutoff, symbol],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def run_scores(store: PilotStore, paths: dict[str, Path]) -> dict[str, Any]:
    audit = json.loads((paths["run_root"] / "fact-audit-report.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("fact audit did not pass; paid scoring is stopped")
    states, blocked = load_score_states(store)
    policy_hash = _file_hash(paths["policy"])
    specs = [
        ModelCallSpec(
            REPLAY_ID,
            SCORE_STAGE,
            "SLOT-SCORE-"
            + _stable_hash(REPLAY_ID, state.symbol, state.cutoff_date.isoformat())[:28],
            state.cutoff_date,
            build_score_prompt(state),
            paths["score_schema"],
            policy_hash,
        )
        for state in states
    ]
    pending = [
        spec
        for spec in specs
        if not store.connection.execute(
            "SELECT 1 FROM semantic_model_calls WHERE replay_id=? AND stage=? AND slot_id=? AND input_sha256=? AND status='PASS'",
            [spec.replay_id, spec.stage, spec.slot_id, spec.input_sha256],
        ).fetchone()
    ]
    if pending:
        _preflight_stage(
            paths,
            stage=SCORE_STAGE,
            trade_date=pending[0].trade_date,
            prompts=[spec.prompt for spec in pending],
            schema=paths["score_schema"],
        )
    runner = _adapter_runner(paths)
    for index, (state, spec) in enumerate(zip(states, specs, strict=True), start=1):
        result = execute_cached_call(store.connection, spec, runner)
        output = json.loads(result.raw_output)
        errors = validate_score_output(
            output, entity_id=state.entity_id, documents=_document_lookup(state)
        )
        if errors:
            raise RuntimeError(
                f"local PDF evidence validation failed for {state.symbol}/{state.cutoff_date}: {'; '.join(errors[:5])}"
            )
        score = compute_full_score(_base_score(store, state.symbol, state.cutoff_date), output)
        store.connection.execute(
            """
            INSERT OR REPLACE INTO semantic_score_results VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                REPLAY_ID,
                state.symbol,
                state.cutoff_date,
                state.entity_id,
                score["status"],
                score["base_36_points"],
                score["pdf_points"],
                score["raw_factor_score"],
                score["known_penalty_points"],
                score["risk_adjusted_score_upper_bound"],
                score["risk_adjusted_score_exact"],
                json.dumps(output["dimensions"], ensure_ascii=False, sort_keys=True),
                json.dumps(output["penalties"], ensure_ascii=False, sort_keys=True),
                json.dumps(output.get("kill_switches", []), ensure_ascii=False),
                result.raw_output,
                spec.input_sha256,
                datetime.now(),
            ],
        )
        print(
            json.dumps(
                {
                    "event": "serenity_full64_progress",
                    "completed": index,
                    "total": len(states),
                    "symbol": state.symbol,
                    "cutoff": state.cutoff_date.isoformat(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    report = {
        "score_contexts": len(states),
        "blocked_contexts": blocked,
        "scored_rows": store.connection.execute(
            "SELECT count(*) FROM semantic_score_results WHERE replay_id=?", [REPLAY_ID]
        ).fetchone()[0],
    }
    _atomic_json(paths["run_root"] / "score-stage-report.json", report)
    return report


def materialize_decisions_and_outcomes(store: PilotStore, paths: dict[str, Path]) -> dict[str, Any]:
    manifest = store.get_meta("manifest", {})
    decision_dates = [date.fromisoformat(value) for value in manifest.get("decision_dates", [])]
    universe = store.universe()
    now = datetime.now()
    for decision_date in decision_dates:
        for company in universe:
            symbol = company["symbol"]
            score_row = store.connection.execute(
                """
                SELECT cutoff_date, dimension_json, penalty_json
                FROM semantic_score_results
                WHERE replay_id=? AND symbol=? AND cutoff_date<=?
                ORDER BY cutoff_date DESC LIMIT 1
                """,
                [REPLAY_ID, symbol, decision_date],
            ).fetchone()
            if score_row:
                current = compute_full_score(
                    _base_score(store, symbol, decision_date),
                    {
                        "dimensions": json.loads(score_row[1]),
                        "penalties": json.loads(score_row[2]),
                    },
                )
                research_score = current["risk_adjusted_score_exact"]
                status = str(current["status"])
                selected = bool(
                    research_score is not None
                    and float(research_score) >= SELECTION_THRESHOLD
                    and status == "COMPLETE"
                )
                values = [
                    REPLAY_ID,
                    decision_date,
                    symbol,
                    company["chain_id"],
                    score_row[0],
                    current["base_36_points"],
                    current["pdf_points"],
                    current["raw_factor_score"],
                    current["known_penalty_points"],
                    research_score,
                    status,
                    selected,
                    False,
                    now,
                ]
            else:
                values = [
                    REPLAY_ID,
                    decision_date,
                    symbol,
                    company["chain_id"],
                    None,
                    _base_score(store, symbol, decision_date),
                    None,
                    None,
                    0.0,
                    None,
                    "DATA_INSUFFICIENT_NO_PDF",
                    False,
                    False,
                    now,
                ]
            store.connection.execute(
                "INSERT OR REPLACE INTO serenity_full_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    selected_rows = store.connection.execute(
        "SELECT decision_date, symbol FROM serenity_full_decisions WHERE replay_id=? AND research_selected ORDER BY 1,2",
        [REPLAY_ID],
    ).fetchall()
    market = (
        _load_market_rows(settings.data_dir, {row[1] for row in selected_rows})
        if selected_rows
        else {}
    )
    index_rows = _load_index_rows(settings.data_dir)
    index_by_date = {row["date"]: row for row in index_rows}
    for decision_date, symbol in selected_rows:
        future = [row for row in market.get(symbol, []) if row["date"] > decision_date]
        for horizon in HORIZONS:
            if len(future) < horizon:
                store.connection.execute(
                    "INSERT OR REPLACE INTO serenity_full_outcomes VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, 'pending', NULL)",
                    [REPLAY_ID, decision_date, symbol, horizon],
                )
                continue
            window = future[:horizon]
            entry = window[0]
            exit_row = window[-1]
            entry_price = float(entry["open"])
            net_return = float(exit_row["close"]) / entry_price - 1 - DEFAULT_COST_BPS / 10000
            benchmark = None
            index_entry = index_by_date.get(entry["date"])
            index_exit = index_by_date.get(exit_row["date"])
            if index_entry and index_exit and float(index_entry["open"]):
                benchmark = float(index_exit["close"]) / float(index_entry["open"]) - 1
            store.connection.execute(
                "INSERT OR REPLACE INTO serenity_full_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'settled', ?)",
                [
                    REPLAY_ID,
                    decision_date,
                    symbol,
                    horizon,
                    entry["date"],
                    exit_row["date"],
                    net_return,
                    benchmark,
                    min(float(row["low"]) / entry_price - 1 for row in window),
                    max(float(row["high"]) / entry_price - 1 for row in window),
                    datetime.now(),
                ],
            )
    summary_rows = store.connection.execute(
        """
        SELECT horizon, count(*), avg(net_return), avg(net_return-benchmark_return), min(mae), max(mfe)
        FROM serenity_full_outcomes WHERE replay_id=? AND status='settled'
        GROUP BY horizon ORDER BY horizon
        """,
        [REPLAY_ID],
    ).fetchall()
    report = {
        "replay_id": REPLAY_ID,
        "decision_rows": store.connection.execute(
            "SELECT count(*) FROM serenity_full_decisions WHERE replay_id=?", [REPLAY_ID]
        ).fetchone()[0],
        "research_selected_positions": len(selected_rows),
        "risk_incomplete_decisions": store.connection.execute(
            "SELECT count(*) FROM serenity_full_decisions WHERE replay_id=? AND status='FACTOR_SCORE_COMPLETE_RISK_INCOMPLETE'",
            [REPLAY_ID],
        ).fetchone()[0],
        "capital_authorized": False,
        "outcomes": [
            {
                "horizon": row[0],
                "positions": row[1],
                "mean_net_return": row[2],
                "mean_alpha_vs_csi300": row[3],
                "worst_mae": row[4],
                "best_mfe": row[5],
            }
            for row in summary_rows
        ],
        "alpha_status": "UNVERIFIED_ALPHA",
        "claim_boundary": "RETROSPECTIVE_ENGINEERING_SAMPLE_NOT_CLEAN_ROOM",
    }
    _atomic_json(paths["run_root"] / "final-report.json", report)
    return report


def status(store: PilotStore, paths: dict[str, Path]) -> dict[str, Any]:
    initialize_semantic_tables(store.connection)
    calls = store.connection.execute(
        """
        SELECT stage, count(*), coalesce(sum(cost_micros_cny),0),
               coalesce(sum(cast(json_extract(usage_json, '$.prompt_tokens') AS BIGINT)),0),
               coalesce(sum(cast(json_extract(usage_json, '$.completion_tokens') AS BIGINT)),0)
        FROM semantic_model_calls WHERE replay_id=? AND status='PASS' GROUP BY stage ORDER BY stage
        """,
        [REPLAY_ID],
    ).fetchall()
    result = {
        "replay_id": REPLAY_ID,
        "calls": [
            {
                "stage": row[0],
                "requests": row[1],
                "cost_cny": row[2] / 1_000_000,
                "prompt_tokens": row[3],
                "completion_tokens": row[4],
            }
            for row in calls
        ],
        "audit": json.loads(
            (paths["run_root"] / "fact-audit-report.json").read_text(encoding="utf-8")
        )
        if (paths["run_root"] / "fact-audit-report.json").is_file()
        else None,
        "score_rows": store.connection.execute(
            "SELECT count(*) FROM semantic_score_results WHERE replay_id=?", [REPLAY_ID]
        ).fetchone()[0],
        "decision_rows": store.connection.execute(
            "SELECT count(*) FROM serenity_full_decisions WHERE replay_id=?", [REPLAY_ID]
        ).fetchone()[0],
        "budget_audits": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(paths["run_root"].glob("budget-audit-*.json"))
        ],
        "final_report": json.loads(
            (paths["run_root"] / "final-report.json").read_text(encoding="utf-8")
        )
        if (paths["run_root"] / "final-report.json").is_file()
        else None,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "audit", "score", "audit-budget", "materialize", "run", "status"),
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    store = PilotStore(args.root.resolve())
    try:
        initialize_semantic_tables(store.connection)
        paths = _write_run_contract(store)
        if args.command == "prepare":
            payload: dict[str, Any] = {
                "status": "prepared",
                "paths": {key: str(value) for key, value in paths.items()},
            }
        elif args.command == "audit":
            payload = run_fact_audit(store, paths)
        elif args.command == "score":
            payload = run_scores(store, paths)
        elif args.command == "audit-budget":
            payload = audit_budget_ledgers(paths)
        elif args.command == "materialize":
            payload = materialize_decisions_and_outcomes(store, paths)
        elif args.command == "run":
            try:
                payload = {
                    "audit": run_fact_audit(store, paths),
                    "scores": run_scores(store, paths),
                    "budget": audit_budget_ledgers(paths),
                    "result": materialize_decisions_and_outcomes(store, paths),
                }
            finally:
                audit_budget_ledgers(paths)
        else:
            payload = status(store, paths)
    finally:
        store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
