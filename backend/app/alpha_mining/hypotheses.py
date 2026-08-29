"""Durable, falsifiable research hypotheses for the Alpha workbench."""
# ruff: noqa: RUF001
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.backtest.factor import FACTOR_COLUMNS

_LOCK = threading.RLock()
_FACTOR_IDS = frozenset(str(row["id"]) for row in FACTOR_COLUMNS)
_SOURCE_KINDS = frozenset({"prior", "ai", "manual", "failure"})
_DIRECTIONS = frozenset({-1, 1})


def system_hypotheses() -> list[dict[str, Any]]:
    """Return versioned A-share priors; they are proposals, never result claims."""
    rows = [
        {
            "hypothesis_id": "ah-system-lottery-reversal-v1",
            "version": "1.0.0",
            "source_kind": "prior",
            "title": "彩票偏好高估后的横截面反转",
            "thesis": "近期出现极端单日上涨、右偏收益和异常换手的股票，未来10日净收益更低。",
            "mechanism": "A股个人投资者偏好小概率暴涨标的，注意力与追涨需求可能把彩票型股票推离可持续价值。",
            "prediction_object": "forward_net_return",
            "asset_type": "stock",
            "forward_horizon": 10,
            "information_domains": ["price_volume", "liquidity", "behavior"],
            "test_spec": {
                "engine_ids": ["cross_sectional_rank"],
                "factor_names": ["max_ret_20d", "ret_skew_20d", "turnover_z_60d"],
                "expected_directions": {
                    "max_ret_20d": -1,
                    "ret_skew_20d": -1,
                    "turnover_z_60d": -1,
                },
                "weights": {
                    "max_ret_20d": 0.4,
                    "ret_skew_20d": 0.3,
                    "turnover_z_60d": 0.3,
                },
                "parameters": {"entry_score": 75.0, "exit_score": 40.0, "top_rank": 20},
            },
            "falsification": [
                "训练窗预注册组合的逐日截面IC低于0.02或方向一致率低于55%",
                "独立样本外净收益不为正，或夏普、回撤、多窗口、成本压力任一硬门槛失败",
            ],
            "data_requirements": [
                "daily_enriched",
                "historical_universe",
                "share_history_pit",
            ],
        },
        {
            "hypothesis_id": "ah-system-selling-exhaustion-v1",
            "version": "1.0.0",
            "source_kind": "prior",
            "title": "缩量止跌后的卖压衰竭",
            "thesis": "短期跌幅较大、成交量收缩但收盘位置改善的股票，未来5日净收益更高。",
            "mechanism": "T+1和散户集中止损会造成阶段性卖压；量能收缩且尾盘承接改善时，边际卖盘可能已经衰竭。",
            "prediction_object": "forward_net_return",
            "asset_type": "stock",
            "forward_horizon": 5,
            "information_domains": ["price_volume", "liquidity", "microstructure"],
            "test_spec": {
                "engine_ids": ["cross_sectional_rank"],
                "factor_names": ["momentum_5d", "vol_ratio_5d", "close_position"],
                "expected_directions": {
                    "momentum_5d": -1,
                    "vol_ratio_5d": -1,
                    "close_position": 1,
                },
                "weights": {
                    "momentum_5d": 0.4,
                    "vol_ratio_5d": 0.25,
                    "close_position": 0.35,
                },
                "parameters": {"entry_score": 78.0, "exit_score": 42.0, "top_rank": 20},
            },
            "falsification": [
                "训练窗组合IC不能稳定为正",
                "独立样本外最近一年或最近三个月净收益不为正",
                "延迟成交或双倍成本后收益失效",
            ],
            "data_requirements": ["daily_enriched", "historical_universe"],
        },
        {
            "hypothesis_id": "ah-system-limit-memory-v1",
            "version": "1.0.0",
            "source_kind": "prior",
            "title": "涨停记忆与资金注意力延续",
            "thesis": "过去60日具有涨停记忆、近期量价重新增强但尚未出现极端乖离的股票，未来5日净收益更高。",
            "mechanism": "A股涨跌停制度使信息和交易需求无法一次出清，历史涨停形成辨识度，二次放量可能反映资金重新聚集。",
            "prediction_object": "forward_net_return",
            "asset_type": "stock",
            "forward_horizon": 5,
            "information_domains": ["price_volume", "attention", "market_structure"],
            "test_spec": {
                "engine_ids": ["cross_sectional_rank"],
                "factor_names": ["limit_up_count_60d", "amount_ratio_5d", "ma20_bias"],
                "expected_directions": {
                    "limit_up_count_60d": 1,
                    "amount_ratio_5d": 1,
                    "ma20_bias": -1,
                },
                "weights": {
                    "limit_up_count_60d": 0.45,
                    "amount_ratio_5d": 0.35,
                    "ma20_bias": 0.2,
                },
                "parameters": {"entry_score": 80.0, "exit_score": 45.0, "top_rank": 15},
            },
            "falsification": [
                "训练窗组合IC方向与预期相反",
                "收益仅来自少数涨停不可成交样本或集中在单一年度",
                "次日开盘与容量压力后净收益不为正",
            ],
            "data_requirements": ["daily_enriched", "historical_universe"],
        },
        {
            "hypothesis_id": "ah-system-quality-repricing-v1",
            "version": "1.0.0",
            "source_kind": "prior",
            "title": "质量增长公告后的缓慢重估",
            "thesis": "最新已公告的盈利增长、收入增长和ROE较高且负债较低的股票，未来20日净收益更高。",
            "mechanism": "财务信息复杂且机构调整仓位存在摩擦，盈利质量改善可能在公告后被市场分步吸收。",
            "prediction_object": "forward_net_return",
            "asset_type": "stock",
            "forward_horizon": 20,
            "information_domains": ["fundamentals", "expectation_revision"],
            "test_spec": {
                "engine_ids": ["cross_sectional_rank"],
                "factor_names": ["net_income_yoy_latest", "revenue_yoy_latest", "roe_latest", "debt_ratio_latest"],
                "expected_directions": {
                    "net_income_yoy_latest": 1,
                    "revenue_yoy_latest": 1,
                    "roe_latest": 1,
                    "debt_ratio_latest": -1,
                },
                "weights": {
                    "net_income_yoy_latest": 0.35,
                    "revenue_yoy_latest": 0.25,
                    "roe_latest": 0.25,
                    "debt_ratio_latest": 0.15,
                },
                "parameters": {"entry_score": 75.0, "exit_score": 40.0, "top_rank": 20},
            },
            "falsification": [
                "公告时点PIT财务覆盖不足则阻断，不允许使用最新快照回填历史",
                "独立样本外净收益、稳定性或成本门槛失败",
            ],
            "data_requirements": ["daily_enriched", "historical_universe", "financial_pit"],
        },
    ]
    return [_normalize(row, allow_system=True) for row in rows]


class AlphaHypothesisStore:
    """Create-only storage; runs may append lineage without changing the hypothesis contract."""

    def __init__(self, data_dir: Path | str) -> None:
        self.root = (Path(data_dir).resolve() / "alpha_mining" / "hypotheses").resolve()

    def list_saved(self) -> list[dict[str, Any]]:
        with _LOCK:
            if not self.root.is_dir():
                return []
            rows = [self._read(path) for path in self.root.glob("ah-*.json")]
            return sorted(rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)

    def list_all(self) -> list[dict[str, Any]]:
        saved = {str(row["hypothesis_id"]): row for row in self.list_saved()}
        for row in system_hypotheses():
            saved.setdefault(str(row["hypothesis_id"]), row)
        return sorted(saved.values(), key=lambda row: (row["source_kind"] != "prior", row["title"]))

    def get(self, hypothesis_id: str) -> dict[str, Any]:
        path = self._path(hypothesis_id)
        with _LOCK:
            if path.is_file():
                return self._read(path)
        builtin = next((row for row in system_hypotheses() if row["hypothesis_id"] == hypothesis_id), None)
        if builtin is None:
            raise KeyError(hypothesis_id)
        return builtin

    def create(self, value: dict[str, Any]) -> dict[str, Any]:
        raw = deepcopy(value)
        raw.setdefault("hypothesis_id", f"ah-{uuid.uuid4().hex[:24]}")
        raw.setdefault("version", "1.0.0")
        raw.setdefault("created_at", datetime.now(UTC).isoformat())
        raw.setdefault("status", "proposed")
        raw.setdefault("run_ids", [])
        normalized = _normalize(raw, allow_system=False)
        path = self._path(str(normalized["hypothesis_id"]))
        with _LOCK:
            if path.exists() or any(
                row["hypothesis_id"] == normalized["hypothesis_id"] for row in system_hypotheses()
            ):
                raise ValueError("Alpha假设ID已存在，已冻结假设不可覆盖")
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, normalized, create_only=True)
        return normalized

    def attach_run(self, hypothesis_id: str, run_id: str) -> dict[str, Any]:
        with _LOCK:
            try:
                value = self.get(hypothesis_id)
            except KeyError as exc:
                raise ValueError("Alpha假设不存在") from exc
            path = self._path(hypothesis_id)
            if not path.exists():
                self.root.mkdir(parents=True, exist_ok=True)
                _atomic_json(path, value, create_only=True)
            updated = self._read(path)
            run_ids = list(updated.get("run_ids") or [])
            if run_id not in run_ids:
                run_ids.append(run_id)
            updated["run_ids"] = run_ids
            updated["status"] = "running"
            updated["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_json(path, updated)
            return updated

    def record_result(
        self,
        hypothesis_id: str,
        run_id: str,
        *,
        verdict: Literal["supported", "rejected", "blocked", "cancelled"],
        conclusion: str,
        next_hypothesis_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        with _LOCK:
            value = self.attach_run(hypothesis_id, run_id)
            value["status"] = verdict
            results = list(value.get("results") or [])
            results.append({
                "run_id": run_id,
                "verdict": verdict,
                "conclusion": conclusion,
                "next_hypothesis_ids": list(next_hypothesis_ids or []),
                "recorded_at": datetime.now(UTC).isoformat(),
            })
            value["results"] = results
            value["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_json(self._path(hypothesis_id), value)
            return value

    def create_from_failure(
        self,
        parent: dict[str, Any],
        run_id: str,
        suggestion: dict[str, Any],
        source_request: dict[str, Any],
    ) -> dict[str, Any]:
        patch = dict(suggestion.get("request_patch") or {})
        factors = list(patch.get("factor_names") or source_request.get("factor_names") or [])
        engines = list(patch.get("engine_ids") or source_request.get("engine_ids") or [])
        previous_spec = dict(parent.get("test_spec") or {})
        prior_directions = dict(previous_spec.get("expected_directions") or {})
        direction_priors = {
            "turnover_rate": -1,
            "amihud_20d": -1,
            "log_amount": 1,
        }
        directions = {factor: int(prior_directions.get(factor, direction_priors.get(factor, 1))) for factor in factors}
        weights = {factor: 1.0 / max(len(factors), 1) for factor in factors}
        fingerprint = hashlib.sha256(
            json.dumps(
                {"parent": parent["hypothesis_id"], "run": run_id, "suggestion": suggestion.get("suggestion_id")},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
        return self.create({
            "hypothesis_id": f"ah-failure-{fingerprint}",
            "version": "1.0.0",
            "source_kind": "failure",
            "parent_hypothesis_id": parent["hypothesis_id"],
            "source_run_id": run_id,
            "source_suggestion_id": suggestion.get("suggestion_id"),
            "title": str(suggestion.get("title") or "失败证据生成的下一轮假设"),
            "thesis": str(suggestion.get("why") or parent.get("thesis") or ""),
            "mechanism": str(parent.get("mechanism") or "基于上一轮失败证据改变可证伪条件"),
            "prediction_object": parent.get("prediction_object", "forward_net_return"),
            "asset_type": patch.get("asset_type", source_request.get("asset_type", "stock")),
            "forward_horizon": patch.get("forward_horizon", source_request.get("forward_horizon", 5)),
            "information_domains": list(parent.get("information_domains") or ["price_volume"]),
            "test_spec": {
                "engine_ids": engines,
                "factor_names": factors,
                "expected_directions": directions,
                "weights": weights,
                "parameters": dict(previous_spec.get("parameters") or {}),
            },
            "falsification": list(parent.get("falsification") or ["独立样本外硬门槛失败"]),
            "data_requirements": list(parent.get("data_requirements") or ["daily_enriched", "historical_universe"]),
            "request_patch": patch,
        })

    def _path(self, hypothesis_id: str) -> Path:
        if not hypothesis_id.startswith("ah-") or "/" in hypothesis_id or ".." in hypothesis_id:
            raise ValueError("Alpha假设ID非法")
        return self.root / f"{hypothesis_id}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Alpha假设证据损坏") from exc
        if not isinstance(value, dict):
            raise ValueError("Alpha假设证据必须是对象")
        # Older prototypes mislabeled fixed templates as "system" hypotheses.
        # Keep their evidence immutable on disk while correcting the API meaning.
        if value.get("source_kind") == "system":
            value["source_kind"] = "prior"
        return value


def _normalize(value: dict[str, Any], *, allow_system: bool) -> dict[str, Any]:
    result = deepcopy(value)
    required_text = (
        "hypothesis_id", "version", "source_kind", "title", "thesis", "mechanism",
        "prediction_object", "asset_type",
    )
    for field in required_text:
        if not isinstance(result.get(field), str) or not str(result[field]).strip():
            raise ValueError(f"Alpha假设缺少字段: {field}")
    if result["source_kind"] not in _SOURCE_KINDS:
        raise ValueError("Alpha假设来源非法")
    if result["source_kind"] == "prior" and not allow_system:
        raise ValueError("不能通过人工接口创建内置研究先验")
    if result["asset_type"] not in {"stock", "etf"}:
        raise ValueError("Alpha假设资产类型非法")
    horizon = result.get("forward_horizon")
    if isinstance(horizon, bool) or horizon not in {1, 3, 5, 10, 20, 60}:
        raise ValueError("Alpha假设预测期限非法")
    spec = result.get("test_spec")
    if not isinstance(spec, dict):
        raise ValueError("Alpha假设缺少检验方案")
    factors = spec.get("factor_names")
    engines = spec.get("engine_ids")
    directions = spec.get("expected_directions")
    weights = spec.get("weights")
    if not isinstance(factors, list) or not factors or len(factors) != len(set(factors)):
        raise ValueError("Alpha假设因子必须非空且唯一")
    unknown = sorted(set(factors) - _FACTOR_IDS)
    if unknown:
        raise ValueError(f"Alpha假设使用未知因子: {unknown}")
    if not isinstance(engines, list) or not engines or len(engines) != len(set(engines)):
        raise ValueError("Alpha假设发现引擎必须非空且唯一")
    if not isinstance(directions, dict) or set(directions) != set(factors):
        raise ValueError("Alpha假设必须为每个因子预注册方向")
    if any(direction not in _DIRECTIONS for direction in directions.values()):
        raise ValueError("Alpha假设方向只能是-1或1")
    if not isinstance(weights, dict) or set(weights) != set(factors):
        raise ValueError("Alpha假设必须为每个因子预注册权重")
    numeric_weights = {key: float(weights[key]) for key in factors}
    if any(weight <= 0 for weight in numeric_weights.values()) or sum(numeric_weights.values()) <= 0:
        raise ValueError("Alpha假设权重必须为正")
    spec["expected_directions"] = {key: int(directions[key]) for key in factors}
    spec["weights"] = numeric_weights
    spec.setdefault("parameters", {"entry_score": 75.0, "exit_score": 40.0, "top_rank": 20})
    result["test_spec"] = spec
    for field in ("information_domains", "falsification", "data_requirements"):
        if not isinstance(result.get(field), list) or not result[field]:
            raise ValueError(f"Alpha假设缺少字段: {field}")
    result.setdefault("status", "proposed")
    result.setdefault("run_ids", [])
    result.setdefault("results", [])
    result.setdefault("created_at", None)
    result.setdefault("updated_at", None)
    return result


def _atomic_json(path: Path, payload: dict[str, Any], *, create_only: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create_only:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ValueError("Alpha假设ID已存在，已冻结假设不可覆盖") from exc
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
