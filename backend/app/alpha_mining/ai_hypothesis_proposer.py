"""DeepSeek-backed Alpha hypothesis proposal with deterministic validation."""
# ruff: noqa: RUF001
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from app.alpha_mining.data_catalog import AlphaResearchDataCatalog
from app.alpha_mining.hypotheses import AlphaHypothesisStore
from app.backtest.factor import FACTOR_COLUMNS
from app.services.ai_provider import (
    ai_configured,
    current_ai_model,
    current_ai_provider,
    generate_ai_text,
)
from app.services.mining_preflight import enriched_partition_dates

TextGenerator = Callable[..., Awaitable[str]]
_SHARE_FACTORS = frozenset({"turnover_rate", "turnover_ratio_5d", "turnover_z_60d"})
_HORIZONS = frozenset({1, 3, 5, 10, 20})


class AlphaAIHypothesisProposer:
    """Let DeepSeek propose; let deterministic code decide what may be persisted."""

    def __init__(self, data_dir: Path | str, *, generator: TextGenerator | None = None) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.store = AlphaHypothesisStore(self.data_dir)
        self.generator = generator or generate_ai_text

    async def propose(
        self,
        *,
        asset_type: Literal["stock", "etf"],
        start: date,
        end: date,
        count: int = 3,
        research_focus: str = "",
    ) -> dict[str, Any]:
        if count < 1 or count > 6:
            raise ValueError("DeepSeek单次只能提出1至6条假设")
        provider = current_ai_provider()
        model = current_ai_model()
        if not ai_configured(provider):
            raise ValueError("DeepSeek尚未配置，请先在设置页配置AI服务")
        if "deepseek" not in model.lower():
            raise ValueError(f"当前AI模型是 {model or '未配置'}，请切换到DeepSeek后再生成假设")
        dates = enriched_partition_dates(self.data_dir, asset_type, start, end)
        if not dates:
            raise ValueError("所选区间没有可用于提出假设的日频数据")
        catalog = AlphaResearchDataCatalog(self.data_dir).snapshot(dates[0], dates[-1], asset_type)
        for dataset_id in ("daily_enriched", "historical_universe"):
            qualification = catalog.datasets[dataset_id]
            if not qualification.ready:
                raise ValueError("DeepSeek假设上下文数据门禁失败: " + "; ".join(qualification.reasons))

        factor_rows = self._available_factors(catalog)
        if len(factor_rows) < 3:
            raise ValueError("当前可用因子不足，无法生成可执行假设")
        market_snapshot = _market_snapshot(self.data_dir, asset_type, dates[-2:])
        existing = [
            {
                "title": item["title"],
                "factors": (item.get("test_spec") or {}).get("factor_names") or [],
                "source": item["source_kind"],
            }
            for item in self.store.list_all()
        ]
        context = {
            "asset_type": asset_type,
            "research_window": [dates[0].isoformat(), dates[-1].isoformat()],
            "market_snapshot": market_snapshot,
            "available_factors": factor_rows,
            "executable_engine": {
                "engine_id": "cross_sectional_rank",
                "meaning": "按每个交易日全市场截面计算冻结加权分数，收盘决策、次日开盘成交",
            },
            "unavailable_datasets": {
                key: list(value.reasons)
                for key, value in catalog.datasets.items()
                if not value.ready
            },
            "existing_hypotheses_for_deduplication": existing,
            "research_focus": research_focus.strip()[:500],
        }
        prompt = _proposal_prompt(context, count)
        prompt_sha = _sha256(prompt)
        context_sha = _sha256(json.dumps(context, ensure_ascii=False, sort_keys=True))
        request_sha = _sha256(
            json.dumps(
                {
                    "provider": provider,
                    "model": model,
                    "prompt_sha256": prompt_sha,
                    "context_sha256": context_sha,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        batch_id = f"ahp-{request_sha[:24]}"
        receipt: dict[str, Any] = {
            "batch_id": batch_id,
            "created_at": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "request_sha256": request_sha,
            "prompt_sha256": prompt_sha,
            "context_sha256": context_sha,
            "outcome_data_exposed": False,
            "hypothesis_ids": [],
            "rejected": [],
            "prompt": prompt,
            "raw_response": None,
            "status": "running",
        }
        final_receipt = _read_receipt(self.data_dir, batch_id)
        if final_receipt is not None:
            if final_receipt.get("status") == "accepted":
                return {
                    "batch_id": batch_id,
                    "provider": provider,
                    "model": model,
                    "outcome_data_exposed": False,
                    "items": [
                        self.store.get(str(hypothesis_id))
                        for hypothesis_id in final_receipt.get("hypothesis_ids") or []
                    ],
                    "rejected": list(final_receipt.get("rejected") or []),
                    "reused_receipt": True,
                }
            raise ValueError(
                "相同DeepSeek请求已有失败凭证；为防止重复计费，系统不会自动重试"
            )

        response_receipt = _read_receipt_stage(self.data_dir, batch_id, "response")
        if response_receipt is not None and response_receipt.get("raw_response"):
            receipt = response_receipt
            response = str(response_receipt["raw_response"])
        else:
            if _read_receipt_stage(self.data_dir, batch_id, "running") is not None:
                raise ValueError(
                    "相同DeepSeek请求已有未决调用凭证；为防止重复计费，系统不会自动重试"
                )
            _write_receipt_stage(self.data_dir, receipt, "running")
            try:
                response = await self.generator(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是A股量化研究负责人。只提出可证伪、可用给定字段执行的研究假设；"
                                "不得承诺收益，不得引用任何现有策略作为底座，不得编造数据或因子。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=None,
                    timeout=180.0,
                )
            except Exception as exc:
                receipt.update({"status": "provider_error", "error": str(exc)[:1000]})
                _write_receipt(self.data_dir, receipt)
                raise
            response_sha = _sha256(response)
            receipt.update(
                {
                    "raw_response": response,
                    "response_sha256": response_sha,
                    "status": "response_received",
                }
            )
            _write_receipt_stage(self.data_dir, receipt, "response")
        response_sha = str(receipt["response_sha256"])
        proposals = _decode_proposals(response)
        if not proposals:
            receipt.update({"status": "contract_rejected", "error": "DeepSeek没有返回可解析的假设"})
            _write_receipt(self.data_dir, receipt)
            raise ValueError("DeepSeek没有返回可解析的假设")
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        available_ids = {row["id"] for row in factor_rows}
        for index, proposal in enumerate(proposals[:count]):
            try:
                payload = _validated_payload(
                    proposal,
                    asset_type=asset_type,
                    available_factor_ids=available_ids,
                    batch_id=batch_id,
                    provider=provider,
                    model=model,
                    prompt_sha=prompt_sha,
                    response_sha=response_sha,
                    context_sha=context_sha,
                    market_snapshot=market_snapshot,
                )
                try:
                    accepted.append(self.store.create(payload))
                except ValueError as exc:
                    if "ID已存在" not in str(exc):
                        raise
                    accepted.append(self.store.get(str(payload["hypothesis_id"])))
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append({"index": str(index), "reason": str(exc)[:500]})
        if not accepted:
            detail = "; ".join(item["reason"] for item in rejected[:3])
            receipt.update({"status": "contract_rejected", "rejected": rejected, "error": detail})
            _write_receipt(self.data_dir, receipt)
            raise ValueError("DeepSeek返回内容未通过确定性合同校验: " + detail)
        receipt.update({
            "status": "accepted",
            "hypothesis_ids": [item["hypothesis_id"] for item in accepted],
            "rejected": rejected,
        })
        _write_receipt(self.data_dir, receipt)
        return {
            "batch_id": batch_id,
            "provider": provider,
            "model": model,
            "outcome_data_exposed": False,
            "items": accepted,
            "rejected": rejected,
        }

    @staticmethod
    def _available_factors(catalog: Any) -> list[dict[str, str]]:
        financial_ready = catalog.datasets["financial_pit"].ready
        shares_ready = catalog.datasets["share_history_pit"].ready
        rows = []
        for item in FACTOR_COLUMNS:
            factor_id = str(item["id"])
            if item.get("group") == "财务" and not financial_ready:
                continue
            if factor_id in _SHARE_FACTORS and not shares_ready:
                continue
            rows.append({
                "id": factor_id,
                "name": str(item.get("label") or factor_id),
                "group": str(item.get("group") or "其他"),
                "definition": str(item.get("desc") or ""),
            })
        return rows


def _proposal_prompt(context: dict[str, Any], count: int) -> str:
    return f"""请提出 {count} 条彼此机制不同的A股Alpha研究假设。

输入上下文只包含可用字段和当前市场状态，不包含任何历史候选的样本外收益；因此不能声称某个方向已经有效。
当前市场快照只用于启发机制，不得把单日状态直接当作收益证据。

强制要求：
1. 每条假设必须解释A股特有或显著的制度/行为机制，例如T+1、涨跌停、散户注意力、融资与流动性、信息扩散或拥挤交易。
2. 每条使用2至5个 available_factors 中的ID；不得输出目录外字段。
3. engine_ids固定为 ["cross_sectional_rank"]；预测期限只能是1、3、5、10、20日之一。
4. expected_directions中高值预期未来净收益更高写1，更低写-1；weights必须全部为正且总和为1。
5. 系统会用方向调整后的加权分数排序，因此“组合分数越高”必须始终表示预期未来净收益越高；证伪条件也必须按这个方向表述。
6. 给出至少2条明确证伪条件。不得使用“收益更好”“表现优秀”等不可检验空话。
7. 不得引用新低反转或任何现有策略作为底座；不得给出回测收益、胜率或夏普等未经检验的数字。
8. 与 existing_hypotheses_for_deduplication 重复的因子组合和论点必须避开。

只返回严格JSON对象，不要Markdown：
{{
  "hypotheses": [
    {{
      "title": "中文标题",
      "thesis": "明确说明哪些事前特征预测未来几日净收益方向",
      "mechanism": "A股机制解释",
      "forward_horizon": 5,
      "information_domains": ["price_volume", "liquidity"],
      "factor_names": ["factor_id_1", "factor_id_2"],
      "expected_directions": {{"factor_id_1": 1, "factor_id_2": -1}},
      "weights": {{"factor_id_1": 0.5, "factor_id_2": 0.5}},
      "falsification": ["条件1", "条件2"]
    }}
  ]
}}

上下文：
{json.dumps(context, ensure_ascii=False, sort_keys=True)}"""


def _decode_proposals(response: str) -> list[dict[str, Any]]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    rows = payload.get("hypotheses") if isinstance(payload, dict) else None
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _validated_payload(
    proposal: dict[str, Any],
    *,
    asset_type: str,
    available_factor_ids: set[str],
    batch_id: str,
    provider: str,
    model: str,
    prompt_sha: str,
    response_sha: str,
    context_sha: str,
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    title = str(proposal.get("title") or "").strip()
    thesis = str(proposal.get("thesis") or "").strip()
    mechanism = str(proposal.get("mechanism") or "").strip()
    if not 3 <= len(title) <= 120 or len(thesis) < 10 or len(mechanism) < 10:
        raise ValueError("标题、论点或机制说明不完整")
    horizon = proposal.get("forward_horizon")
    if isinstance(horizon, bool) or horizon not in _HORIZONS:
        raise ValueError("预测期限不受支持")
    factors = proposal.get("factor_names")
    if not isinstance(factors, list) or not 2 <= len(factors) <= 5:
        raise ValueError("每条假设必须包含2至5个因子")
    factors = [str(value) for value in factors]
    if len(factors) != len(set(factors)) or not set(factors) <= available_factor_ids:
        raise ValueError("DeepSeek使用了重复或当前不可用的因子")
    directions = proposal.get("expected_directions")
    weights = proposal.get("weights")
    if not isinstance(directions, dict) or set(directions) != set(factors):
        raise ValueError("DeepSeek没有为全部因子预注册方向")
    if not isinstance(weights, dict) or set(weights) != set(factors):
        raise ValueError("DeepSeek没有为全部因子预注册权重")
    normalized_directions = {factor: int(directions[factor]) for factor in factors}
    if any(value not in (-1, 1) for value in normalized_directions.values()):
        raise ValueError("因子方向只能是-1或1")
    raw_weights = {factor: float(weights[factor]) for factor in factors}
    if any(value <= 0 for value in raw_weights.values()):
        raise ValueError("因子权重必须为正")
    total = sum(raw_weights.values())
    normalized_weights = {factor: value / total for factor, value in raw_weights.items()}
    falsification = proposal.get("falsification")
    if not isinstance(falsification, list) or len(falsification) < 2:
        raise ValueError("至少需要2条证伪条件")
    domains = proposal.get("information_domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("信息域不能为空")
    requirements = ["daily_enriched", "historical_universe"]
    if set(factors) & _SHARE_FACTORS:
        requirements.append("share_history_pit")
    if any(item.get("id") in factors and item.get("group") == "财务" for item in FACTOR_COLUMNS):
        requirements.append("financial_pit")
    identity = {
        "title": title,
        "thesis": thesis,
        "horizon": horizon,
        "factors": factors,
        "directions": normalized_directions,
        "weights": normalized_weights,
    }
    hypothesis_id = "ah-ai-" + _sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True)
    )[:20]
    return {
        "hypothesis_id": hypothesis_id,
        "version": "1.0.0",
        "source_kind": "ai",
        "title": title,
        "thesis": thesis,
        "mechanism": mechanism,
        "prediction_object": "forward_net_return",
        "asset_type": asset_type,
        "forward_horizon": horizon,
        "information_domains": [str(value) for value in domains[:8]],
        "test_spec": {
            "engine_ids": ["cross_sectional_rank"],
            "factor_names": factors,
            "expected_directions": normalized_directions,
            "weights": normalized_weights,
            "parameters": {"entry_score": 75.0, "exit_score": 40.0, "top_rank": 20},
        },
        "falsification": [str(value) for value in falsification[:8]],
        "data_requirements": requirements,
        "provenance": {
            "proposal_batch_id": batch_id,
            "provider": provider,
            "model": model,
            "prompt_sha256": prompt_sha,
            "response_sha256": response_sha,
            "context_sha256": context_sha,
            "market_snapshot_date": market_snapshot.get("date"),
            "outcome_data_exposed": False,
        },
    }


def _market_snapshot(data_dir: Path, asset_type: str, dates: list[date]) -> dict[str, Any]:
    root = data_dir / ("kline_daily_enriched" if asset_type == "stock" else "kline_daily_enriched_etf")
    frames = []
    for day in dates:
        paths = sorted((root / f"date={day.isoformat()}").glob("*.parquet"))
        if paths:
            schema = pl.read_parquet_schema(paths[0])
            columns = [
                column
                for column in (
                    "symbol", "date", "close", "amount", "turnover_rate",
                    "consecutive_limit_ups", "consecutive_limit_downs",
                )
                if column in schema
            ]
            frames.append(pl.concat(
                [pl.read_parquet(path, columns=columns) for path in paths],
                how="diagonal_relaxed",
            ))
    if not frames:
        return {}
    panel = pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "date"])
    panel = panel.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("return")
    )
    latest_date = panel.get_column("date").max()
    latest = panel.filter(pl.col("date") == latest_date)
    snapshot: dict[str, Any] = {
        "date": latest_date.isoformat() if isinstance(latest_date, date) else str(latest_date),
        "stock_count": latest.height,
        "equal_weight_return": _finite(latest.get_column("return").mean()),
        "median_return": _finite(latest.get_column("return").median()),
        "advance_ratio": _finite((latest.get_column("return") > 0).mean()),
    }
    if "turnover_rate" in latest.columns:
        snapshot["median_turnover_rate"] = _finite(latest.get_column("turnover_rate").median())
    if "amount" in latest.columns:
        snapshot["median_amount"] = _finite(latest.get_column("amount").median())
    if "consecutive_limit_ups" in latest.columns:
        snapshot["limit_up_count"] = int((latest.get_column("consecutive_limit_ups") > 0).sum())
    if "consecutive_limit_downs" in latest.columns:
        snapshot["limit_down_count"] = int((latest.get_column("consecutive_limit_downs") > 0).sum())
    return snapshot


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _receipt_root(data_dir: Path) -> Path:
    root = data_dir / "alpha_mining" / "hypothesis_proposals"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _read_receipt(data_dir: Path, batch_id: str) -> dict[str, Any] | None:
    return _read_json(_receipt_root(data_dir) / f"{batch_id}.json")


def _read_receipt_stage(
    data_dir: Path, batch_id: str, stage: str
) -> dict[str, Any] | None:
    return _read_json(_receipt_root(data_dir) / f"{batch_id}.{stage}.json")


def _write_immutable_json(target: Path, payload: dict[str, Any]) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, target)
    temporary.unlink(missing_ok=True)


def _write_receipt(data_dir: Path, payload: dict[str, Any]) -> None:
    target = _receipt_root(data_dir) / f"{payload['batch_id']}.json"
    _write_immutable_json(target, payload)


def _write_receipt_stage(
    data_dir: Path, payload: dict[str, Any], stage: str
) -> None:
    target = _receipt_root(data_dir) / f"{payload['batch_id']}.{stage}.json"
    _write_immutable_json(target, payload)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
