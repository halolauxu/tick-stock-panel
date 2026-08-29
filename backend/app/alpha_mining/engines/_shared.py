from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import polars as pl

from app.alpha_mining.contracts import (
    CandidateSpec,
    DataCatalogContext,
    DataQualification,
    FrozenSignalSpec,
)


def qualification(
    context: DataCatalogContext,
    required: Iterable[str] = (),
) -> DataQualification:
    missing = sorted(set(required) - set(context.available_features))
    reasons = tuple(f"缺少字段: {name}" for name in missing)
    return DataQualification(
        ready=not reasons,
        reasons=reasons,
        observations={"available_feature_count": len(context.available_features)},
    )


def rank_univariate_candidates(
    *,
    context,
    budget,
    engine_id: str,
    engine_version: str,
    recipe_prefix: str,
    name_prefix: str,
    thesis: str,
    features: Iterable[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> list[CandidateSpec]:
    """Shared deterministic univariate search; callers own the research thesis."""
    pool = tuple(features or context.feature_names)
    ranked: list[tuple[float, str, dict[str, float | int]]] = []
    for feature in pool[: budget.max_trials]:
        if feature not in context.frame.columns:
            record_trial(context, f"{recipe_prefix}.{feature}", "failed", {"reason": "missing_feature"})
            continue
        metric = daily_rank_ic(
            context.frame,
            feature,
            context.target_column,
            min_cross_section=budget.min_cross_section,
        )
        if metric is None or int(metric["valid_dates"]) < budget.min_dates:
            record_trial(context, f"{recipe_prefix}.{feature}", "failed", {
                "reason": "insufficient_valid_dates",
                "metric": metric,
            })
            continue
        if not discovery_signal(metric):
            record_trial(context, f"{recipe_prefix}.{feature}", "failed", {
                "reason": "effect_floor",
                "metric": metric,
            })
            continue
        score = abs(float(metric["ic_mean"])) * float(metric["positive_date_ratio"])
        record_trial(context, f"{recipe_prefix}.{feature}", "eligible", {
            "metric": metric,
            "selection_score": score,
        })
        ranked.append((score, feature, metric))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    defaults = {"entry_score": 70.0, "exit_score": 40.0, "top_rank": 20}
    defaults.update(parameters or {})
    return [
        CandidateSpec(
            recipe_id=f"{recipe_prefix}.{feature}",
            engine_id=engine_id,
            engine_version=engine_version,
            name=f"{name_prefix} · {feature}",
            thesis=thesis,
            signal_kind="factor_rank",
            features=(feature,),
            directions=(1 if float(metric["ic_mean"]) >= 0 else -1,),
            weights=(1.0,),
            parameters=defaults,
            train_evidence={**metric, "selection_score": score},
        )
        for score, feature, metric in ranked[: budget.max_candidates]
    ]


def discovery_signal(metric: dict[str, float | int]) -> bool:
    """Pre-registered effect floor used before hidden-window selection."""
    return (
        abs(float(metric["ic_mean"])) >= 0.02
        and abs(float(metric["ic_ir"])) >= 0.20
        and float(metric["positive_date_ratio"]) >= 0.55
    )


def record_trial(context, recipe_id: str, status: str, evidence: dict[str, Any]) -> None:
    """Append one attempted recipe to the orchestrator-owned immutable ledger sink."""
    sink = context.metadata.get("trial_audit") if context.metadata else None
    if isinstance(sink, list):
        sink.append({
            "recipe_id": recipe_id,
            "status": status,
            "evidence": evidence,
        })


def daily_rank_ic(
    frame: pl.DataFrame,
    feature: str,
    target: str,
    *,
    min_cross_section: int,
) -> dict[str, float | int] | None:
    daily = (
        frame.select("date", feature, target)
        .drop_nulls()
        .group_by("date")
        .agg(
            pl.len().alias("n"),
            pl.corr(feature, target, method="spearman").alias("ic"),
        )
        .filter((pl.col("n") >= min_cross_section) & pl.col("ic").is_finite())
        .sort("date")
    )
    if daily.is_empty():
        return None
    values = [float(value) for value in daily.get_column("ic").to_list()]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    std = math.sqrt(variance)
    coverage = frame.select(pl.col(feature).is_not_null().mean()).item()
    return {
        "ic_mean": mean,
        # A perfectly stable non-zero IC has zero dispersion and therefore an
        # unbounded information ratio. Treating it as zero rejects the
        # strongest possible deterministic true positive.
        "ic_ir": mean / std if std > 0 else (math.copysign(math.inf, mean) if mean else 0.0),
        "positive_date_ratio": sum(value * mean > 0 for value in values) / len(values),
        "valid_dates": len(values),
        "coverage": float(coverage or 0.0),
    }


def materialize(candidate: CandidateSpec) -> FrozenSignalSpec:
    return FrozenSignalSpec.from_candidate(candidate)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
