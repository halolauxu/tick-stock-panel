"""Cross-sectional rank persistence/reversal discovery."""
from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    CandidateSpec,
    DataCatalogContext,
    DatasetRequirement,
    FeatureContext,
    FrozenSignalSpec,
    TrainOnlyContext,
    TrialBudget,
)
from app.alpha_mining.engines._shared import (
    daily_rank_ic,
    discovery_signal,
    materialize,
    qualification,
    record_trial,
)


class CrossSectionalEngine:
    manifest = AlphaEngineManifest(
        engine_id="cross_sectional_rank",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="截面排序发现",
        family="cross_sectional",
        information_domains=("price_volume", "liquidity", "fundamentals"),
        mechanism_classes=("risk_compensation", "behavioral_underreaction"),
        economic_mechanism="风险补偿、行为偏差与资金拥挤在同日股票截面留下可排序差异",
        discovery_classes=("cross_sectional_rank",),
        discovery_method="逐日秩相关与稳定性筛选",
        prediction_objects=("forward_net_return", "rank_outperformance"),
        forecast_targets=("1d", "3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(1, 3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="逐日比较全市场股票, 不依赖任何既有策略作为底座。",
    )

    def preflight(self, context: DataCatalogContext):
        return qualification(context)

    def discover(
        self,
        context: TrainOnlyContext,
        budget: TrialBudget,
    ) -> list[CandidateSpec]:
        hypothesis = context.metadata.get("hypothesis_contract")
        if isinstance(hypothesis, Mapping):
            return self._test_preregistered_hypothesis(context, budget, hypothesis)
        ranked: list[tuple[float, str, dict[str, float | int]]] = []
        for feature in context.feature_names[: budget.max_trials]:
            metric = daily_rank_ic(
                context.frame,
                feature,
                context.target_column,
                min_cross_section=budget.min_cross_section,
            )
            if metric is None or int(metric["valid_dates"]) < budget.min_dates:
                record_trial(context, f"cross_sectional_rank.{feature}", "failed", {
                    "reason": "insufficient_valid_dates",
                    "metric": metric,
                })
                continue
            if not discovery_signal(metric):
                record_trial(context, f"cross_sectional_rank.{feature}", "failed", {
                    "reason": "effect_floor",
                    "metric": metric,
                })
                continue
            score = abs(float(metric["ic_mean"])) * max(float(metric["positive_date_ratio"]), 0.0)
            record_trial(context, f"cross_sectional_rank.{feature}", "eligible", {
                "metric": metric,
                "selection_score": score,
            })
            ranked.append((score, feature, metric))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        output: list[CandidateSpec] = []
        for score, feature, metric in ranked[: budget.max_candidates]:
            direction = 1 if float(metric["ic_mean"]) >= 0 else -1
            output.append(CandidateSpec(
                recipe_id=f"cross_sectional_rank.{feature}",
                engine_id=self.manifest.engine_id,
                engine_version=self.manifest.version,
                name=f"截面排序 · {feature}",
                thesis="训练窗内该特征与未来净收益的逐日截面排序关系具有一致性。",
                signal_kind="factor_rank",
                features=(feature,),
                directions=(direction,),
                weights=(1.0,),
                parameters={"entry_score": 70.0, "exit_score": 40.0, "top_rank": 20},
                train_evidence={**metric, "selection_score": score},
            ))
        return output

    def _test_preregistered_hypothesis(
        self,
        context: TrainOnlyContext,
        budget: TrialBudget,
        hypothesis: Mapping[str, object],
    ) -> list[CandidateSpec]:
        spec = hypothesis.get("test_spec")
        if not isinstance(spec, Mapping):
            record_trial(context, "preregistered.invalid", "failed", {"reason": "missing_test_spec"})
            return []
        factors = tuple(str(value) for value in spec.get("factor_names") or ())
        directions = spec.get("expected_directions")
        weights = spec.get("weights")
        if (
            not factors
            or not isinstance(directions, Mapping)
            or not isinstance(weights, Mapping)
            or set(factors) != set(directions)
            or set(factors) != set(weights)
        ):
            record_trial(context, "preregistered.invalid", "failed", {"reason": "invalid_factor_contract"})
            return []
        missing = sorted(set(factors) - set(context.frame.columns))
        if missing:
            record_trial(context, "preregistered.invalid", "failed", {"reason": "missing_features", "features": missing})
            return []
        signed_directions = tuple(int(directions[factor]) for factor in factors)
        raw_weights = tuple(float(weights[factor]) for factor in factors)
        total_weight = sum(abs(value) for value in raw_weights)
        if any(value not in (-1, 1) for value in signed_directions) or total_weight <= 0:
            record_trial(context, "preregistered.invalid", "failed", {"reason": "invalid_direction_or_weight"})
            return []
        normalized_weights = tuple(abs(value) / total_weight for value in raw_weights)
        rank_columns = []
        score_parts = []
        for index, (factor, direction, weight) in enumerate(
            zip(factors, signed_directions, normalized_weights, strict=True)
        ):
            name = f"_hypothesis_rank_{index}"
            rank_columns.append(
                (pl.col(factor).rank(method="average").over("date") / pl.len().over("date")).alias(name)
            )
            score = pl.col(name) if direction > 0 else 1.0 - pl.col(name)
            score_parts.append(score * weight)
        scored = context.frame.with_columns(rank_columns).with_columns(
            pl.sum_horizontal(score_parts).alias("_hypothesis_score")
        )
        metric = daily_rank_ic(
            scored,
            "_hypothesis_score",
            context.target_column,
            min_cross_section=budget.min_cross_section,
        )
        recipe_id = f"preregistered.{hypothesis.get('hypothesis_id') or 'anonymous'!s}"
        if (
            metric is None
            or int(metric["valid_dates"]) < budget.min_dates
            or float(metric["ic_mean"]) < 0.02
            or float(metric["ic_ir"]) < 0.20
            or float(metric["positive_date_ratio"]) < 0.55
        ):
            record_trial(context, recipe_id, "failed", {
                "reason": "preregistered_effect_floor",
                "metric": metric,
                "directions_locked_before_test": True,
            })
            return []
        record_trial(context, recipe_id, "eligible", {
            "metric": metric,
            "directions_locked_before_test": True,
        })
        parameters = dict(spec.get("parameters") or {})
        parameters.setdefault("entry_score", 75.0)
        parameters.setdefault("exit_score", 40.0)
        parameters.setdefault("top_rank", 20)
        return [CandidateSpec(
            recipe_id=recipe_id,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            name=str(hypothesis.get("title") or "预注册Alpha假设"),
            thesis=str(hypothesis.get("thesis") or "预注册方向组合预测未来净收益"),
            signal_kind="factor_rank",
            features=factors,
            directions=signed_directions,
            weights=normalized_weights,
            parameters={**parameters, "hypothesis_id": str(hypothesis.get("hypothesis_id") or "")},
            train_evidence={**metric, "directions_locked_before_test": True},
        )]

    def materialize(
        self,
        candidate: CandidateSpec,
        context: FeatureContext,
    ) -> FrozenSignalSpec:
        del context
        return materialize(candidate)


ENGINE = CrossSectionalEngine()
