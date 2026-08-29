"""Pairwise conjunctive interactions discovered strictly inside training windows."""
from __future__ import annotations

from itertools import combinations, product

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


class NonlinearInteractionEngine:
    manifest = AlphaEngineManifest(
        engine_id="nonlinear_interaction",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="非线性交互发现",
        family="interaction",
        information_domains=("price_volume", "liquidity", "fundamentals"),
        mechanism_classes=("behavioral_underreaction", "liquidity_pressure"),
        economic_mechanism="单因子弱效应可能只在另一状态同时满足时转化为可交易收益",
        discovery_classes=("nonlinear_interaction",),
        discovery_method="训练窗双变量分位交集搜索",
        prediction_objects=("forward_net_return", "rank_outperformance"),
        forecast_targets=("3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(3, 5, 10, 20),
        output_candidate_types=("factor_rank_intersection",),
        description="先在训练集发现双变量交集, 再冻结为高阈值双因子评分规则。",
    )

    def preflight(self, context: DataCatalogContext):
        return qualification(context)

    def discover(
        self,
        context: TrainOnlyContext,
        budget: TrialBudget,
    ) -> list[CandidateSpec]:
        singles: list[tuple[float, str]] = []
        trial_count = 0
        for feature in context.feature_names[: min(budget.max_trials, 32)]:
            trial_count += 1
            metric = daily_rank_ic(
                context.frame,
                feature,
                context.target_column,
                min_cross_section=budget.min_cross_section,
            )
            if (
                metric is not None
                and int(metric["valid_dates"]) >= budget.min_dates
                and discovery_signal(metric)
            ):
                singles.append((abs(float(metric["ic_mean"])), feature))
                record_trial(context, f"nonlinear_single.{feature}", "eligible", {"metric": metric})
            else:
                record_trial(context, f"nonlinear_single.{feature}", "failed", {
                    "reason": "effect_floor_or_sample",
                    "metric": metric,
                })
        feature_pool = [feature for _, feature in sorted(singles, reverse=True)[:8]]
        ranked: list[tuple[float, tuple[str, str], tuple[int, int], dict]] = []
        for left, right in combinations(feature_pool, 2):
            scoped = context.frame.select("date", left, right, context.target_column).drop_nulls()
            if scoped.is_empty():
                continue
            with_pct = scoped.with_columns(
                (pl.col(left).rank(method="average").over("date") / pl.len().over("date")).alias("_left_pct"),
                (pl.col(right).rank(method="average").over("date") / pl.len().over("date")).alias("_right_pct"),
            )
            for directions in product((-1, 1), repeat=2):
                trial_count += 1
                if trial_count > budget.max_trials:
                    break
                left_score = pl.col("_left_pct") if directions[0] > 0 else 1.0 - pl.col("_left_pct")
                right_score = pl.col("_right_pct") if directions[1] > 0 else 1.0 - pl.col("_right_pct")
                interaction = with_pct.with_columns(
                    pl.min_horizontal(left_score, right_score).alias("_interaction")
                )
                metric = daily_rank_ic(
                    interaction,
                    "_interaction",
                    context.target_column,
                    min_cross_section=budget.min_cross_section,
                )
                if metric is None or int(metric["valid_dates"]) < budget.min_dates:
                    record_trial(context, f"nonlinear_interaction.{left}.{directions[0]}.{right}.{directions[1]}", "failed", {
                        "reason": "insufficient_valid_dates",
                        "metric": metric,
                    })
                    continue
                if not discovery_signal(metric):
                    record_trial(context, f"nonlinear_interaction.{left}.{directions[0]}.{right}.{directions[1]}", "failed", {
                        "reason": "effect_floor",
                        "metric": metric,
                    })
                    continue
                ic = float(metric["ic_mean"])
                if ic < 0:
                    record_trial(context, f"nonlinear_interaction.{left}.{directions[0]}.{right}.{directions[1]}", "failed", {
                        "reason": "negative_interaction",
                        "metric": metric,
                    })
                    continue
                score = ic * float(metric["positive_date_ratio"])
                record_trial(context, f"nonlinear_interaction.{left}.{directions[0]}.{right}.{directions[1]}", "eligible", {
                    "metric": metric,
                    "selection_score": score,
                })
                ranked.append((score, (left, right), directions, metric))
            if trial_count > budget.max_trials:
                break
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        output: list[CandidateSpec] = []
        for score, features, directions, metric in ranked[: budget.max_candidates]:
            output.append(CandidateSpec(
                recipe_id=f"nonlinear_interaction.{features[0]}.{directions[0]}.{features[1]}.{directions[1]}",
                engine_id=self.manifest.engine_id,
                engine_version=self.manifest.version,
                name=f"双条件交互 · {features[0]} x {features[1]}",
                thesis="两个事前状态同时满足时, 训练窗未来净收益排序显著强于单独条件。",
                signal_kind="factor_rank_intersection",
                features=features,
                directions=directions,
                weights=(0.5, 0.5),
                parameters={
                    "entry_score": 80.0,
                    "exit_score": 45.0,
                    "top_rank": 15,
                    "selection_logic": "high_score_intersection_proxy",
                },
                train_evidence={**metric, "selection_score": score, "trials_used": trial_count},
            ))
        return output

    def materialize(
        self,
        candidate: CandidateSpec,
        context: FeatureContext,
    ) -> FrozenSignalSpec:
        del context
        return materialize(candidate)


ENGINE = NonlinearInteractionEngine()
