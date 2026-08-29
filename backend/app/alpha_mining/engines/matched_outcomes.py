"""Winner/loser outcome attribution instead of incumbent-strategy filtering."""
from __future__ import annotations

import math

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
from app.alpha_mining.engines._shared import materialize, qualification, record_trial


class MatchedOutcomesEngine:
    manifest = AlphaEngineManifest(
        engine_id="matched_outcomes",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="赢家/输家归因",
        family="outcome_attribution",
        information_domains=("price_volume", "liquidity", "fundamentals"),
        mechanism_classes=("behavioral_underreaction", "relative_mispricing"),
        economic_mechanism="同期赢家与输家的事前差异可揭示被原有策略遗漏或误纳的特征",
        discovery_classes=("matched_outcome_attribution",),
        discovery_method="逐日未来收益分组、标准化效应量与年度方向稳定性",
        prediction_objects=("forward_net_return", "rank_outperformance"),
        forecast_targets=("3d", "5d", "10d", "20d", "60d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(3, 5, 10, 20, 60),
        output_candidate_types=("factor_rank",),
        description="从全体合格股票的赢家与输家出发, 不以新低反转样本池为边界。",
    )

    def preflight(self, context: DataCatalogContext):
        return qualification(context)

    def discover(
        self,
        context: TrainOnlyContext,
        budget: TrialBudget,
    ) -> list[CandidateSpec]:
        target = context.target_column
        base = context.frame.with_columns(
            (
                pl.col(target).rank(method="average", descending=False).over("date")
                / pl.len().over("date")
            ).alias("_outcome_pct")
        )
        ranked: list[tuple[float, str, dict[str, float | int]]] = []
        for feature in context.feature_names[: budget.max_trials]:
            scoped = base.select("date", feature, "_outcome_pct").drop_nulls()
            if scoped.is_empty():
                record_trial(context, f"matched_outcomes.{feature}", "failed", {
                    "reason": "empty_sample",
                })
                continue
            effects = (
                scoped.group_by("date")
                .agg(
                    pl.len().alias("n"),
                    pl.col(feature).filter(pl.col("_outcome_pct") >= 0.8).mean().alias("winner"),
                    pl.col(feature).filter(pl.col("_outcome_pct") <= 0.2).mean().alias("loser"),
                    pl.col(feature).std().alias("std"),
                )
                .filter(
                    (pl.col("n") >= budget.min_cross_section)
                    & pl.col("std").is_not_null()
                    & (pl.col("std") > 0)
                )
                .with_columns(((pl.col("winner") - pl.col("loser")) / pl.col("std")).alias("effect"))
                .filter(pl.col("effect").is_finite())
            )
            if effects.height < budget.min_dates:
                record_trial(context, f"matched_outcomes.{feature}", "failed", {
                    "reason": "insufficient_valid_dates",
                    "valid_dates": effects.height,
                })
                continue
            values = [float(value) for value in effects.get_column("effect").to_list()]
            mean = sum(values) / len(values)
            consistency = sum(value * mean > 0 for value in values) / len(values)
            if abs(mean) < 0.10 or consistency < 0.55:
                record_trial(context, f"matched_outcomes.{feature}", "failed", {
                    "reason": "effect_floor",
                    "standardized_winner_loser_gap": mean,
                    "direction_consistency": consistency,
                })
                continue
            score = abs(mean) * consistency * math.sqrt(len(values))
            record_trial(context, f"matched_outcomes.{feature}", "eligible", {
                "standardized_winner_loser_gap": mean,
                "direction_consistency": consistency,
                "valid_dates": len(values),
                "selection_score": score,
            })
            ranked.append((score, feature, {
                "standardized_winner_loser_gap": mean,
                "direction_consistency": consistency,
                "valid_dates": len(values),
                "selection_score": score,
            }))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            CandidateSpec(
                recipe_id=f"matched_outcomes.{feature}",
                engine_id=self.manifest.engine_id,
                engine_version=self.manifest.version,
                name=f"赢家归因 · {feature}",
                thesis="未来赢家相对输家在训练窗内持续呈现该事前特征差异。",
                signal_kind="factor_rank",
                features=(feature,),
                directions=(1 if float(metric["standardized_winner_loser_gap"]) >= 0 else -1,),
                weights=(1.0,),
                parameters={"entry_score": 75.0, "exit_score": 40.0, "top_rank": 20},
                train_evidence=metric,
            )
            for _, feature, metric in ranked[: budget.max_candidates]
        ]

    def materialize(
        self,
        candidate: CandidateSpec,
        context: FeatureContext,
    ) -> FrozenSignalSpec:
        del context
        return materialize(candidate)


ENGINE = MatchedOutcomesEngine()
