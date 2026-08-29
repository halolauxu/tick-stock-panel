"""Cross-sectional rank persistence/reversal discovery."""
from __future__ import annotations

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

    def materialize(
        self,
        candidate: CandidateSpec,
        context: FeatureContext,
    ) -> FrozenSignalSpec:
        del context
        return materialize(candidate)


ENGINE = CrossSectionalEngine()
