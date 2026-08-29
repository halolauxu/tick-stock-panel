"""Concentrated market-residual factor discovery."""
from __future__ import annotations

import polars as pl

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DataQualification,
    DatasetRequirement,
    qualify_manifest_datasets,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates


class PortfolioResidualEngine:
    manifest = AlphaEngineManifest(
        engine_id="portfolio_residual",
        version="1.1.0",
        api_version=ENGINE_API_VERSION,
        name="集中式市场残差因子",
        family="portfolio_residual",
        information_domains=("price_volume", "liquidity", "fundamentals"),
        mechanism_classes=("relative_mispricing",),
        economic_mechanism="剥离市场平均收益后的稳定横截面错价可能在更集中的持仓中兑现",
        discovery_classes=("residual_attribution", "cross_sectional_rank"),
        discovery_method="逐项检验事前因子与未来市场残差收益的排序关系, 并使用更高入选分和更少候选",
        prediction_objects=("market_residual_return",),
        forecast_targets=("3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="当前实现不读取冠军信号, 也不宣称已经计算候选间相关性或组合互补性。",
    )

    def preflight(self, context: DataCatalogContext):
        data = qualify_manifest_datasets(self.manifest, context.datasets)
        return DataQualification(
            False,
            (*data.reasons, "当前模块尚未实现策略残差、候选相关性惩罚或组合互补优化"),
            {**data.observations, "implementation_status": "prototype_not_independent"},
        )

    def discover(self, context, budget):
        frame = context.frame
        if "target_residual_return" in frame.columns:
            frame = frame.with_columns(
                pl.col("target_residual_return").alias(context.target_column)
            )
            context = type(context)(
                frame=frame,
                date_labels=context.date_labels,
                feature_names=context.feature_names,
                target_column=context.target_column,
                asset_type=context.asset_type,
                metadata=context.metadata,
            )
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="portfolio_residual",
            name_prefix="残差因子",
            thesis="该事前特征在训练窗中预测横截面市场残差收益, 采用更集中的入选规则, 且不依赖现有策略样本池。",
            parameters={"entry_score": 75.0, "exit_score": 35.0, "top_rank": 15},
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = PortfolioResidualEngine()
