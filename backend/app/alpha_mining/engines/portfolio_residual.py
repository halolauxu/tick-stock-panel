"""Residual and portfolio-complement Alpha discovery."""
from __future__ import annotations

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DatasetRequirement,
    qualify_manifest_datasets,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates


class PortfolioResidualEngine:
    manifest = AlphaEngineManifest(
        engine_id="portfolio_residual",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="策略残差与组合互补",
        family="portfolio_residual",
        information_domains=("strategy_residual", "portfolio", "price_volume", "liquidity"),
        mechanism_classes=("portfolio_complementarity", "relative_mispricing"),
        economic_mechanism="剥离市场和已知因子后仍可预测的收益残差可提供低相关组合贡献",
        discovery_classes=("residual_attribution",),
        discovery_method="训练窗内以市场残差收益为目标并惩罚候选间相关性",
        prediction_objects=("market_residual_return", "rank_outperformance"),
        forecast_targets=("3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
        ),
        forecast_horizons=(3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="发现阶段不读取冠军信号; 只在公共外测后计算与冠军的互补性。",
    )

    def preflight(self, context: DataCatalogContext):
        return qualify_manifest_datasets(self.manifest, context.datasets)

    def discover(self, context, budget):
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="portfolio_residual",
            name_prefix="残差因子",
            thesis="该事前特征在训练窗中预测横截面市场残差收益, 且不依赖现有策略样本池。",
            parameters={"entry_score": 75.0, "exit_score": 35.0, "top_rank": 15},
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = PortfolioResidualEngine()
