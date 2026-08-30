# P0 ETF 跨资产数据 v2 结果

结论：**DATA_QUALIFIED，可以冻结开发集策略规则；这不代表策略有效，严格策略计数仍为 `0/5`。**

v2 没有删除 v1 缺失的退市 ETF，而是用 Tushare `fund_daily` 逐只补齐。最终 421 只冻结主表中 420 只有日线，覆盖率 99.76%；90 只后来退市的 ETF 被保留。420 只有日线证券全部具有复权因子，31.8 万行日线无重复键和非法 OHLCV。

唯一无日线证券为 `161211.SZ`，总体仍超过预注册 95% 门槛。所有补齐行标记 `tushare_gap_fill`，原 TickFlow 行标记 `tickflow`。

## 证据

- 服务器审计：`/mnt/data/tick-stock-panel/data/research/p0_etf_cross_asset_data_v2_audit.json`
- SHA-256：`80879e202ade237d1d664ca29c452cb7ecd6955084778287f506adfce07b8013`
- 日线 318,264 行 / 420 只；补齐 70,140 行 / 107 只；复权 332,756 行 / 420 只。
- `validation_returns_read=false`，尚未读取 2021 年以后的 ETF 收益。
