# P0 ETF 份额流数据结果

结论：**SAMPLE_SUFFICIENT，可以执行预先冻结的开发研究；这不代表策略有效，
严格计数仍为 `0/5`。**

服务器在未读取任何 ETF 价格或未来收益的前提下，审计了 2013–2020 历史份额：

- 点时股票 ETF 356 只，份额覆盖 355 只，覆盖率 99.72%；
- 份额记录 231,854 行、1,957 个不同观察日；
- 2013 年已有 75 只、240 个观察日；2020 年为 337 只、243 个观察日；
- 每一年均超过预注册的 20 只证券和 100 个观察日门槛；
- 代码/日期唯一、份额为正，数据表不含价格和结果字段。

证据：

- `/mnt/data/tick-stock-panel/data/research/p0_etf_share_flow_data_audit.json`
- SHA-256：`7cfa481b24f3105fded043d3860f91658b907e4370ef05b86e3509de04d28487`
- `price_data_read=false`，`future_returns_read=false`。
