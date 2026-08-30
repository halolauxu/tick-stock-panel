# P0 中国商品期货真实合约数据 v3 结果

结论：**DATA_QUALIFIED，可以冻结可执行趋势规则；严格策略计数仍为 `0/5`。**

610 个真实合约全部落盘，共 142,520 行日线。34,089 个主力映射日全部匹配同日真实合约行情；590 次主力换月全部同时具有旧合约和新合约真实行情。活跃价格、成交量和持仓量检查全部通过。

连续线只用于构造去换月跳变的信号，账户盈亏、开平仓、换月、容量和费用必须使用本数据域的真实合约价格。

## 证据

- 服务器审计：`/mnt/data/tick-stock-panel/data/research/p0_cn_commodity_futures_contract_data_v3_audit.json`
- SHA-256：`066b2d2273fab2788345508111ee59e6ed44ac047c3ef1698a2464c618f5f413`
- `validation_returns_read=false`。
