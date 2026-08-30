# P0 中国商品期货盘前限价数据 v4 结果

结论：**DATA_PERMISSION_BLOCKED。**

Tushare `ft_limit` 是盘前发布的期货合约涨跌停价与最低保证金率接口，但服务器账号调用返回权限拒绝，未生成限价数据文件。研究撮合器已增加全天一字锁板的保守拒单代理；代理不能替代完整盘前限价数据，任何期货候选即使收益过门槛，也不得据此计入严格策略。

证据：服务器 `/mnt/data/tick-stock-panel/data/research/p0_cn_commodity_futures_limit_data_v4_audit.json`，SHA-256 `86c5ecc6c7142cb9b5f7a0b40f6d4302fd28061847f35a58748d896c08dfa3de`。
