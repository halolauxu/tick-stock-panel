# P0 同月收益季节性：历史预热数据结果

运行日期：2026-08-31

结论：**DATA_QUALIFIED，可以按已冻结规则启动开发期账户；严格策略计数仍为 `0/5`。**

数据合同：[`p0-return-seasonality-data-contract.md`](p0-return-seasonality-data-contract.md)

服务器数据：`/mnt/data/tick-stock-panel/data/research/return_seasonality_warmup/monthly.parquet`

数据 SHA-256：`8c6dc559d5eb629504f524bf59b572f2ef5e0f9a6a28436701d80278a162aee7`

服务器审计：`/mnt/data/tick-stock-panel/data/research/p0_return_seasonality_warmup_audit.json`

审计 SHA-256：`280b38bdc4cec422c96f51c9b5808a6d33274017cfb4a9eea1d67583979d6ffd`

## 数据结论

- 2007-12 至 2012-12 共 61 / 61 个自然月完整，114,887 行、2,482 只历史 A 股；
- `(symbol, month_end)` 重复键为 0，非法证券代码、非法 OHLC、非正价格和未来复权
  因子均为 0；
- 月线记录全部取得不晚于月末的同源复权因子，最终缺失为 0；
- 9 个月共 11 只月末停牌股票没有月末当日因子，使用该证券月末前最近一条复权记录；
  最大滞后 15 个自然日，实际因子日期和滞后天数均随行保存；
- 没有使用默认复权值、未来因子或后续价格回填；没有计算策略候选、账户或收益；
- 2021 年以后独立验证与压力数据仍未为本机制读取。

该结论只证明 5 年同月形成窗口能够被因果重建。开发期能否达到 50%、相对对照是否
有增量、回撤和真实成交是否合格，仍由已经提交的开发合同独立裁决。
