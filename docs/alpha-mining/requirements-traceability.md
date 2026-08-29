# Alpha Mining 需求追踪矩阵

基准：`AM-CHARTER-v1.0`

证据日期：`2026-08-29 Asia/Shanghai`

本表只使用 `PASS` / `FAIL`。`PASS` 表示系统能力及其验收证据成立；不代表当前数据已合格、候选有效或已经找到 Alpha。运行结果状态单列在文末，禁止用工程测试替代。

## Stage 0：宪章与冻结基线

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S0-001 | 正式 Charter | `charter.md` | PASS |
| AM-S0-002 | 固定需求编号 | 本追踪矩阵 | PASS |
| AM-S0-003 | 数据、研究、策略、前向和上线状态 | `stage-0-evidence.md` §6 | PASS |
| AM-S0-004 | 冻结旧 `/mining` 页面和 API 基线 | `stage-0-evidence.md` §3 | PASS |
| AM-S0-005 | 登记既有研究及污染状态 | `stage-0-evidence.md` §4-5 | PASS |
| AM-S0-006 | 无必选现有策略底座 | Charter §2.1 | PASS |
| AM-S0-007 | 冠军只进入公共验证 | Charter §2.2 | PASS |
| AM-S0-008 | 历史范围固定为 2013 至当前 | Charter §2.3 | PASS |
| AM-S0-009 | PIT、T+1 和费用口径 | Charter §2.4 | PASS |
| AM-S0-010 | 受控字段变更须用户确认 | Charter §8 | PASS |
| AM-S0-011 | Stage 0 执行授权 | `stage-0-evidence.md` §7 | PASS |

## Stage 1：新页面与物理隔离

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S1-001 | 独立 `/alpha-mining` 页面 | `frontend/src/router.tsx`、`AlphaMining.tsx` | PASS |
| AM-S1-002 | 独立 `/api/alpha-mining/v1/*` | `backend/app/api/alpha_mining.py`、`test_api.py` | PASS |
| AM-S1-003 | 独立 `alpha-*` 任务 ID | `alpha_mining_manager.py`、`test_manager.py` | PASS |
| AM-S1-004 | 独立运行目录 | `AlphaMiningJobManager` 目录断言 | PASS |
| AM-S1-005 | 默认关闭的功能开关 | `config_store.py`、`test_config.py`、`test_manager.py` | PASS |
| AM-S1-006 | 导航位于旧挖掘下方 | `Layout.tsx`、浏览器验收 | PASS |
| AM-S1-007 | 加载、空、错、禁用、刷新和窄屏 | API 测试、浏览器直接刷新及 390px 验收 | PASS |
| AM-S1-008 | Alpha 删除/损坏不影响旧系统 | `optional_alpha.py`、`test_isolation.py` | PASS |
| AM-S1-009 | 旧 API 与全量回归 | 后端全量测试、`/mining` 浏览器回归 | PASS |
| AM-S1-010 | 不生成虚假研究结论 | 正式准入失败时启动按钮禁用，API 失败闭合 | PASS |

## Stage 2：发现引擎注册系统

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S2-001 | `AlphaDiscoveryEngine` 契约 | `contracts.py` | PASS |
| AM-S2-002 | Manifest 四维分类 | `contracts.py`、`taxonomy.py` | PASS |
| AM-S2-003 | 资产、频率、决策时钟 | Manifest 校验及 `test_builtin_engines.py` | PASS |
| AM-S2-004 | 数据集、PIT、时间字段和最低覆盖 | `DatasetRequirement`、运行前准入 | PASS |
| AM-S2-005 | 期限、输出、参数/产物版本、自动权限 | Manifest 校验及配置测试 | PASS |
| AM-S2-006 | 重复 ID 与错误版本拒绝 | `test_registry.py` | PASS |
| AM-S2-007 | 单引擎异常隔离 | 单引擎失败隔离测试 | PASS |
| AM-S2-008 | 启动后冻结 | 注册表冻结测试 | PASS |
| AM-S2-009 | 新增引擎不改编排器 | 动态新增/删除模块及 API 列表联动测试 | PASS |
| AM-S2-010 | 引擎仅见训练区 | `TrainOnlyContext(slots=True)` 与恶意引擎越界测试 | PASS |
| AM-S2-011 | `ResearchFeatureProvider` | `providers.py`、`test_providers.py` | PASS |
| AM-S2-012 | `ResearchEventProvider` | `providers.py`、`test_providers.py` | PASS |
| AM-S2-013 | `CandidateRenderer` | `providers.py`、冻结候选测试 | PASS |
| AM-S2-014 | 禁止引擎直连供应商 | 注册前 AST 边界审计及恶意导入测试 | PASS |
| AM-S2-015 | 测试引擎全生命周期 | 注册、发现、冻结、验证、证伪及 API 测试 | PASS |
| AM-S2-016 | 删除引擎后恢复且旧系统不受影响 | 动态删除测试、隔离测试 | PASS |

## Stage 3：PIT 数据与研究时钟

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S3-001 | 历史上市/退市/名称/ST/股本股票池 | `data_catalog.py`、边界日期测试 | PASS |
| AM-S3-002 | 历史行业成员关系 | PIT `join_asof` 与区间门控测试 | PASS |
| AM-S3-003 | 财务按公告日门控并保留修订 | 财务公告日前为空测试 | PASS |
| AM-S3-004 | 带发布时间的自然事件表 | `TimestampedEventProvider` | PASS |
| AM-S3-005 | 非交易日映射到下一决策时点 | 周末事件映射测试 | PASS |
| AM-S3-006 | 随机历史日期股票池可复现 | 同输入重复快照/边界测试 | PASS |
| AM-S3-007 | 公告前财务字段为空 | `test_data_catalog.py` | PASS |
| AM-S3-008 | 周末事件仅进入下一交易决策 | `test_providers.py` | PASS |
| AM-S3-009 | 当前概念快照禁入历史回测 | Catalog 固定 `ready=false` | PASS |
| AM-S3-010 | 数据缺失失败闭合 | 运行时最低覆盖校验与页面阻断 | PASS |
| AM-S3-011 | 正式引擎只能通过统一层取数 | 引擎上下文隔离、供应商导入审计 | PASS |

## Stage 4：标签、特征与不可变账本

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S4-001 | 1/3/5/10/20/60 日净收益标签 | `labels.py`、`test_labels.py` | PASS |
| AM-S4-002 | 残差收益标签 | 基准收益残差测试 | PASS |
| AM-S4-003 | MFE、MAE 标签 | 标签数值及未来窗口测试 | PASS |
| AM-S4-004 | 跳空和不可成交风险标签 | 涨跌停/停牌/缺失开盘标签测试 | PASS |
| AM-S4-005 | 受控特征与事件序列表达 | Provider 字段白名单、事件聚合 | PASS |
| AM-S4-006 | 不可变实验合同 | `AlphaEvidenceStore.create_experiment` | PASS |
| AM-S4-007 | 数据/代码/特征/股票池/成本指纹 | 经理写入完整合同及内容哈希 | PASS |
| AM-S4-008 | 所有失败尝试计入预算 | 运行时 `trial_ledger` 及失败试验测试 | PASS |
| AM-S4-009 | 冻结候选不可修改 | write-once 候选与冲突拒绝测试 | PASS |
| AM-S4-010 | 同指纹可复现，变更生成新实验 | 内容寻址及冲突测试 | PASS |
| AM-S4-011 | 股票/日期/数据/代码链路追溯 | 候选研究描述、实验哈希、逐折和交易证据 | PASS |

## Stage 5：首批发现引擎

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S5-001 | 全市场截面因子 | `cross_sectional.py` | PASS |
| AM-S5-002 | 市场与板块时序 | 当前仅实现市场残差截面排序，未做市场/行业状态分层 | FAIL |
| AM-S5-003 | 事件与事件序列 | `event_sequence.py` | PASS |
| AM-S5-004 | 行业/概念/产业链扩散 | `network_diffusion.py` | PASS |
| AM-S5-005 | 财务变化与预期差 | `financial_revision.py` | PASS |
| AM-S5-006 | 赢家/输家匹配归因 | `matched_outcomes.py` | PASS |
| AM-S5-007 | 策略残差与组合互补 | 当前不读取策略残差，也未做相关性惩罚或组合互补优化 | FAIL |
| AM-S5-008 | 非线性交互 | `nonlinear_interaction.py` | PASS |
| AM-S5-009 | 独立预算与全市场发现 | `TrialBudget`、运行时逐引擎预算 | PASS |
| AM-S5-010 | 至少一个引擎不引用现有策略 | 七个非残差引擎均从合格股票池和训练特征出发 | PASS |
| AM-S5-011 | 已知 Alpha 真阳性测试 | 现有测试复用同一强排序因子，不是逐引擎机制真阳性 | FAIL |
| AM-S5-012 | 随机噪声假阳性控制 | 八引擎噪声拒绝测试 | PASS |
| AM-S5-013 | 缺数据明确停止 | Manifest 数据要求、引擎预检、API 阻断 | PASS |

## Stage 6：公共验证与冠军系统

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S6-001 | 嵌套逐折发现 | `runtime.py` | PASS |
| AM-S6-002 | 外测折对引擎不可见 | `TrainOnlyContext` 恶意越界测试 | PASS |
| AM-S6-003 | 拼接样本外净值 | 逐折 OOS 交易合并与指标计算 | PASS |
| AM-S6-004 | 多重试验惩罚 | 试验总数进入惩罚项和账本 | PASS |
| AM-S6-005 | 统一成交和成本 | 公共矩阵回测器、T+1、费用/滑点请求合同 | PASS |
| AM-S6-006 | 双倍成本压力 | `double_cost_return` 实际回测 | PASS |
| AM-S6-007 | 延迟成交压力 | `entry_delay_days` 真正移位及回测测试 | PASS |
| AM-S6-008 | 参数扰动压力 | 三组冻结参数扰动及最差结果 | PASS |
| AM-S6-009 | 容量和集中度压力 | 双倍持仓、股票/行业/交易贡献集中度 | PASS |
| AM-S6-010 | 动态冠军榜 | `champion.py`、API、页面 | PASS |
| AM-S6-011 | 状态只能由证据推进 | 哈希事件链及非法跳转测试 | PASS |
| AM-S6-012 | 未过门槛禁止发布 | 发布服务状态门控与测试 | PASS |

## Stage 7：八区产品页

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S7-001 | 研究总览 | 页面区域 1 | PASS |
| AM-S7-002 | 四维机会地图及扩展插槽 | 页面区域 2 | PASS |
| AM-S7-003 | 信息与数据覆盖 | 页面区域 3 | PASS |
| AM-S7-004 | 发现引擎 | 页面区域 4，动态读取注册表 | PASS |
| AM-S7-005 | 实验账本 | 页面区域 5 | PASS |
| AM-S7-006 | 候选证据钻取 | 已改为因子、规则、撮合、覆盖和裁决说明；待线上浏览器复验 | FAIL |
| AM-S7-007 | 完整冠军挑战 | 页面区域 7 | PASS |
| AM-S7-008 | 前向模拟 | 页面区域 8 | PASS |
| AM-S7-009 | 加载/空/错/取消/重连/窄屏 | 本轮修改后尚未完成线上桌面与窄屏复验 | FAIL |
| AM-S7-010 | 旧挖掘浏览器无回归 | `/mining` 页面浏览器验收 | PASS |

## Stage 8：前向模拟闭环

| ID | 需求 | 证据 | 结论 |
|---|---|---|---|
| AM-S8-001 | 候选独立模拟账户 | `AlphaShadowService.start` | PASS |
| AM-S8-002 | 信号/订单/成交/持仓/收益对账 | `PaperLedger` 投影和 reconcile | PASS |
| AM-S8-003 | 实际滑点和不可成交统计 | 前向评估及信号跳过生命周期 | PASS |
| AM-S8-004 | 因子衰减检测 | 冻结入场分数与实现收益 Spearman | PASS |
| AM-S8-005 | 行情失效和漂移检测 | 漂移组合门槛 | PASS |
| AM-S8-006 | 漂移时暂停且不自动调参 | `evaluate_and_advance` 失败路径 | PASS |
| AM-S8-007 | 研究/回测/模拟订单同日一致 | 冻结合同、`open_t+1` 和信号订单闭环测试 | PASS |
| AM-S8-008 | 无成交不伪造收益 | `no_synthetic_profit` 测试 | PASS |
| AM-S8-009 | 前向失败保留证据且不回改 | write-once forward evidence 与拒绝状态 | PASS |
| AM-S8-010 | 预注册天数和交易样本门槛 | 配置门槛及不足时 `qualified=false` | PASS |

## Stage 9：冠军晋级

系统已实现所有门槛的计算、阻断、显式发布、回滚和动态冠军账本；但下列 11 项是对真实候选的结果验收。线上近一年快速研究冻结的 6 个候选已全部被证伪，没有正式研究候选，因此必须判 `FAIL`，不能用任务成功、候选数量或合成测试冒充 Alpha。

| ID | 需求 | 当前真实证据 | 结论 |
|---|---|---|---|
| AM-S9-001 | 13 年拼接 OOS 收益高于冠军 | 尚未完成完整历史正式研究；快速研究候选全部为负 | FAIL |
| AM-S9-002 | Sharpe 达到冠军+0.20且不低于0.8 | 无正式候选 | FAIL |
| AM-S9-003 | 回撤不超过25%且不劣于冠军 | 无正式候选 | FAIL |
| AM-S9-004 | 正半年比例至少65% | 无正式候选 | FAIL |
| AM-S9-005 | 跑赢冠军半年窗口至少60% | 无正式候选 | FAIL |
| AM-S9-006 | 最近一年跑赢冠军 | 无正式候选 | FAIL |
| AM-S9-007 | 最近三个月绝对收益为正 | 无正式候选 | FAIL |
| AM-S9-008 | 两倍成本仍为正 | 无正式候选 | FAIL |
| AM-S9-009 | 延迟与参数扰动不失效 | 无正式候选 | FAIL |
| AM-S9-010 | 收益集中度受控 | 无正式候选 | FAIL |
| AM-S9-011 | 前向模拟通过 | 尚无满足观察期的真实前向账户 | FAIL |
| AM-S9-012 | 自动挑战、显式发布和冠军晋级 | 状态门控、create-only 发布、回滚和冠军账本测试 | PASS |

## 全局防偏离治理

| ID | 规则 | 证据 | 结论 |
|---|---|---|---|
| AM-GOV-001 | 代码、测试、证据引用需求 ID | 核心模块及专项测试文件头 | PASS |
| AM-GOV-002 | 前一阶段不通过不得宣称后一阶段完成 | `stage-acceptance.md` 二元结论 | PASS |
| AM-GOV-003 | 不以页面/测试/任务数宣称 Alpha 成功 | Stage 9 真实结果保持 FAIL | PASS |
| AM-GOV-004 | 受控字段变更须用户确认 | Charter §8 | PASS |
| AM-GOV-005 | 每阶段固定五项证据 | `stage-acceptance.md` | PASS |
| AM-GOV-006 | 阶段只用 PASS/FAIL | 本矩阵及 `stage-acceptance.md` | PASS |

## 当前数据与研究结果状态

| 状态 | 证据 | 结论 |
|---|---|---|
| 工程系统能力（Stage 0-8） | 专项测试、全量回归、构建和浏览器验收 | PASS |
| 本地正式数据准入 | 245 日分区；历史股票池、股本、财务、行业和事件历史缺失 | FAIL |
| 正式研究候选 | 近一年快速研究冻结 6 个候选并全部证伪；无候选通过正式准入 | FAIL |
| 前向通过 | 无正式研究候选 | FAIL |
| 找到稳定优于当前冠军的 Alpha | 无真实证据 | FAIL |
