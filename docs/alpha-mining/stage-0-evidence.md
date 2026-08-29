# Stage 0 验收证据

阶段：`STAGE-0`

证据日期：`2026-08-29 Asia/Shanghai`

当前结论：`PASS`

用户已明确要求恢复宪章约束，并在 `2026-08-29` 要求一次性完成验收表全部工程缺口；该指令作为 Stage 0 执行授权。后续阶段仍必须分别给出二元结论。

## 1. 实际改动

- 新增 `docs/alpha-mining/charter.md`，固化目标、边界、完成状态、阶段门禁、冠军门槛和变更控制。
- 新增 `docs/alpha-mining/requirements-traceability.md`，为Stage 0至Stage 9及全局治理规则逐条编号。
- 新增本证据文件，冻结旧挖掘基线、历史研究污染状态和Stage 0验收结论。

## 2. 未改动范围

- 未修改旧 `/mining` 页面、API、候选、调度和运行产物。
- 未继续修改现有Alpha原型的页面、API、引擎、验证器或状态机。
- 未修改、删除或重跑 `research/` 下的历史研究结果。
- 未修改或删除用户工作区中已有的Tushare、历史股票池、股本及研究脚本改动。
- 未提交、未部署、未发布策略、未创建模拟账户。

## 3. 旧挖掘冻结基线

### 页面和导航

- 页面：`/mining`
- 前端组件：`frontend/src/pages/Mining.tsx`
- 导航：`frontend/src/components/Layout.tsx`
- 产品合同：`docs/mining.md`

### API

基准前缀：`/api/backtest/mining`

- `GET /availability`
- `GET /runs`
- `POST /runs`
- `GET /runs/{run_id}`
- `POST /runs/{run_id}/cancel`
- `GET /runs/{run_id}/result`
- `GET /runs/{run_id}/events`
- `POST /runs/{run_id}/candidates/{signature}/promote`
- `POST /runs/{run_id}/candidates/{signature}/publish`
- `GET /config`
- `PATCH /config`

### 任务与产物

- 旧任务ID命名空间保持不变。
- 旧运行根目录：`data/research/mining/runs/`。
- 本地盘点到2个旧运行：
  - `764ce9f1890043559173340bce69d528`
  - `d3ae8e0be09442089483732bd860d417`
- 两个运行均为 `exploratory`，状态为 `succeeded_with_budget_exhausted`，只能作为旧功能基线，不能作为正式Alpha证据。

任何后续阶段若改变以上合同，必须先取得用户明确确认并更新需求编号。

## 4. 已存在Alpha原型的隔离状态

下列实现早于本次正式阶段门禁，统一标记为 `UNACCEPTED_PROTOTYPE`：

- `backend/app/alpha_mining/`
- `backend/app/api/alpha_mining.py`
- `backend/app/services/alpha_mining_manager.py`
- `backend/tests/alpha_mining/`
- `frontend/src/pages/AlphaMining.tsx`
- 旧系统接线文件中的Alpha相关改动

本地Alpha运行：

- ID：`alpha-5ccad00515014760aa080f23`
- 档位：`exploratory`
- 外层样本外窗口：1
- 试验数：6
- 三个引擎候选状态：全部 `rejected`

该运行只能证明原型链路曾执行，不能证明Stage 1至Stage 9任何阶段通过，也不能作为Alpha候选或冠军证据。

## 5. 历史研究结果与污染状态

`research/results/` 当前包含40个JSON结果和3个Markdown报告。由于这些研究发生在正式宪章、需求编号、统一试验预算和阶段门禁建立之前，统一标记：

`PRE_CHARTER_RESEARCH / REFERENCE_ONLY / NOT_PROMOTABLE`

主要研究族包括：

- 新低反转及修复研究：`reversal_discovery_stage1` 至 `stage12`、OOS补充、legacy audit及first-principles报告。
- 独立Alpha家族：`independent_alpha_families_stage1` 至 `stage3`、final及研究报告。
- 近期赢家与新收益池：winner pool、recent winner feature、accumulation、barbell、core-satellite。
- 情绪与行情适配：sentiment long、halfyear、recent-period、fix study。
- 次级启动与执行稳健性：secondary ignition、fold validation、execution robustness、strong market。
- 线上同口径复验：exact online year benchmark、stability validation。

污染含义不是结果一定错误，而是缺少至少一项正式准入证据：预注册需求ID、完整试验次数、冻结数据/代码/股票池/成本指纹、统一PIT数据资格、未见外层结果证明或前向验证。因此：

- 可以用于提出新假设、失败模式和数据需求；
- 不得直接作为研究候选、挑战者或冠军；
- 不得通过补写文档追认成正式样本外结果；
- 若要重新检验，必须在Stage 3和Stage 4通过后创建全新实验合同。

## 6. 状态定义

### 数据状态

| 状态 | 定义 |
|---|---|
| `DATA_UNREGISTERED` | 未进入Alpha数据目录。 |
| `DATA_GAP` | 覆盖或字段不足，禁止正式研究。 |
| `PIT_UNVERIFIED` | 有历史数据，但尚未证明发布时间和历史可见性。 |
| `DATA_QUALIFIED` | 指定版本通过覆盖、PIT、完整性和未来信息审计。 |
| `DATA_REVOKED` | 后续发现污染，相关实验全部失效。 |

### 实验和候选状态

| 状态 | 定义 |
|---|---|
| `DRAFT` | 假设尚未冻结。 |
| `REGISTERED` | 实验合同及预算已登记。 |
| `DATA_READY` | 所需数据版本全部合格。 |
| `DISCOVERY` | 仅在训练区发现候选。 |
| `FROZEN` | 候选定义不可修改。 |
| `OUTER_EVALUATED` | 已完成外层样本外评价。 |
| `REJECTED` | 任一硬门槛失败，永久保留证据。 |
| `RESEARCH_CANDIDATE` | 历史和压力门槛全部通过。 |
| `SHADOW` | 正在执行前向模拟。 |
| `CHALLENGER` | 前向模拟通过，等待冠军比较。 |
| `CHAMPION` | 全部门槛通过并完成显式晋级。 |

### 工程和上线状态

| 状态 | 定义 |
|---|---|
| `UNACCEPTED_PROTOTYPE` | 存在代码或页面，但所属阶段未通过。 |
| `STAGE_PENDING` | 阶段开发或证据未完成。 |
| `STAGE_FAILED` | 至少一个硬验收项失败。 |
| `STAGE_PASS` | 全部硬验收项通过且用户已确认。 |
| `RELEASE_BLOCKED` | 未满足发布前置条件。 |
| `SHADOW_ONLY` | 只允许前向模拟，不允许真实资金。 |
| `RELEASED` | 通过显式发布流程，但仍不等于未来收益保证。 |

## 7. Stage 0 二元验收

| ID | 硬验收项 | 自检证据 | 结论 |
|---|---|---|---|
| AM-S0-001 | Charter存在 | `charter.md` | PASS |
| AM-S0-002 | 需求已编号 | `requirements-traceability.md` | PASS |
| AM-S0-003 | 状态已定义 | 本文件§6 | PASS |
| AM-S0-004 | 旧挖掘基线已冻结 | 本文件§3 | PASS |
| AM-S0-005 | 历史研究及污染状态已登记 | 本文件§5 | PASS |
| AM-S0-006 | 无必选策略底座 | Charter §2.1 | PASS |
| AM-S0-007 | 冠军只参与公共比较 | Charter §2.2 | PASS |
| AM-S0-008 | 历史范围固定 | Charter §2.3 | PASS |
| AM-S0-009 | PIT、T+1、费用口径固定 | Charter §2.4 | PASS |
| AM-S0-010 | 变更必须用户确认 | Charter §8 | PASS |
| AM-S0-011 | 用户确认Stage 0 | 用户明确恢复宪章约束并要求执行全部缺口 | PASS |

Stage 0总体结论：`PASS`。

## 8. 测试与页面证据

Stage 0只新增治理文档，没有修改运行代码，因此本阶段不以代码测试或页面截图代替验收。旧原型此前的测试和浏览器结果仅登记为背景信息，不计入Stage 0通过条件。

## 9. 剩余风险

- 现有未验收Alpha原型仍在工作区中，必须通过后续各阶段逐项接管或删除，不能默认继承。
- 历史研究结果数量较多，当前按研究族登记；正式重跑前仍需为具体假设建立独立实验合同。
- Stage 1 已重新建立可归因基线并修复数据完整性流程的回归；最终全量结果见 `stage-acceptance.md`。
