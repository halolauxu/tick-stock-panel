import type {
  AlphaCandidateResult,
  AlphaEngineManifest,
  AlphaGateEvidence,
  AlphaRun,
  FactorColumn,
  MiningBudgetProfile,
} from '@/lib/api'

export type AlphaWorkbenchView =
  | 'overview'
  | 'discovery'
  | 'oos'
  | 'market'
  | 'robustness'
  | 'candidates'
  | 'forward'
  | 'audit'

export interface AlphaWorkbenchDraft {
  assetType: 'stock' | 'etf'
  profile: MiningBudgetProfile
  horizon: 1 | 3 | 5 | 10 | 20 | 60
  start: string
  end: string
  datePreset: '1y' | '3y' | 'all' | 'custom'
  factorNames: string[]
  engineIds: string[]
  commissionBps: string
  stampTaxBps: string
  slippageBps: string
  maxPositions: string
}

export const ACTIVE_RUN_STATUSES = new Set(['queued', 'running', 'cancelling'])
export const SUCCESS_RUN_STATUSES = new Set(['succeeded', 'succeeded_with_budget_exhausted'])

export const PROFILE_META: Record<MiningBudgetProfile, { title: string; description: string; trials: number; candidates: number }> = {
  exploratory: { title: '快速研究', description: '最近一年快速发现与证伪，不能直接晋级', trials: 24, candidates: 2 },
  balanced: { title: '标准研究', description: '更长历史、多窗口样本外和完整压力测试', trials: 64, candidates: 4 },
  strict: { title: '严格研究', description: '全部可用历史、最多窗口与最大验证预算', trials: 128, candidates: 8 },
}

export const VIEW_META: { id: AlphaWorkbenchView; label: string }[] = [
  { id: 'overview', label: '结果总览' },
  { id: 'discovery', label: '发现证据' },
  { id: 'oos', label: '样本外' },
  { id: 'market', label: '市场归因' },
  { id: 'robustness', label: '稳健性' },
  { id: 'candidates', label: '候选方案' },
  { id: 'forward', label: '前向模拟' },
  { id: 'audit', label: '审计' },
]

export const GATE_LABELS: Record<string, string> = {
  return_vs_champion: '拼接样本外净收益', sharpe: '样本外夏普', drawdown: '最大回撤',
  positive_half_years: '正收益半年窗口', beat_champion_windows: '半年窗口稳定性', recent_year: '最近一年收益',
  recent_quarter: '最近三个月收益', double_cost: '双倍成本', delay: '延迟成交',
  parameter_perturbation: '参数扰动', capacity: '持仓容量', concentration: '收益集中度', forward_shadow: '前向模拟',
}

export const DATASET_LABELS: Record<string, { title: string; description: string }> = {
  daily_enriched: { title: '完整日线与衍生指标', description: '行情、成交量和技术指标' },
  historical_universe: { title: '历史股票池', description: '上市、退市、名称、风险警示和股本' },
  financial_pit: { title: '公告时点财务数据', description: '只使用当时已经公开的财务信息' },
  industry_pit: { title: '历史行业归属', description: '按当时有效的行业关系研究' },
  event_history: { title: '公司事件历史', description: '包含真实发布时间和决策时钟' },
  concept_snapshot: { title: '当前概念快照', description: '仅供当前观察，不进入历史研究' },
}

export const TERM_LABELS: Record<string, string> = {
  price_volume: '价格与成交量', liquidity: '流动性', fundamentals: '财务基本面', market_regime: '市场环境',
  industry: '行业关系', concept_network: '概念传播网络', corporate_event: '公司事件', event_sequence: '事件演化',
  strategy_residual: '策略残差', portfolio: '组合层面', auction_microstructure: '集合竞价与微观结构',
  holder_supply: '股东与筹码供给', event_text: '公告与文本', cross_asset: '跨资产联动', risk_compensation: '风险补偿',
  behavioral_underreaction: '反应不足', behavioral_overreaction: '过度反应', liquidity_pressure: '流动性压力',
  information_diffusion: '信息扩散', expectation_revision: '预期修正', crowding_unwind: '拥挤交易退潮',
  structural_flow: '结构性资金流', relative_mispricing: '相对错价', portfolio_complementarity: '组合互补',
  cross_sectional_rank: '横截面排序', conditional_time_series: '条件时序', matched_outcome_attribution: '匹配样本归因',
  event_study: '事件研究', sequence_pattern: '序列模式', network_diffusion: '网络扩散', revision_surprise: '预期修订意外',
  residual_attribution: '残差归因', nonlinear_interaction: '非线性交互', relative_value: '相对价值',
  forward_net_return: '未来净收益', market_residual_return: '相对市场超额收益', mfe: '最大有利波动',
  mae: '最大不利波动', gap_risk: '跳空风险', untradable_risk: '不可成交风险', rank_outperformance: '排序超额表现',
}

export function isWorkbenchView(value: string | null): value is AlphaWorkbenchView {
  return VIEW_META.some(item => item.id === value)
}

export function yearsBefore(isoDate: string, years: number) {
  const [year, month, day] = isoDate.split('-').map(Number)
  const value = new Date(Date.UTC(year, month - 1, day))
  value.setUTCFullYear(value.getUTCFullYear() - years)
  return value.toISOString().slice(0, 10)
}

export function fmtPct(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%` : '—'
}

export function fmtNumber(value: number | null | undefined, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export function statusLabel(status?: string | null) {
  return ({
    queued: '排队中', running: '研究中', cancelling: '取消中', succeeded: '研究结束',
    succeeded_with_budget_exhausted: '研究结束（预算用完）', failed: '运行失败', cancelled: '已取消', interrupted: '进程中断',
    outer_evaluated: '样本外已评估', validation_candidate: '待严格验证', rejected: '已证伪', research_candidate: '严格验证通过', shadow: '前向模拟中',
    challenger: '挑战者', champion: '正式冠军', retired: '历史冠军', frozen: '已冻结', discovery: '发现中', data_ready: '数据就绪',
    registered: '已登记', draft: '草稿',
  } as Record<string, string>)[status || ''] || '未开始'
}

export function translateReason(reason: string) {
  const exact: Record<string, string> = {
    '缺少带发布时间的历史事件表': '缺少带真实发布时间的公司事件历史',
    '当前概念成员表是快照; 禁止进入历史正式研究': '当前概念数据只有最新快照，不能用于历史研究',
  }
  if (exact[reason]) return exact[reason]
  return reason.replaceAll('PIT', '时点防泄漏').replaceAll('ResearchEventProvider', '事件数据接口')
    .replaceAll('event_history', '公司事件历史').replaceAll('daily_enriched', '完整日线数据')
    .replaceAll('historical_universe', '历史股票池').replaceAll('financial_pit', '公告时点财务数据')
    .replaceAll('industry_pit', '历史行业归属').replaceAll('concept_snapshot', '当前概念快照')
}

export function translateRunError(error?: string | null) {
  if (!error) return '没有记录具体失败原因'
  if (error.includes("Alpha labels require columns: ['high', 'low', 'open']")) return '研究面板缺少开盘价、最高价和最低价，无法生成未来收益与风险标签'
  const firstLine = error.split('\n', 1)[0]
  return /[一-鿿]/.test(firstLine) ? translateReason(firstLine) : '研究任务在执行阶段异常退出，技术详情已保留在审计记录中'
}

export function termLabel(value: string) {
  return TERM_LABELS[value] || value.replaceAll('_', ' ')
}

export function factorRule(candidate: AlphaCandidateResult, factorMap: Map<string, FactorColumn>) {
  const frozen = candidate.frozen_candidate
  if (!frozen) return '训练或隐藏内选没有形成可回测方案'
  return frozen.features.map((feature, index) => `${factorMap.get(feature)?.label || feature}（${(frozen.directions[index] || 1) > 0 ? '高值优先' : '低值优先'}）`).join(' + ')
}

export function successfulOosRange(candidate: AlphaCandidateResult) {
  return candidate.metrics.oos_start && candidate.metrics.oos_end ? { start: candidate.metrics.oos_start, end: candidate.metrics.oos_end } : null
}

export function hasPeriodCoverage(candidate: AlphaCandidateResult, period: 'year' | 'quarter') {
  const explicit = period === 'year' ? candidate.metrics.recent_1y_available : candidate.metrics.recent_3m_available
  if (typeof explicit === 'boolean') return explicit
  const range = successfulOosRange(candidate)
  return !!range && (Date.parse(range.end) - Date.parse(range.start)) / 86_400_000 >= (period === 'year' ? 365 : 92)
}

export function gateCounts(candidate: AlphaCandidateResult) {
  return candidate.gates.reduce((output, gate) => {
    const unavailable = (gate.id === 'recent_year' && !hasPeriodCoverage(candidate, 'year')) || (gate.id === 'recent_quarter' && !hasPeriodCoverage(candidate, 'quarter'))
    output[unavailable ? 'pending' : gate.status] += 1
    return output
  }, { passed: 0, failed: 0, pending: 0 })
}

export function formatGateActual(gate: AlphaGateEvidence) {
  if (gate.actual == null) return '证据不足'
  if (typeof gate.actual === 'boolean') return gate.actual ? '通过' : '未通过'
  if (typeof gate.actual === 'number') return gate.id === 'sharpe' ? fmtNumber(gate.actual) : fmtPct(gate.actual)
  return String(gate.actual)
}

export function formatGateRequired(gate: AlphaGateEvidence) {
  const required = gate.required
  if (required == null) return '—'
  if (typeof required === 'boolean') return required ? '必须通过' : '不要求通过'
  if (typeof required === 'number') {
    return gate.id === 'sharpe' ? fmtNumber(required) : fmtPct(required)
  }
  if (typeof required === 'string') return required
  if (typeof required === 'object') {
    const contract = required as { comparison?: unknown; minimum?: unknown }
    const comparison = typeof contract.comparison === 'string' ? contract.comparison : '统一比较口径'
    const minimum = typeof contract.minimum === 'number' ? `，最低${fmtPct(contract.minimum)}` : ''
    return `${comparison}${minimum}`
  }
  return '—'
}

export function engineSearchText(engine: AlphaEngineManifest) {
  return [engine.name, engine.description, engine.discovery_method, ...engine.information_domains, ...engine.mechanism_classes].join(' ').toLowerCase()
}

export function runLabel(run: AlphaRun) {
  const range = [run.request.start, run.request.end].filter(Boolean).join(' 至 ')
  return `${range || '未限定区间'} · ${statusLabel(run.status)}`
}
