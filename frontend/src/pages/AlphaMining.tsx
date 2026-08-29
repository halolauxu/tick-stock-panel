// Requirements: AM-S7-001 through AM-S7-010.
import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Database,
  Info,
  LoaderCircle,
  Pause,
  Play,
  ShieldCheck,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { toast } from '@/components/Toast'
import {
  api,
  type AlphaCandidateResult,
  type AlphaConfig,
  type AlphaMiningRequest,
  type AlphaResult,
  type AlphaRun,
  type MiningBudgetProfile,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { cn } from '@/lib/cn'

const ACTIVE = new Set(['queued', 'running', 'cancelling'])
const SUCCESS = new Set(['succeeded', 'succeeded_with_budget_exhausted'])
const INPUT = 'h-8 w-full rounded-input border border-border bg-surface px-2 text-xs text-foreground outline-none focus:border-accent'
const CARD = 'rounded-card border border-border bg-surface'

const DIMENSION_META: Record<string, { title: string; description: string }> = {
  information_domain: { title: '信息来源', description: '从哪些市场与公司信息中寻找机会' },
  mechanism: { title: '收益形成机制', description: '为什么这个现象可能产生超额收益' },
  discovery: { title: '发现方法', description: '系统用什么研究方法寻找规律' },
  prediction_object: { title: '预测目标', description: '候选因子最终要预测什么' },
}

const TERM_LABELS: Record<string, string> = {
  price_volume: '价格与成交量',
  liquidity: '流动性',
  fundamentals: '财务基本面',
  market_regime: '市场环境',
  industry: '行业关系',
  concept_network: '概念传播网络',
  corporate_event: '公司事件',
  event_sequence: '事件演化',
  strategy_residual: '现有策略残差',
  portfolio: '组合层面',
  auction_microstructure: '集合竞价与微观结构',
  holder_supply: '股东与筹码供给',
  event_text: '公告与文本',
  cross_asset: '跨资产联动',
  risk_compensation: '风险补偿',
  behavioral_underreaction: '反应不足',
  behavioral_overreaction: '过度反应',
  liquidity_pressure: '流动性压力',
  information_diffusion: '信息扩散',
  expectation_revision: '预期修正',
  crowding_unwind: '拥挤交易退潮',
  structural_flow: '结构性资金流',
  relative_mispricing: '相对错价',
  portfolio_complementarity: '组合互补',
  cross_sectional_rank: '横截面排序',
  conditional_time_series: '条件时序',
  matched_outcome_attribution: '匹配样本归因',
  event_study: '事件研究',
  sequence_pattern: '序列模式',
  network_diffusion: '网络扩散',
  revision_surprise: '预期修订意外',
  residual_attribution: '残差归因',
  nonlinear_interaction: '非线性交互',
  relative_value: '相对价值',
  forward_net_return: '未来净收益',
  market_residual_return: '相对市场超额收益',
  mfe: '持有期最大有利波动',
  mae: '持有期最大不利波动',
  gap_risk: '跳空风险',
  untradable_risk: '不可成交风险',
  rank_outperformance: '排序超额表现',
}

const DATASET_LABELS: Record<string, { title: string; description: string }> = {
  daily_enriched: { title: '完整日线与衍生指标', description: '行情、成交量和技术指标' },
  historical_universe: { title: '历史股票池', description: '上市、退市、名称和风险警示状态' },
  financial_pit: { title: '公告时点财务数据', description: '只使用当时已经公开的财务信息' },
  industry_pit: { title: '历史行业归属', description: '按当时有效的行业关系研究' },
  event_history: { title: '公司事件历史', description: '必须包含真实发布时间' },
  concept_snapshot: { title: '当前概念快照', description: '仅供当前观察，不进入历史研究' },
}

const PROFILE_META: Record<MiningBudgetProfile, { title: string; description: string; budget: string }> = {
  exploratory: {
    title: '近1年快速研究',
    description: '快速发现与证伪，结果不能直接晋级',
    budget: '每引擎最多24次尝试 / 2个候选',
  },
  balanced: {
    title: '标准研究',
    description: '更长历史与多窗口样本外验证',
    budget: '每引擎最多64次尝试 / 4个候选',
  },
  strict: {
    title: '严格研究',
    description: '最长历史、更多窗口和最大试验预算',
    budget: '每引擎最多128次尝试 / 8个候选',
  },
}

const EVIDENCE_KEY_LABELS: Record<string, string> = {
  recipe_id: '方案编号',
  engine_id: '发现引擎',
  engine_version: '引擎版本',
  name: '名称',
  thesis: '研究假设',
  signal_kind: '信号类型',
  features: '使用因子',
  directions: '因子方向',
  weights: '因子权重',
  parameters: '参数',
  train_evidence: '训练期证据',
  metrics: '指标',
  gates: '门槛',
  status: '状态',
  passed: '通过',
  failed: '失败',
  pending: '待验证',
  stitched_oos_return: '拼接样本外收益',
  stitched_oos_sharpe: '拼接样本外夏普',
  max_drawdown: '最大回撤',
  recent_1y_return: '近一年收益',
  recent_3m_return: '近三个月收益',
  double_cost_return: '双倍成本收益',
  delayed_entry_return: '延迟成交收益',
  worst_parameter_return: '参数扰动最差收益',
}

function fmtPct(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
    : '—'
}

function fmtNumber(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—'
}

function statusLabel(status?: string | null) {
  return ({
    queued: '排队中',
    running: '运行中',
    cancelling: '取消中',
    succeeded: '已完成',
    succeeded_with_budget_exhausted: '预算已用完',
    failed: '失败',
    cancelled: '已取消',
    interrupted: '进程中断',
    outer_evaluated: '样本外已评估',
    rejected: '已证伪',
    research_candidate: '研究候选',
    shadow: '前向模拟',
    challenger: '挑战者',
    champion: '冠军',
    frozen: '已冻结',
    discovery: '发现中',
    data_ready: '数据就绪',
    registered: '已登记',
    draft: '草稿',
  } as Record<string, string>)[status || ''] || '未开始'
}

function termLabel(value: string) {
  return TERM_LABELS[value] || value.replaceAll('_', ' ')
}

function strategyLabel(strategyId?: string | null) {
  return strategyId || '—'
}

function translateReason(reason: string) {
  const exact: Record<string, string> = {
    '缺少带发布时间的历史事件表': '缺少带真实发布时间的公司事件历史',
    '当前概念成员表是快照; 禁止进入历史正式研究': '当前概念数据只有最新快照，不能用于历史研究',
  }
  if (exact[reason]) return exact[reason]
  return reason
    .replaceAll('PIT', '时点防泄漏')
    .replaceAll('ResearchEventProvider', '事件数据接口')
    .replaceAll('event_*', '事件类')
    .replaceAll('event_history', '公司事件历史')
    .replaceAll('daily_enriched', '完整日线数据')
    .replaceAll('historical_universe', '历史股票池')
    .replaceAll('financial_pit', '公告时点财务数据')
    .replaceAll('industry_pit', '历史行业归属')
    .replaceAll('concept_snapshot', '当前概念快照')
}

function translateRunError(error?: string | null) {
  if (!error) return '未记录失败原因'
  if (error.includes("Alpha labels require columns: ['high', 'low', 'open']")) {
    return '研究面板缺少开盘价、最高价和最低价，无法生成未来收益与风险标签'
  }
  const firstLine = error.split('\n', 1)[0]
  if (/[一-鿿]/.test(firstLine)) return translateReason(firstLine)
  return '研究任务在执行阶段异常退出，失败证据已保留'
}

function yearsBefore(isoDate: string, years: number) {
  const [year, month, day] = isoDate.split('-').map(Number)
  const value = new Date(Date.UTC(year, month - 1, day))
  value.setUTCFullYear(value.getUTCFullYear() - years)
  return value.toISOString().slice(0, 10)
}

function translateEvidence(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(translateEvidence)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        EVIDENCE_KEY_LABELS[key] || TERM_LABELS[key] || key.replaceAll('_', ' '),
        translateEvidence(item),
      ]),
    )
  }
  if (typeof value === 'string') return statusLabel(value) !== '未开始' ? statusLabel(value) : termLabel(value)
  return value
}

function SectionTitle({ index, title, subtitle }: { index: number; title: string; subtitle: string }) {
  return (
    <div className="border-b border-border px-3 py-2.5">
      <div className="flex items-start gap-2">
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent/10 font-mono text-[9px] text-accent">{index}</span>
        <div>
          <div className="text-xs font-semibold text-foreground">{title}</div>
          <div className="mt-0.5 text-[9px] text-muted">{subtitle}</div>
        </div>
      </div>
    </div>
  )
}

function gateCounts(candidate: AlphaCandidateResult) {
  return candidate.gates.reduce(
    (acc, gate) => {
      acc[gate.status] += 1
      return acc
    },
    { passed: 0, failed: 0, pending: 0 },
  )
}

export function AlphaMining() {
  const queryClient = useQueryClient()
  const [assetType, setAssetType] = useState<'stock' | 'etf'>('stock')
  const [profile, setProfile] = useState<MiningBudgetProfile>('exploratory')
  const [horizon, setHorizon] = useState<1 | 3 | 5 | 10 | 20 | 60>(5)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [datePreset, setDatePreset] = useState<'1y' | '3y' | 'all' | 'custom'>('1y')
  const [selectedEngines, setSelectedEngines] = useState<string[]>([])
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null)

  const charterQuery = useQuery({ queryKey: QK.alphaCharter, queryFn: api.alphaCharter, staleTime: Infinity })
  const enginesQuery = useQuery({ queryKey: QK.alphaEngines, queryFn: api.alphaEngines, staleTime: Infinity })
  const configQuery = useQuery({ queryKey: QK.alphaConfig, queryFn: api.alphaConfig })
  const factorsQuery = useQuery({ queryKey: QK.factorColumns, queryFn: api.factorColumns, staleTime: Infinity })
  const runsQuery = useQuery({ queryKey: QK.alphaRuns, queryFn: api.alphaRuns, refetchInterval: 5000 })
  const experimentsQuery = useQuery({ queryKey: QK.alphaExperiments, queryFn: api.alphaExperiments, refetchInterval: 10000 })
  const candidatesQuery = useQuery({ queryKey: QK.alphaCandidates, queryFn: api.alphaCandidates, refetchInterval: 10000 })
  const championQuery = useQuery({ queryKey: QK.alphaChampion, queryFn: api.alphaChampion, refetchInterval: 10000 })
  const availabilityQuery = useQuery({
    queryKey: QK.alphaAvailability(assetType, profile, horizon, start, end),
    queryFn: () => api.alphaAvailability({ assetType, budgetProfile: profile, forwardHorizon: horizon, start, end }),
  })
  const candidateQuery = useQuery({
    queryKey: QK.alphaCandidate(selectedCandidateId || ''),
    queryFn: () => api.alphaCandidate(selectedCandidateId!),
    enabled: !!selectedCandidateId,
  })
  const selectedEvidence = candidatesQuery.data?.items.find(item => item.candidate_id === selectedCandidateId)
  const shadowQuery = useQuery({
    queryKey: QK.alphaShadow(selectedCandidateId || ''),
    queryFn: () => api.alphaShadow(selectedCandidateId!),
    enabled: !!selectedCandidateId && ['shadow', 'challenger'].includes(selectedEvidence?.state.state || ''),
    retry: false,
  })

  const readyIds = useMemo(
    () => new Set(availabilityQuery.data?.engines.filter(item => item.ready).map(item => item.engine_id) || []),
    [availabilityQuery.data],
  )

  useEffect(() => {
    const latest = availabilityQuery.data?.available_end
    if (!latest || start || end) return
    setStart(yearsBefore(latest, 1))
    setEnd(latest)
  }, [availabilityQuery.data?.available_end, end, start])

  useEffect(() => {
    if (!availabilityQuery.data) return
    setSelectedEngines(current => {
      const retained = current.filter(id => readyIds.has(id))
      return retained.length ? retained : [...readyIds]
    })
  }, [availabilityQuery.data, readyIds])

  useEffect(() => {
    if (currentRunId) return
    const runs = runsQuery.data?.items || []
    const preferred = runs.find(item => ACTIVE.has(item.status)) || runs[0]
    if (!preferred) return
    setCurrentRunId(preferred.run_id)
    if (SUCCESS.has(preferred.status)) setSelectedRunId(preferred.run_id)
  }, [currentRunId, runsQuery.data])

  const runQuery = useQuery({
    queryKey: QK.alphaRun(currentRunId || ''),
    queryFn: () => api.alphaRun(currentRunId!),
    enabled: !!currentRunId,
    refetchInterval: query => (ACTIVE.has(query.state.data?.status || '') ? 1500 : false),
  })
  const effectiveResultId = selectedRunId || (runQuery.data && SUCCESS.has(runQuery.data.status) ? runQuery.data.run_id : null)
  const resultQuery = useQuery({
    queryKey: QK.alphaResult(effectiveResultId || ''),
    queryFn: () => api.alphaResult(effectiveResultId!),
    enabled: !!effectiveResultId,
    retry: false,
  })

  useEffect(() => {
    const run = runQuery.data
    if (!run || ACTIVE.has(run.status)) return
    void queryClient.invalidateQueries({ queryKey: QK.alphaRuns })
    void queryClient.invalidateQueries({ queryKey: QK.alphaExperiments })
    void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
    if (SUCCESS.has(run.status)) setSelectedRunId(run.run_id)
  }, [queryClient, runQuery.data])

  const startMutation = useMutation({
    mutationFn: async (payload: AlphaMiningRequest) => {
      if (!configQuery.data?.enabled) {
        const updated = await api.alphaUpdateConfig({ enabled: true })
        queryClient.setQueryData(QK.alphaConfig, updated)
      }
      return api.alphaStart(payload)
    },
    onSuccess: run => {
      setCurrentRunId(run.run_id)
      setSelectedRunId(null)
      toast('研究任务已启动，全部过程会写入独立证据账本', 'success')
      void queryClient.invalidateQueries({ queryKey: QK.alphaRuns })
    },
    onError: error => toast(error instanceof Error ? error.message : '任务启动失败', 'error'),
  })
  const cancelMutation = useMutation({
    mutationFn: api.alphaCancel,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: QK.alphaRuns }),
  })
  const configMutation = useMutation({
    mutationFn: (patch: Partial<AlphaConfig>) => api.alphaUpdateConfig(patch),
    onSuccess: value => {
      queryClient.setQueryData(QK.alphaConfig, value)
      toast('研究设置已保存，旧挖掘功能不受影响', 'success')
    },
  })
  const shadowStartMutation = useMutation({
    mutationFn: (candidateId: string) => api.alphaShadowStart(candidateId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      void queryClient.invalidateQueries({ queryKey: QK.alphaShadow(selectedCandidateId || '') })
      toast('独立前向模拟账户已创建', 'success')
    },
  })
  const shadowEvaluateMutation = useMutation({
    mutationFn: (candidateId: string) => api.alphaShadowEvaluate(candidateId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      void queryClient.invalidateQueries({ queryKey: QK.alphaChampion })
    },
  })
  const promoteMutation = useMutation({
    mutationFn: (candidateId: string) => api.alphaPromote(candidateId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaChampion })
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      toast('挑战者已发布并晋级为新的动态冠军', 'success')
    },
  })

  const factorNames = factorsQuery.data?.columns.map(item => item.id) || []
  const currentRun = runQuery.data || runsQuery.data?.items.find(item => item.run_id === currentRunId) || null
  const active = !!currentRun && ACTIVE.has(currentRun.status)
  const result = resultQuery.data
  const catalogRows = Object.entries(availabilityQuery.data?.catalog.datasets || {})
  const evidenceItems = candidatesQuery.data?.items || []
  const hasTopError = [charterQuery, enginesQuery, configQuery, runsQuery, availabilityQuery].some(query => query.isError)
  const blockers = useMemo(() => {
    const availability = availabilityQuery.data
    if (availabilityQuery.isError) return ['研究条件检查失败，请刷新后重试']
    if (!availability) return ['正在检查数据与研究条件']
    const reasons: string[] = []
    if (availability.trading_bars < availability.required_bars) {
      reasons.push(`当前只有 ${availability.trading_bars} 个交易日，本档研究至少需要 ${availability.required_bars} 个`)
    }
    if (availability.outer_folds < 1) reasons.push('当前区间无法形成独立的训练与检验窗口')
    const universe = availability.catalog.datasets.historical_universe
    if (universe && !universe.ready) reasons.push(...universe.reasons.map(translateReason))
    if (readyIds.size === 0) reasons.push('当前数据下没有可运行的发现引擎')
    if (readyIds.size > 0 && selectedEngines.length === 0) reasons.push('请至少选择一个可用的发现引擎')
    return [...new Set(reasons)]
  }, [availabilityQuery.data, availabilityQuery.isError, readyIds, selectedEngines.length])

  function applyDatePreset(preset: '1y' | '3y' | 'all') {
    const availability = availabilityQuery.data
    const latest = availability?.available_end
    const earliest = availability?.available_start
    if (!latest || !earliest) return
    setDatePreset(preset)
    setEnd(latest)
    setStart(preset === 'all' ? earliest : yearsBefore(latest, preset === '1y' ? 1 : 3))
    if (preset === '1y') setProfile('exploratory')
  }

  function startResearch() {
    if (blockers.length) return toast(blockers[0], 'error')
    if (!selectedEngines.length || !factorNames.length) return toast('请至少选择一个可用的发现引擎', 'error')
    startMutation.mutate({
      engine_ids: selectedEngines,
      factor_names: factorNames,
      asset_type: assetType,
      start: start || null,
      end: end || null,
      budget_profile: profile,
      forward_horizon: horizon,
      commission_pct: 0.0002,
      stamp_tax_pct: 0.0005,
      slippage_bps: 5,
      max_positions: 10,
      max_candidates_per_engine: profile === 'strict' ? 8 : profile === 'balanced' ? 4 : 2,
      max_trials_per_engine: profile === 'strict' ? 128 : profile === 'balanced' ? 64 : 24,
    })
  }

  const launchBusy = startMutation.isPending || availabilityQuery.isFetching || configQuery.isLoading
  const usableEngineCount = availabilityQuery.data?.engines.filter(item => item.ready).length || 0

  return (
    <div className="flex min-h-full flex-col bg-base">
      <PageHeader
        title="Alpha挖掘"
        subtitle={<span className="hidden md:inline">独立研究 · 时点防泄漏 · 样本外检验 · 前向模拟 · 动态冠军</span>}
        className="shrink-0 flex-wrap gap-y-2 bg-base/95 px-3 lg:px-5"
        right={
          <div className="flex flex-wrap items-center justify-end gap-2 text-[10px]">
            <span className="rounded-full border border-border bg-surface px-2 py-1 text-muted">旧挖掘独立保留</span>
            <button
              type="button"
              disabled={configMutation.isPending}
              onClick={() => configMutation.mutate({ auto_run_enabled: !configQuery.data?.auto_run_enabled })}
              className={cn(
                'rounded-full border px-2 py-1',
                configQuery.data?.auto_run_enabled
                  ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                  : 'border-border bg-elevated text-muted',
              )}
            >
              自动研究：{configQuery.data?.auto_run_enabled ? '已开启' : '已关闭'}
            </button>
            <span className={cn(
              'rounded-full border px-2 py-1',
              configQuery.data?.enabled
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : 'border-border bg-elevated text-muted',
            )}>
              {configQuery.data?.enabled ? '手动研究已开启' : '点击启动时自动开启'}
            </span>
          </div>
        }
      />
      <main className="space-y-3 p-3 lg:p-4">
        {hasTopError && (
          <div className="rounded-card border border-danger/30 bg-danger/5 px-3 py-2 text-[10px] text-danger">
            研究服务加载失败。旧挖掘仍可使用，本页不会用空数据伪装成功。
          </div>
        )}

        <section className={CARD}>
          <SectionTitle index={1} title="研究总览" subtitle="默认研究最近一年；开始前直接告诉你数据是否够、为什么不能运行。" />
          <div className="grid gap-3 p-3 xl:grid-cols-[360px_minmax(0,1fr)]">
            <div>
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-[10px] font-medium text-foreground">研究区间</div>
                <div className="flex rounded-md border border-border bg-base/40 p-0.5 text-[9px]">
                  {([
                    ['1y', '近1年'],
                    ['3y', '近3年'],
                    ['all', '全部历史'],
                  ] as const).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => applyDatePreset(value)}
                      className={cn(
                        'rounded px-2 py-1',
                        datePreset === value ? 'bg-accent text-white' : 'text-muted hover:text-foreground',
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="mb-3">
                <div className="mb-1.5 text-[9px] font-medium text-foreground">研究强度</div>
                <div className="grid gap-1.5 sm:grid-cols-3 xl:grid-cols-1">
                  {(Object.entries(PROFILE_META) as [MiningBudgetProfile, typeof PROFILE_META[MiningBudgetProfile]][]).map(([value, meta]) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setProfile(value)}
                      className={cn(
                        'rounded-lg border px-2.5 py-2 text-left transition-colors',
                        profile === value
                          ? 'border-accent/50 bg-accent/10'
                          : 'border-border bg-base/30 hover:border-secondary/40',
                      )}
                    >
                      <div className={cn('text-[9px] font-semibold', profile === value ? 'text-accent' : 'text-foreground')}>{meta.title}</div>
                      <div className="mt-0.5 text-[8px] leading-relaxed text-muted">{meta.description}</div>
                      <div className="mt-1 font-mono text-[7px] text-secondary">{meta.budget}</div>
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <label className="text-[9px] text-muted">
                  资产
                  <select className={INPUT} value={assetType} onChange={event => setAssetType(event.target.value as 'stock' | 'etf')}>
                    <option value="stock">A股</option>
                    <option value="etf">ETF</option>
                  </select>
                </label>
                <label className="text-[9px] text-muted">
                  预测周期
                  <select className={INPUT} value={horizon} onChange={event => setHorizon(Number(event.target.value) as typeof horizon)}>
                    {[1, 3, 5, 10, 20, 60].map(value => <option key={value} value={value}>{value}个交易日</option>)}
                  </select>
                </label>
                <label className="text-[9px] text-muted">
                  开始日期
                  <input
                    className={INPUT}
                    type="date"
                    value={start}
                    onChange={event => { setStart(event.target.value); setDatePreset('custom') }}
                  />
                </label>
                <label className="text-[9px] text-muted">
                  结束日期
                  <input
                    className={INPUT}
                    type="date"
                    value={end}
                    onChange={event => { setEnd(event.target.value); setDatePreset('custom') }}
                  />
                </label>
              </div>
              <button
                type="button"
                disabled={active || launchBusy}
                onClick={startResearch}
                className="mt-3 inline-flex h-10 w-full items-center justify-center gap-2 rounded-btn bg-accent text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                {active || launchBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {active
                  ? statusLabel(currentRun?.status)
                  : launchBusy
                    ? '正在检查研究条件'
                    : profile === 'exploratory'
                      ? '启动近1年研究'
                      : '启动正式研究'}
              </button>
              {active && currentRun && (
                <button
                  type="button"
                  onClick={() => cancelMutation.mutate(currentRun.run_id)}
                  className="mt-2 inline-flex h-8 w-full items-center justify-center gap-1 rounded-btn border border-border text-[10px] text-secondary"
                >
                  <Pause className="h-3 w-3" />取消并保留证据
                </button>
              )}
              <ResearchReadiness
                blockers={blockers}
                tradingBars={availabilityQuery.data?.trading_bars}
                requiredBars={availabilityQuery.data?.required_bars}
                outerFolds={availabilityQuery.data?.outer_folds}
                usableEngineCount={usableEngineCount}
                onUseQuick={() => setProfile('exploratory')}
                onUseAll={() => applyDatePreset('all')}
                showQuickAction={profile !== 'exploratory'}
                profile={profile}
              />
            </div>
            <div className="grid gap-2 sm:grid-cols-3">
              {charterQuery.data?.completion_levels.map((level, index) => (
                <div key={level.id} className="rounded-lg border border-border bg-base/40 p-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                    <span className="font-mono text-accent">0{index + 1}</span>{level.label}
                  </div>
                  <p className="mt-2 text-[10px] leading-relaxed text-muted">{level.meaning}</p>
                </div>
              ))}
              <div className="sm:col-span-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg bg-border md:grid-cols-4">
                <Metric label="当前交易日" value={availabilityQuery.data?.trading_bars ?? '—'} />
                <Metric label="本档最低需要" value={availabilityQuery.data?.required_bars ?? '—'} />
                <Metric label="独立检验窗口" value={availabilityQuery.data?.outer_folds ?? '—'} />
                <Metric
                  label="能否开始研究"
                  value={blockers.length ? '暂不可开始' : '可以开始'}
                  tone={blockers.length ? 'bad' : 'good'}
                />
              </div>
              <div className="sm:col-span-3 rounded-lg border border-accent/20 bg-accent/5 px-3 py-2 text-[9px] text-secondary">
                研究边界：从当时的全部合格股票出发，任何现有策略都不进入发现样本。只有Alpha系统已产生正式冠军时，才在公共验证的最后一步追加同口径比较。
              </div>
            </div>
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={2} title="机会地图" subtitle="四类视角帮助检查研究是否偏科；已覆盖代表系统已有引擎，待扩展代表预留方向。" />
          <div className="grid gap-3 p-3 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(enginesQuery.data?.taxonomy || {}).map(([dimension, rows]) => {
              const meta = DIMENSION_META[dimension] || { title: dimension, description: '' }
              const covered = rows.filter(row => row.covered).length
              return (
                <div key={dimension} className="rounded-lg border border-border bg-base/30 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] font-semibold text-foreground">{meta.title}</div>
                      <div className="mt-1 text-[8px] leading-relaxed text-muted">{meta.description}</div>
                    </div>
                    <span className="shrink-0 rounded-full bg-accent/10 px-2 py-1 text-[8px] text-accent">已覆盖 {covered}/{rows.length}</span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {rows.map(row => (
                      <span
                        key={row.id}
                        className={cn(
                          'rounded-md border px-2 py-1 text-[8px]',
                          row.covered
                            ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-300'
                            : 'border-border bg-surface text-muted',
                        )}
                      >
                        {termLabel(row.id)} · {row.covered ? `${row.engine_ids.length}个引擎` : '待扩展'}
                      </span>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={3} title="信息与数据覆盖" subtitle="所有历史研究只使用当时已经可见的数据；关键数据缺失时直接停止，不允许用当前快照倒填。" />
          <div className="grid gap-2 p-3 sm:grid-cols-2 lg:grid-cols-3">
            {catalogRows.map(([id, item]) => {
              const meta = DATASET_LABELS[id] || { title: id, description: '' }
              const coverage = Number(item.observations.coverage)
              return (
                <div key={id} className={cn('rounded-lg border p-3', item.ready ? 'border-emerald-500/20 bg-emerald-500/5' : 'border-danger/20 bg-danger/5')}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[10px] font-medium text-foreground">{meta.title}</div>
                      <div className="mt-1 text-[8px] text-muted">{meta.description}</div>
                    </div>
                    {item.ready
                      ? <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
                      : <AlertTriangle className="h-4 w-4 shrink-0 text-danger" />}
                  </div>
                  <div className={cn('mt-2 text-[9px] leading-relaxed', item.ready ? 'text-emerald-300' : 'text-danger')}>
                    {item.ready
                      ? `${Number.isFinite(coverage) ? `覆盖率 ${Math.round(coverage * 100)}% · ` : ''}已通过时点防泄漏检查`
                      : item.reasons.map(translateReason).join('；')}
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={4} title="发现引擎" subtitle="默认选择当前可运行的全部引擎；缺数据的引擎会明确说明原因，不影响其他引擎研究。" />
          <div className="grid gap-2 p-3 md:grid-cols-2 xl:grid-cols-4">
            {enginesQuery.data?.items.map(engine => {
              const readiness = availabilityQuery.data?.engines.find(item => item.engine_id === engine.engine_id)
              const checked = selectedEngines.includes(engine.engine_id)
              return (
                <label
                  key={engine.engine_id}
                  className={cn(
                    'rounded-lg border p-3',
                    checked ? 'border-accent/40 bg-accent/5' : 'border-border bg-base/30',
                    !readiness?.ready && 'cursor-not-allowed opacity-70',
                  )}
                >
                  <div className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      disabled={!readiness?.ready}
                      checked={checked}
                      onChange={() => setSelectedEngines(current => checked ? current.filter(id => id !== engine.engine_id) : [...current, engine.engine_id])}
                      className="mt-0.5 accent-accent"
                    />
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <div className="text-[10px] font-medium text-foreground">{engine.name}</div>
                        <span className={cn('rounded-full px-1.5 py-0.5 text-[7px]', readiness?.ready ? 'bg-emerald-500/10 text-emerald-300' : 'bg-danger/10 text-danger')}>
                          {readiness?.ready ? '可运行' : '暂不可用'}
                        </span>
                      </div>
                      <div className="mt-1 text-[8px] leading-relaxed text-muted">
                        {readiness?.ready
                          ? `${engine.discovery_classes.map(termLabel).join(' / ')} · 预测${engine.forecast_horizons.join('/')}个交易日`
                          : readiness?.reasons.map(translateReason).join('；') || '正在检查资格'}
                      </div>
                    </div>
                  </div>
                </label>
              )
            })}
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={5} title="实验账本" subtitle="每次尝试、失败原因、数据版本和研究结果都会保留，不能事后覆盖。" />
          <div className="grid gap-3 p-3 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] text-left text-[9px]">
                <thead className="text-muted">
                  <tr>{['实验编号', '状态', '创建时间', '当前进度', '研究结论', '失败原因'].map(value => <th key={value} className="border-b border-border px-2 py-2 font-medium">{value}</th>)}</tr>
                </thead>
                <tbody>
                  {runsQuery.data?.items.slice(0, 12).map((run: AlphaRun) => (
                    <tr
                      key={run.run_id}
                      className="cursor-pointer border-b border-border/60 hover:bg-elevated/40"
                      onClick={() => { setCurrentRunId(run.run_id); if (SUCCESS.has(run.status)) setSelectedRunId(run.run_id) }}
                    >
                      <td className="px-2 py-2 font-mono text-secondary">{run.run_id}</td>
                      <td className="px-2 py-2 text-foreground">{statusLabel(run.status)}</td>
                      <td className="px-2 py-2 text-muted">{run.created_at?.slice(0, 19)}</td>
                      <td className="px-2 py-2 text-muted">{run.progress?.label || '—'}</td>
                      <td className="px-2 py-2 text-muted">{statusLabel(run.research_state)}</td>
                      <td className="max-w-[260px] px-2 py-2 text-danger">{run.status === 'failed' ? translateRunError(run.error) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!runsQuery.data?.items.length && <Empty text="尚无实验；点击上方按钮开始第一轮研究。" />}
            </div>
            <div className="rounded-lg border border-border bg-base/30 p-3">
              <div className="flex items-center gap-2 text-[10px] font-medium text-foreground"><Database className="h-3.5 w-3.5 text-accent" />证据留存</div>
              <div className="mt-2 space-y-1 text-[9px] text-muted">
                <div>已记录实验：{experimentsQuery.data?.items.length || 0}</div>
                <div>已冻结候选：{evidenceItems.length}</div>
                <div>失败尝试是否保留：是</div>
                <div>候选定义能否事后修改：不能</div>
              </div>
              <details className="mt-3 border-t border-border pt-2 text-[8px] text-muted">
                <summary className="cursor-pointer text-secondary">查看审计标识</summary>
                <div className="mt-2 break-all font-mono">数据版本：{availabilityQuery.data?.catalog.fingerprint || '—'}</div>
              </details>
            </div>
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={6} title="候选证据" subtitle="统一展示样本外收益、近期表现、成本、延迟、参数、容量和集中度；任一硬门槛失败即淘汰。" />
          <RunOutcome run={currentRun} result={result} />
          {result
            ? <CandidateTable candidates={result.candidates} onSelect={id => { if (id) setSelectedCandidateId(id) }} />
            : <Empty text="尚无可裁决结果；系统不会凭空生成候选。" />}
          {candidateQuery.data && (
            <div className="grid gap-2 border-t border-border p-3 lg:grid-cols-3">
              <EvidenceBlock title="已冻结的候选方案" value={candidateQuery.data.candidate.candidate} />
              <EvidenceBlock title="样本外与压力测试证据" value={candidateQuery.data.candidate.outer_evaluation} />
              <div className="rounded-lg border border-border bg-base/30 p-3 text-[9px] text-muted">
                <div className="font-medium text-foreground">不可修改的状态记录</div>
                <div className="mt-2 font-mono">{candidateQuery.data.candidate.candidate_id}</div>
                <div className="mt-1">状态事件：{candidateQuery.data.events.length} 条</div>
                <div className="mt-1 break-all">内容指纹：{candidateQuery.data.candidate.content_sha256}</div>
                <div className="mt-2 space-y-1">
                  {candidateQuery.data.events.slice(-5).map(event => (
                    <div key={String(event.event_sha256)}>{statusLabel(String(event.from))} → {statusLabel(String(event.to))}</div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        <section className={CARD}>
          <SectionTitle index={7} title="独立准入与冠军晋级" subtitle="发现阶段不使用任何现有策略；首任冠军先过绝对收益、风险和稳定性门槛，之后挑战者才追加同口径冠军比较。" />
          <div className="grid gap-3 p-3 lg:grid-cols-[260px_minmax(0,1fr)]">
            <div className="rounded-lg border border-accent/30 bg-accent/5 p-3">
              <div className="flex items-center gap-2 text-[10px] text-accent"><ShieldCheck className="h-4 w-4" />Alpha系统正式冠军</div>
              <div className="mt-2 text-sm font-semibold text-foreground">{championQuery.data?.champion.strategy_id ? strategyLabel(championQuery.data.champion.strategy_id) : '尚无正式冠军'}</div>
              <div className="mt-1 text-[9px] leading-relaxed text-muted">{championQuery.data?.champion.reason || '正在读取冠军账本'}</div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[620px] text-left text-[9px]">
                <thead className="text-muted"><tr>{['候选', '状态', '样本外收益', '夏普比率', '最大回撤', '门槛'].map(value => <th key={value} className="border-b border-border px-2 py-2 font-medium">{value}</th>)}</tr></thead>
                <tbody>
                  {championQuery.data?.challengers.map(item => (
                    <tr key={item.candidate_id} className="border-b border-border/60">
                      <td className="px-2 py-2 font-mono text-secondary">{item.candidate_id}</td>
                      <td className="px-2 py-2 text-foreground">{statusLabel(item.state)}</td>
                      <td className="px-2 py-2 font-mono">{fmtPct(item.return)}</td>
                      <td className="px-2 py-2 font-mono">{fmtNumber(item.sharpe)}</td>
                      <td className="px-2 py-2 font-mono">{fmtPct(item.max_drawdown)}</td>
                      <td className="px-2 py-2"><span className="text-emerald-400">{item.gates_passed}过</span> <span className="text-danger">{item.gates_failed}败</span> <span className="text-muted">{item.gates_pending}待验证</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className={CARD}>
          <SectionTitle index={8} title="前向模拟" subtitle="候选进入独立模拟账户，核对信号、订单、成交、持仓、滑点、因子衰减和净值；发现漂移立即暂停。" />
          <div className="grid gap-3 p-3 lg:grid-cols-[minmax(0,1fr)_auto]">
            <div>
              {selectedEvidence ? (
                <div className="rounded-lg border border-border bg-base/30 p-3">
                  <div className="font-mono text-[10px] text-foreground">{selectedEvidence.candidate_id}</div>
                  <div className="mt-1 text-[9px] text-muted">
                    {statusLabel(selectedEvidence.state.state)} · 最低{configQuery.data?.shadow_min_trading_days || '—'}个交易日 / {configQuery.data?.shadow_min_fills || '—'}笔成交 / {configQuery.data?.shadow_min_factor_round_trips || '—'}组闭环交易
                  </div>
                  {shadowQuery.data?.evaluation ? (
                    <div className="mt-3 grid gap-px overflow-hidden rounded-lg bg-border sm:grid-cols-3">
                      <Metric label="前向净收益" value={fmtPct(shadowQuery.data.evaluation.total_return)} />
                      <Metric label="实际滑点（基点）" value={fmtNumber(shadowQuery.data.evaluation.average_slippage_bps)} />
                      <Metric label="因子排序相关性" value={fmtNumber(shadowQuery.data.evaluation.factor_decay?.rank_ic)} tone={shadowQuery.data.evaluation.factor_decay?.status === 'passed' ? 'good' : shadowQuery.data.evaluation.factor_decay?.status === 'failed' ? 'bad' : undefined} />
                      <Metric label="账实核对" value={shadowQuery.data.evaluation.reconcile_ok ? '通过' : '失败'} tone={shadowQuery.data.evaluation.reconcile_ok ? 'good' : 'bad'} />
                      <Metric label="信号订单闭环" value={shadowQuery.data.evaluation.signal_order_parity ? '通过' : '失败'} tone={shadowQuery.data.evaluation.signal_order_parity ? 'good' : 'bad'} />
                      <Metric label="漂移判定" value={shadowQuery.data.evaluation.drift_detected ? '已暂停' : '未发现'} tone={shadowQuery.data.evaluation.drift_detected ? 'bad' : 'good'} />
                    </div>
                  ) : <div className="mt-2 text-[9px] text-muted">选择研究候选后创建独立账户；没有成交时收益必须保持为零。</div>}
                </div>
              ) : <Empty text="先在候选证据区选择一个候选" compact />}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" disabled={selectedEvidence?.state.state !== 'research_candidate' || shadowStartMutation.isPending} onClick={() => selectedCandidateId && shadowStartMutation.mutate(selectedCandidateId)} className="h-8 rounded-btn border border-border px-3 text-[10px] text-secondary disabled:opacity-40">创建前向账户</button>
              <button type="button" disabled={selectedEvidence?.state.state !== 'shadow' || shadowEvaluateMutation.isPending} onClick={() => selectedCandidateId && shadowEvaluateMutation.mutate(selectedCandidateId)} className="h-8 rounded-btn border border-border px-3 text-[10px] text-secondary disabled:opacity-40">核对并判定</button>
              <button type="button" disabled={selectedEvidence?.state.state !== 'challenger' || promoteMutation.isPending} onClick={() => selectedCandidateId && promoteMutation.mutate(selectedCandidateId)} className="h-8 rounded-btn bg-accent px-3 text-[10px] font-semibold text-white disabled:opacity-40">发布并晋级冠军</button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

function RunOutcome({ run, result }: { run: AlphaRun | null; result?: AlphaResult }) {
  if (!run) {
    return <div className="border-b border-border px-3 py-2 text-[9px] text-muted">尚未运行研究，任务完成后会在这里直接展示结果或失败原因。</div>
  }
  if (run.status === 'failed') {
    return (
      <div className="border-b border-danger/20 bg-danger/5 px-3 py-3 text-[9px] text-danger">
        <div className="flex items-center gap-1.5 font-semibold"><AlertTriangle className="h-3.5 w-3.5" />本次研究未产生结果</div>
        <div className="mt-1.5 text-[10px] leading-relaxed">{translateRunError(run.error)}</div>
        <div className="mt-1 text-danger/70">停止阶段：{run.progress?.label || '未记录'} · 失败证据已写入实验账本</div>
        {run.error && (
          <details className="mt-2 text-[8px] text-muted">
            <summary className="cursor-pointer">查看技术详情</summary>
            <pre className="mt-1 max-h-36 overflow-auto whitespace-pre-wrap break-all rounded bg-base/50 p-2 font-mono">{run.error}</pre>
          </details>
        )}
      </div>
    )
  }
  if (ACTIVE.has(run.status)) {
    return (
      <div className="border-b border-accent/20 bg-accent/5 px-3 py-3 text-[9px] text-secondary">
        <div className="flex items-center gap-1.5 font-medium"><LoaderCircle className="h-3.5 w-3.5 animate-spin" />研究正在运行</div>
        <div className="mt-1">{run.progress?.label || '正在准备'}{run.progress?.total ? ` · ${run.progress.done || 0}/${run.progress.total}` : ''}</div>
      </div>
    )
  }
  if (SUCCESS.has(run.status) && result) {
    return (
      <div className="grid gap-px border-b border-border bg-border sm:grid-cols-3 lg:grid-cols-6">
        <Metric label="研究结论" value={statusLabel(result.research_state)} tone="good" />
        <Metric label="外层检验窗口" value={result.summary.outer_fold_count} />
        <Metric label="已评估引擎" value={result.summary.candidate_engine_count} />
        <Metric label="记录尝试" value={result.summary.trial_count} />
        <Metric label="真实回测" value={result.summary.backtest_count} />
        <Metric label="引擎异常" value={result.engine_failures.length} tone={result.engine_failures.length ? 'bad' : 'good'} />
        {result.engine_failures.length > 0 && (
          <div className="sm:col-span-3 lg:col-span-6 bg-surface px-3 py-2 text-[8px] text-danger">
            {result.engine_failures.map(item => `${item.engine_id}：${translateRunError(item.error)}`).join('；')}
          </div>
        )}
      </div>
    )
  }
  return (
    <div className="border-b border-border px-3 py-2 text-[9px] text-muted">
      任务状态：{statusLabel(run.status)}。该任务没有可展示的候选结果。
    </div>
  )
}

function ResearchReadiness({
  blockers,
  tradingBars,
  requiredBars,
  outerFolds,
  usableEngineCount,
  onUseQuick,
  onUseAll,
  showQuickAction,
  profile,
}: {
  blockers: string[]
  tradingBars?: number
  requiredBars?: number
  outerFolds?: number
  usableEngineCount: number
  onUseQuick: () => void
  onUseAll: () => void
  showQuickAction: boolean
  profile: MiningBudgetProfile
}) {
  if (!blockers.length) {
    return (
      <div className="mt-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-2.5 text-[9px] leading-relaxed text-emerald-300">
        <div className="flex items-center gap-1.5 font-medium"><CheckCircle2 className="h-3.5 w-3.5" />{profile === 'exploratory' ? '近1年研究可以开始' : '正式研究可以开始'}</div>
        <div className="mt-1 text-emerald-300/80">{tradingBars} 个交易日，最低需要 {requiredBars} 个；可形成 {outerFolds} 个独立检验窗口，{usableEngineCount} 个发现引擎可运行。</div>
        {profile === 'exploratory' && (
          <div className="mt-1 border-t border-emerald-500/10 pt-1 text-emerald-300/70">
            近1年用于快速发现和证伪；结果只能进入研究候选，不能直接晋级冠军。正式晋级还必须通过更长历史的标准研究、压力测试和前向模拟。
          </div>
        )}
      </div>
    )
  }
  return (
    <div className="mt-2 rounded-lg border border-danger/20 bg-danger/5 p-2.5 text-[9px] leading-relaxed text-danger">
      <div className="flex items-center gap-1.5 font-medium"><Info className="h-3.5 w-3.5" />暂时不能开始，原因如下</div>
      <ul className="mt-1 list-disc space-y-0.5 pl-4">{blockers.map(reason => <li key={reason}>{reason}</li>)}</ul>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {showQuickAction && <button type="button" onClick={onUseQuick} className="rounded border border-danger/30 px-2 py-1 text-[8px] hover:bg-danger/10">改用快速研究</button>}
        <button type="button" onClick={onUseAll} className="rounded border border-danger/30 px-2 py-1 text-[8px] hover:bg-danger/10">使用全部历史</button>
      </div>
    </div>
  )
}

function Metric({ label, value, tone }: { label: string; value: string | number; tone?: 'good' | 'bad' }) {
  return <div className="bg-surface px-3 py-2"><div className="text-[8px] text-muted">{label}</div><div className={cn('mt-0.5 font-mono text-xs font-semibold text-foreground', tone === 'good' && 'text-emerald-400', tone === 'bad' && 'text-danger')}>{value}</div></div>
}

function Empty({ text, compact = false }: { text: string; compact?: boolean }) {
  return <div className={cn('grid place-items-center text-center', compact ? 'min-h-20' : 'min-h-40')}><div><CircleDashed className="mx-auto h-6 w-6 text-muted" /><div className="mt-2 text-[10px] text-muted">{text}</div></div></div>
}

function EvidenceBlock({ title, value }: { title: string; value: unknown }) {
  const translated = value ? translateEvidence(value) : null
  return <div className="rounded-lg border border-border bg-base/30 p-3 text-[9px]"><div className="font-medium text-foreground">{title}</div><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all font-mono leading-relaxed text-muted">{translated ? JSON.stringify(translated, null, 2) : '尚无证据'}</pre></div>
}

function CandidateTable({ candidates, onSelect }: { candidates: AlphaCandidateResult[]; onSelect: (id?: string | null) => void }) {
  if (!candidates.length) return <Empty text="本次实验没有冻结候选" />
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1180px] text-left text-[9px]">
        <thead className="border-b border-border bg-base/40 text-muted"><tr>{['发现引擎', '状态', '样本外收益', '夏普比率', '最大回撤', '近1年', '近3个月', '双倍成本', '延迟成交', '参数扰动', '容量', '集中度', '门槛'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead>
        <tbody>
          {candidates.map(candidate => {
            const counts = gateCounts(candidate)
            return (
              <tr key={candidate.engine_id} onClick={() => onSelect(candidate.candidate_id)} className={cn('border-b border-border/60 last:border-0', candidate.candidate_id && 'cursor-pointer hover:bg-elevated/40')}>
                <td className="px-3 py-2"><div className="font-medium text-foreground">{candidate.engine_name}</div><div className="font-mono text-[8px] text-muted">{candidate.candidate_id || candidate.engine_id}</div></td>
                <td className="px-3 py-2 text-secondary">{statusLabel(candidate.state)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.stitched_oos_return)}</td>
                <td className="px-3 py-2 font-mono">{fmtNumber(candidate.metrics.stitched_oos_sharpe)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.max_drawdown)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.recent_1y_return)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.recent_3m_return)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.double_cost_return)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.delayed_entry_return)}</td>
                <td className="px-3 py-2 font-mono">{fmtPct(candidate.metrics.worst_parameter_return)}</td>
                <td className="px-3 py-2">{candidate.metrics.capacity_passed == null ? '—' : candidate.metrics.capacity_passed ? '通过' : '失败'}</td>
                <td className="px-3 py-2">{candidate.metrics.concentration_passed == null ? '—' : candidate.metrics.concentration_passed ? '通过' : '失败'}</td>
                <td className="px-3 py-2"><span className="text-emerald-400">{counts.passed}过</span> <span className="text-danger">{counts.failed}败</span> <span className="text-muted">{counts.pending}待验证</span></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
