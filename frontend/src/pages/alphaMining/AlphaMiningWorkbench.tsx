// Requirements: AM-S7R-001 through AM-S7R-009; AM-WB-P1-001 through AM-WB-P1-010.
import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Database,
  FileSearch,
  FlaskConical,
  GitBranch,
  History,
  Link2,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  Sparkles,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { toast } from '@/components/Toast'
import {
  api,
  type AlphaAvailability,
  type AlphaCandidateResult,
  type AlphaEvidenceCandidate,
  type AlphaEngineManifest,
  type AlphaHypothesis,
  type AlphaHypothesisCreate,
  type AlphaLeaderboard,
  type AlphaMarketAttribution,
  type AlphaMiningRequest,
  type AlphaNextResearchSuggestion,
  type AlphaResult,
  type AlphaRun,
  type AlphaShadowStatus,
  type FactorColumn,
} from '@/lib/api'
import { cn } from '@/lib/cn'
import { QK } from '@/lib/queryKeys'
import {
  ACTIVE_RUN_STATUSES,
  DATASET_LABELS,
  GATE_LABELS,
  PROFILE_META,
  SUCCESS_RUN_STATUSES,
  VIEW_META,
  type AlphaWorkbenchDraft,
  type AlphaWorkbenchView,
  engineSearchText,
  factorRule,
  fmtNumber,
  fmtPct,
  formatGateActual,
  formatGateRequired,
  gateCounts,
  hasPeriodCoverage,
  isWorkbenchView,
  runLabel,
  statusLabel,
  successfulOosRange,
  termLabel,
  translateReason,
  translateRunError,
  yearsBefore,
} from './model'
import { EquityCurveChart, EvidenceBarChart } from './AlphaEvidenceCharts'

const DRAFT_KEY = 'alpha_mining_workbench_draft_v1'
const ORIGIN_KEY = 'alpha_mining_workbench_origin_v1'
const INPUT = 'h-8 w-full rounded-input border border-border bg-surface px-2 text-xs text-foreground outline-none transition-colors focus:border-accent'
const LABEL = 'mb-1 block text-[9px] font-medium text-secondary'

const DEFAULT_DRAFT: AlphaWorkbenchDraft = {
  assetType: 'stock',
  profile: 'exploratory',
  horizon: 5,
  start: '',
  end: '',
  datePreset: '1y',
  factorNames: [],
  engineIds: [],
  commissionBps: '2',
  stampTaxBps: '5',
  slippageBps: '5',
  maxPositions: '10',
}

interface AlphaDraftOrigin {
  sourceRunId: string
  suggestionId: string
  title: string
  why: string
  keep: string[]
  changes: AlphaNextResearchSuggestion['changes']
}

function loadDraft(): AlphaWorkbenchDraft {
  try {
    const parsed = JSON.parse(localStorage.getItem(DRAFT_KEY) || '')
    return parsed && typeof parsed === 'object' ? { ...DEFAULT_DRAFT, ...parsed } : DEFAULT_DRAFT
  } catch {
    return DEFAULT_DRAFT
  }
}

function loadOrigin(): AlphaDraftOrigin | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(ORIGIN_KEY) || '')
    return parsed && typeof parsed === 'object' && parsed.sourceRunId && parsed.suggestionId ? parsed : null
  } catch {
    return null
  }
}

function requestToDraft(request: AlphaMiningRequest): AlphaWorkbenchDraft {
  return {
    assetType: request.asset_type,
    profile: request.budget_profile,
    horizon: request.forward_horizon,
    start: request.start || '',
    end: request.end || '',
    datePreset: 'custom',
    factorNames: [...request.factor_names],
    engineIds: [...request.engine_ids],
    commissionBps: String(request.commission_pct * 10_000),
    stampTaxBps: String(request.stamp_tax_pct * 10_000),
    slippageBps: String(request.slippage_bps),
    maxPositions: String(request.max_positions),
  }
}

function suggestionToDraft(request: AlphaMiningRequest, suggestion: AlphaNextResearchSuggestion): AlphaWorkbenchDraft {
  const base = requestToDraft(request)
  const patch = suggestion.request_patch
  if (Array.isArray(patch.engine_ids)) base.engineIds = [...patch.engine_ids]
  if (Array.isArray(patch.factor_names)) base.factorNames = [...patch.factor_names]
  if (patch.asset_type === 'stock' || patch.asset_type === 'etf') base.assetType = patch.asset_type
  if (patch.budget_profile && patch.budget_profile in PROFILE_META) base.profile = patch.budget_profile
  if ([1, 3, 5, 10, 20, 60].includes(Number(patch.forward_horizon))) base.horizon = patch.forward_horizon!
  if (typeof patch.start === 'string') base.start = patch.start
  if (typeof patch.end === 'string') base.end = patch.end
  if (typeof patch.commission_pct === 'number') base.commissionBps = String(patch.commission_pct * 10_000)
  if (typeof patch.stamp_tax_pct === 'number') base.stampTaxBps = String(patch.stamp_tax_pct * 10_000)
  if (typeof patch.slippage_bps === 'number') base.slippageBps = String(patch.slippage_bps)
  if (typeof patch.max_positions === 'number') base.maxPositions = String(patch.max_positions)
  return base
}

function parseNumber(value: string, label: string, minimum: number, maximum: number, integer = false) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum || (integer && !Number.isInteger(parsed))) {
    toast(`${label}必须是 ${minimum} 至 ${maximum}${integer ? ' 的整数' : ''}`, 'error')
    return null
  }
  return parsed
}

export function AlphaMiningWorkbench() {
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const [draft, setDraft] = useState(loadDraft)
  const [draftOrigin, setDraftOrigin] = useState<AlphaDraftOrigin | null>(loadOrigin)
  const [factorSearch, setFactorSearch] = useState('')
  const [engineSearch, setEngineSearch] = useState('')
  const initializedFactors = useRef(false)
  const initializedEngines = useRef(false)
  const runId = searchParams.get('run') || ''
  const candidateId = searchParams.get('candidate') || ''
  const hypothesisParam = searchParams.get('hypothesis') || ''
  const viewParam = searchParams.get('view')
  const view: AlphaWorkbenchView = isWorkbenchView(viewParam) ? viewParam : 'overview'

  const charterQuery = useQuery({ queryKey: QK.alphaCharter, queryFn: api.alphaCharter, staleTime: Infinity })
  const enginesQuery = useQuery({ queryKey: QK.alphaEngines, queryFn: api.alphaEngines, staleTime: Infinity })
  const configQuery = useQuery({ queryKey: QK.alphaConfig, queryFn: api.alphaConfig })
  const factorsQuery = useQuery({ queryKey: QK.factorColumns, queryFn: api.factorColumns, staleTime: Infinity })
  const hypothesesQuery = useQuery({
    queryKey: QK.alphaHypotheses(draft.assetType, draft.start, draft.end),
    queryFn: () => api.alphaHypotheses({ assetType: draft.assetType, start: draft.start, end: draft.end }),
  })
  const runsQuery = useQuery({ queryKey: QK.alphaRuns, queryFn: api.alphaRuns, refetchInterval: 5000 })
  const experimentsQuery = useQuery({ queryKey: QK.alphaExperiments, queryFn: api.alphaExperiments, refetchInterval: 10000 })
  const candidatesQuery = useQuery({ queryKey: QK.alphaCandidates, queryFn: api.alphaCandidates, refetchInterval: 10000 })
  const championQuery = useQuery({ queryKey: QK.alphaChampion, queryFn: api.alphaChampion, refetchInterval: 10000 })
  const availabilityQuery = useQuery({
    queryKey: QK.alphaAvailability(draft.assetType, draft.profile, draft.horizon, draft.start, draft.end),
    queryFn: () => api.alphaAvailability({
      assetType: draft.assetType,
      budgetProfile: draft.profile,
      forwardHorizon: draft.horizon,
      start: draft.start,
      end: draft.end,
    }),
  })
  const runQuery = useQuery({
    queryKey: QK.alphaRun(runId),
    queryFn: () => api.alphaRun(runId),
    enabled: !!runId,
    retry: false,
    refetchInterval: query => ACTIVE_RUN_STATUSES.has(query.state.data?.status || '') ? 1500 : false,
  })
  const currentRun = runQuery.data || runsQuery.data?.items.find(item => item.run_id === runId) || null
  const selectedHypothesisId = currentRun?.request.hypothesis_id || hypothesisParam
  const selectedHypothesis = hypothesesQuery.data?.items.find(item => item.hypothesis_id === selectedHypothesisId) || null
  const active = !!currentRun && ACTIVE_RUN_STATUSES.has(currentRun.status)
  const resultQuery = useQuery({
    queryKey: QK.alphaResult(runId),
    queryFn: () => api.alphaResult(runId),
    enabled: !!runId && !!currentRun && SUCCESS_RUN_STATUSES.has(currentRun.status),
    retry: false,
  })
  const candidateQuery = useQuery({
    queryKey: QK.alphaCandidate(candidateId),
    queryFn: () => api.alphaCandidate(candidateId),
    enabled: !!candidateId,
    retry: false,
  })
  const evidenceCandidate = candidatesQuery.data?.items.find(item => item.candidate_id === candidateId)
  const shadowQuery = useQuery({
    queryKey: QK.alphaShadow(candidateId),
    queryFn: () => api.alphaShadow(candidateId),
    enabled: !!candidateId && ['shadow', 'challenger', 'champion'].includes(evidenceCandidate?.state.state || ''),
    retry: false,
  })

  const readyEngineIds = useMemo(
    () => new Set(availabilityQuery.data?.engines.filter(item => item.ready).map(item => item.engine_id) || []),
    [availabilityQuery.data],
  )
  const factorMap = useMemo(() => new Map((factorsQuery.data?.columns || []).map(item => [item.id, item])), [factorsQuery.data])
  const engineMap = useMemo(() => new Map((enginesQuery.data?.items || []).map(item => [item.engine_id, item])), [enginesQuery.data])
  const unavailableFactorReasons = useMemo(() => {
    const reasons = new Map<string, string>()
    const financialReady = availabilityQuery.data?.catalog.datasets.financial_pit?.ready === true
    for (const factor of factorsQuery.data?.columns || []) {
      if (factor.group === '财务' && !financialReady) reasons.set(factor.id, '缺少公告时点财务数据，不能进入正式历史研究')
    }
    return reasons
  }, [availabilityQuery.data?.catalog.datasets.financial_pit?.ready, factorsQuery.data?.columns])

  useEffect(() => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  }, [draft])

  useEffect(() => {
    if (draftOrigin) localStorage.setItem(ORIGIN_KEY, JSON.stringify(draftOrigin))
    else localStorage.removeItem(ORIGIN_KEY)
  }, [draftOrigin])

  useEffect(() => {
    const latest = availabilityQuery.data?.available_end
    const earliest = availabilityQuery.data?.available_start
    if (!latest || !earliest || draft.start || draft.end) return
    setDraft(current => ({ ...current, start: yearsBefore(latest, 1), end: latest, datePreset: '1y' }))
  }, [availabilityQuery.data?.available_end, availabilityQuery.data?.available_start, draft.end, draft.start])

  useEffect(() => {
    if (initializedFactors.current || !factorsQuery.data?.columns.length) return
    initializedFactors.current = true
    const known = new Set(factorsQuery.data.columns.filter(item => !unavailableFactorReasons.has(item.id)).map(item => item.id))
    const retained = draft.factorNames.filter(id => known.has(id))
    if (!retained.length) setDraft(current => ({ ...current, factorNames: [...known] }))
  }, [draft.factorNames, factorsQuery.data, unavailableFactorReasons])

  useEffect(() => {
    if (!unavailableFactorReasons.size) return
    const retained = draft.factorNames.filter(id => !unavailableFactorReasons.has(id))
    if (retained.length !== draft.factorNames.length) setDraft(current => ({ ...current, factorNames: retained }))
  }, [draft.factorNames, unavailableFactorReasons])

  useEffect(() => {
    if (!availabilityQuery.data) return
    const retained = draft.engineIds.filter(id => readyEngineIds.has(id))
    if (!initializedEngines.current || (!retained.length && readyEngineIds.size)) {
      initializedEngines.current = true
      setDraft(current => ({ ...current, engineIds: retained.length ? retained : [...readyEngineIds] }))
    } else if (retained.length !== draft.engineIds.length) {
      setDraft(current => ({ ...current, engineIds: retained }))
    }
  }, [availabilityQuery.data, draft.engineIds, readyEngineIds])

  useEffect(() => {
    if (!currentRun || ACTIVE_RUN_STATUSES.has(currentRun.status)) return
    void queryClient.invalidateQueries({ queryKey: QK.alphaRuns })
    void queryClient.invalidateQueries({ queryKey: QK.alphaExperiments })
    void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
  }, [currentRun, queryClient])

  useEffect(() => {
    const result = resultQuery.data
    if (!result || candidateId) return
    const first = result.candidates.find(item => item.candidate_id)?.candidate_id
    if (first) navigate({ candidate: first }, true)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidateId, resultQuery.data])

  const blockers = useMemo(() => readinessBlockers(availabilityQuery.data, availabilityQuery.isError, draft.engineIds), [availabilityQuery.data, availabilityQuery.isError, draft.engineIds])
  const factorGroups = useMemo(() => {
    const output: Record<string, FactorColumn[]> = {}
    const needle = factorSearch.trim().toLowerCase()
    for (const factor of factorsQuery.data?.columns || []) {
      if (needle && !`${factor.label} ${factor.id} ${factor.desc}`.toLowerCase().includes(needle)) continue
      ;(output[factor.group || '其他'] ||= []).push(factor)
    }
    return output
  }, [factorSearch, factorsQuery.data])
  const visibleEngines = useMemo(() => {
    const needle = engineSearch.trim().toLowerCase()
    return (enginesQuery.data?.items || []).filter(engine => !needle || engineSearchText(engine).includes(needle))
  }, [engineSearch, enginesQuery.data])

  const startMutation = useMutation({
    mutationFn: async () => {
      const commission = parseNumber(draft.commissionBps, '佣金', 0, 500)
      const stampTax = parseNumber(draft.stampTaxBps, '印花税', 0, 500)
      const slippage = parseNumber(draft.slippageBps, '滑点', 0, 1000)
      const maxPositions = parseNumber(draft.maxPositions, '最大持仓数', 1, 50, true)
      if ([commission, stampTax, slippage, maxPositions].some(value => value == null)) throw new Error('研究合同参数不合法')
      if (!configQuery.data?.enabled) {
        const config = await api.alphaUpdateConfig({ enabled: true })
        queryClient.setQueryData(QK.alphaConfig, config)
      }
      const budget = PROFILE_META[draft.profile]
      if (selectedHypothesis) {
        return api.alphaHypothesisStart(selectedHypothesis.hypothesis_id, {
          start: draft.start || null,
          end: draft.end || null,
          budget_profile: draft.profile,
          commission_pct: commission! / 10000,
          stamp_tax_pct: stampTax! / 10000,
          slippage_bps: slippage!,
          max_positions: maxPositions!,
        })
      }
      return api.alphaStart({
        engine_ids: draft.engineIds,
        factor_names: draft.factorNames,
        asset_type: draft.assetType,
        start: draft.start || null,
        end: draft.end || null,
        budget_profile: draft.profile,
        forward_horizon: draft.horizon,
        commission_pct: commission! / 10000,
        stamp_tax_pct: stampTax! / 10000,
        slippage_bps: slippage!,
        max_positions: maxPositions!,
        max_candidates_per_engine: budget.candidates,
        max_trials_per_engine: budget.trials,
        source_run_id: draftOrigin?.sourceRunId || null,
        source_suggestion_id: draftOrigin?.suggestionId || null,
      })
    },
    onSuccess: run => {
      setDraftOrigin(null)
      navigate({ run: run.run_id, view: 'overview', candidate: null })
      void queryClient.invalidateQueries({ queryKey: QK.alphaRuns })
      toast('研究任务已启动，刷新页面不会中断运行', 'success')
    },
    onError: error => toast(error instanceof Error ? error.message : '研究启动失败', 'error'),
  })
  const cancelMutation = useMutation({
    mutationFn: api.alphaCancel,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: QK.alphaRuns }),
  })
  const createHypothesisMutation = useMutation({
    mutationFn: api.alphaHypothesisCreate,
    onSuccess: hypothesis => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaHypotheses(draft.assetType, draft.start, draft.end) })
      chooseHypothesis(hypothesis)
      toast('自定义假设已冻结；现在可以直接启动研究', 'success')
    },
    onError: error => toast(error instanceof Error ? error.message : '假设创建失败', 'error'),
  })
  const proposeAIHypothesesMutation = useMutation({
    mutationFn: () => {
      if (!draft.start || !draft.end) throw new Error('请先选择研究区间，DeepSeek需要基于该区间的数据条件提出假设')
      return api.alphaAIHypothesisPropose({
        asset_type: draft.assetType,
        start: draft.start,
        end: draft.end,
        count: 3,
      })
    },
    onSuccess: result => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaHypotheses(draft.assetType, draft.start, draft.end) })
      if (result.items[0]) chooseHypothesis(result.items[0])
      const rejected = result.rejected.length ? `，另有${result.rejected.length}条未通过合同校验` : ''
      toast(`DeepSeek已提出并冻结${result.items.length}条新假设${rejected}`, 'success')
    },
    onError: error => toast(error instanceof Error ? error.message : 'DeepSeek假设生成失败', 'error'),
  })
  const strictValidationMutation = useMutation({
    mutationFn: (id: string) => api.alphaStrictValidation(id),
    onSuccess: run => {
      navigate({ run: run.run_id, view: 'overview', candidate: null })
      void queryClient.invalidateQueries({ queryKey: QK.alphaRuns })
      toast('全历史严格验证已作为独立运行启动', 'success')
    },
    onError: error => toast(error instanceof Error ? error.message : '严格验证启动失败', 'error'),
  })
  const shadowStartMutation = useMutation({
    mutationFn: (id: string) => api.alphaShadowStart(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      void queryClient.invalidateQueries({ queryKey: QK.alphaShadow(candidateId) })
      toast('独立前向账户已创建', 'success')
    },
    onError: error => toast(error instanceof Error ? error.message : '前向账户创建失败', 'error'),
  })
  const shadowEvaluateMutation = useMutation({
    mutationFn: (id: string) => api.alphaShadowEvaluate(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      void queryClient.invalidateQueries({ queryKey: QK.alphaChampion })
      void queryClient.invalidateQueries({ queryKey: QK.alphaShadow(candidateId) })
    },
  })
  const promoteMutation = useMutation({
    mutationFn: (id: string) => api.alphaPromote(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.alphaChampion })
      void queryClient.invalidateQueries({ queryKey: QK.alphaCandidates })
      toast('挑战者已通过发布状态机并晋级', 'success')
    },
  })

  function updateDraft<K extends keyof AlphaWorkbenchDraft>(key: K, value: AlphaWorkbenchDraft[K]) {
    setDraft(current => ({ ...current, [key]: value }))
  }

  function navigate(change: { run?: string; view?: AlphaWorkbenchView; candidate?: string | null }, replace = false) {
    const params = new URLSearchParams(searchParams)
    if (change.run === '') params.delete('run')
    else if (change.run !== undefined) params.set('run', change.run)
    if (change.view !== undefined) params.set('view', change.view)
    if (change.candidate === null) params.delete('candidate')
    else if (change.candidate !== undefined) params.set('candidate', change.candidate)
    setSearchParams(params, { replace })
  }

  function applyDatePreset(preset: '1y' | '3y' | 'all') {
    const latest = availabilityQuery.data?.available_end
    const earliest = availabilityQuery.data?.available_start
    if (!latest || !earliest) return
    setDraft(current => ({
      ...current,
      datePreset: preset,
      start: preset === 'all' ? earliest : yearsBefore(latest, preset === '1y' ? 1 : 3),
      end: latest,
      profile: preset === '1y' ? 'exploratory' : current.profile,
    }))
  }

  function applyProfile(profile: AlphaWorkbenchDraft['profile']) {
    setDraft(current => ({ ...current, profile }))
    if (profile === 'exploratory') applyDatePreset('1y')
    else applyDatePreset('all')
  }

  function startResearch() {
    if (active) return
    if (!selectedHypothesis) return toast('请先选择一个系统假设，或创建自己的Alpha假设', 'error')
    if (selectedHypothesis.readiness && !selectedHypothesis.readiness.ready) {
      return toast(selectedHypothesis.readiness.reasons[0] || '该假设当前数据条件不足', 'error')
    }
    if (!draft.factorNames.length) return toast('请至少选择一个研究因子', 'error')
    const unavailableFactor = draft.factorNames.find(id => unavailableFactorReasons.has(id))
    if (unavailableFactor) return toast(unavailableFactorReasons.get(unavailableFactor)!, 'error')
    if (!draft.engineIds.length) return toast('请至少选择一个可运行的发现引擎', 'error')
    if (blockers.length) return toast(blockers[0], 'error')
    startMutation.mutate()
  }

  function chooseHypothesis(hypothesis: AlphaHypothesis) {
    setDraft(current => ({
      ...current,
      assetType: hypothesis.asset_type,
      horizon: hypothesis.forward_horizon,
      factorNames: [...hypothesis.test_spec.factor_names],
      engineIds: [...hypothesis.test_spec.engine_ids],
    }))
    setDraftOrigin(null)
    const params = new URLSearchParams(searchParams)
    params.set('hypothesis', hypothesis.hypothesis_id)
    params.delete('run')
    params.delete('candidate')
    params.set('view', 'overview')
    setSearchParams(params)
  }

  function createFromFailure(suggestion: AlphaNextResearchSuggestion) {
    if (!currentRun) return
    const nextHypothesis = result?.next_hypotheses?.find(item => item.source_suggestion_id === suggestion.suggestion_id)
    if (nextHypothesis) {
      chooseHypothesis(nextHypothesis)
      toast('已进入由本轮失败证据生成的下一条假设；旧运行保持不可变', 'success')
      return
    }
    setDraft(suggestionToDraft(currentRun.request, suggestion))
    setDraftOrigin({
      sourceRunId: currentRun.run_id,
      suggestionId: suggestion.suggestion_id,
      title: suggestion.title,
      why: suggestion.why,
      keep: suggestion.keep,
      changes: suggestion.changes,
    })
    navigate({ run: '', view: 'overview', candidate: null })
    toast('已创建未启动的新研究草稿；来源运行和旧证据均未修改', 'success')
  }

  function copyFailedRun() {
    if (!currentRun) return
    setDraft(requestToDraft(currentRun.request))
    setDraftOrigin(null)
    navigate({ run: '', view: 'overview', candidate: null })
    toast('已复制失败运行的冻结合同，请修正后创建新实验', 'success')
  }

  const topError = [charterQuery, enginesQuery, hypothesesQuery, configQuery, runsQuery, availabilityQuery].some(query => query.isError)
  const result = resultQuery.data
  const selectedResultCandidate = result?.candidates.find(item => item.candidate_id === candidateId) || null
  const sidebarDraft = currentRun ? requestToDraft(currentRun.request) : draft

  return (
    <div className="flex min-h-full flex-col bg-base">
      <PageHeader
        title="Alpha挖掘"
        subtitle={<span className="hidden md:inline">独立发现、样本外验证与失败研究闭环</span>}
        className="shrink-0 flex-wrap gap-y-2 bg-base/95 px-3 [&_h1]:whitespace-nowrap lg:px-5"
        right={(
          <div className="flex min-w-0 w-full items-center justify-end gap-2 sm:w-auto">
            <RunPicker runs={runsQuery.data?.items || []} selectedId={runId} loading={runsQuery.isFetching} onRefresh={() => void runsQuery.refetch()} onSelect={id => navigate({ run: id, view: 'overview', candidate: null })} />
            <span className="hidden rounded-full border border-border px-2 py-1 text-[9px] text-muted xl:inline">旧挖掘保持独立</span>
          </div>
        )}
      />
      {topError && <div className="border-b border-danger/30 bg-danger/5 px-4 py-2 text-[10px] text-danger">研究服务部分加载失败。页面不会用空值伪装成功，旧挖掘不受影响。</div>}
      <main className="grid min-h-[calc(100vh-7rem)] flex-1 grid-cols-1 overflow-hidden border-b border-border xl:grid-cols-[22rem_minmax(0,1fr)]">
        <ResearchSidebar
          draft={sidebarDraft}
          hypothesis={selectedHypothesis}
          availability={availabilityQuery.data}
          availabilityLoading={availabilityQuery.isFetching}
          blockers={blockers}
          factorGroups={factorGroups}
          unavailableFactorReasons={unavailableFactorReasons}
          factorSearch={factorSearch}
          engineSearch={engineSearch}
          engines={visibleEngines}
          readyEngineIds={readyEngineIds}
          active={active}
          launching={startMutation.isPending}
          locked={!!currentRun}
          onFactorSearch={setFactorSearch}
          onEngineSearch={setEngineSearch}
          onDraft={updateDraft}
          onPreset={applyDatePreset}
          onProfile={applyProfile}
          onStart={startResearch}
          onCancel={() => currentRun && cancelMutation.mutate(currentRun.run_id)}
        />
        <section className="min-w-0 bg-surface xl:max-h-[calc(100vh-7rem)] xl:overflow-y-auto">
          <WorkspaceHeader run={currentRun} result={result} />
          {runId && runQuery.isError ? (
            <InvalidRun onLatest={() => {
              const latest = runsQuery.data?.items[0]
              if (latest) navigate({ run: latest.run_id, view: 'overview', candidate: null })
            }} />
          ) : !currentRun ? (
            <div className="space-y-3 p-3 lg:p-4">
              <HypothesisWorkbench
                items={hypothesesQuery.data?.items || []}
                selected={selectedHypothesis}
                factors={factorsQuery.data?.columns || []}
                loading={hypothesesQuery.isLoading}
                creating={createHypothesisMutation.isPending}
                generatingAI={proposeAIHypothesesMutation.isPending}
                onSelect={chooseHypothesis}
                onCreate={payload => createHypothesisMutation.mutate(payload)}
                onGenerateAI={() => proposeAIHypothesesMutation.mutate()}
              />
              {selectedHypothesis && <ReadyWorkspace draft={draft} origin={draftOrigin} availability={availabilityQuery.data} engines={enginesQuery.data?.items || []} blockers={blockers} />}
            </div>
          ) : ACTIVE_RUN_STATUSES.has(currentRun.status) ? (
            <RunningWorkspace run={currentRun} engines={enginesQuery.data?.items.filter(item => currentRun.request.engine_ids.includes(item.engine_id)) || []} />
          ) : currentRun.status === 'failed' ? (
            <FailedWorkspace run={currentRun} onRetry={copyFailedRun} />
          ) : !SUCCESS_RUN_STATUSES.has(currentRun.status) ? (
            <TerminalWorkspace run={currentRun} />
          ) : resultQuery.isLoading ? (
            <Centered icon={LoaderCircle} title="正在读取不可变研究结果" hint="运行已经结束，正在加载候选与证据。" spin />
          ) : resultQuery.isError || !result ? (
            <Centered icon={AlertTriangle} title="研究结果读取失败" hint="任务状态和结果产物不一致，请刷新或查看审计记录。" danger />
          ) : (
            <CompletedWorkspace
              view={view}
              result={result}
              run={currentRun}
              availability={availabilityQuery.data}
              factors={factorMap}
              engines={engineMap}
              selectedCandidate={selectedResultCandidate}
              candidateEvidence={candidateQuery.data}
              evidenceCandidate={evidenceCandidate}
              shadow={shadowQuery.data}
              leaderboard={championQuery.data}
              experiments={experimentsQuery.data?.items || []}
              onView={next => navigate({ view: next })}
              onCandidate={id => navigate({ candidate: id })}
              onOpenCandidate={id => navigate({ candidate: id, view: 'candidates' })}
              onStrictValidation={() => candidateId && strictValidationMutation.mutate(candidateId)}
              onShadowStart={() => candidateId && shadowStartMutation.mutate(candidateId)}
              onShadowEvaluate={() => candidateId && shadowEvaluateMutation.mutate(candidateId)}
              onPromote={() => candidateId && promoteMutation.mutate(candidateId)}
              onCreateFromFailure={createFromFailure}
            />
          )}
          {runId && <div className="flex items-center gap-1.5 border-t border-border px-3 py-2 text-[9px] text-muted"><Link2 className="h-3 w-3" />当前运行已写入地址；刷新或复制链接会恢复同一结果和候选。</div>}
        </section>
      </main>
    </div>
  )
}

function readinessBlockers(availability: AlphaAvailability | undefined, failed: boolean, engineIds: string[]) {
  if (failed) return ['研究条件检查失败，请刷新后重试']
  if (!availability) return ['正在核验研究数据和外测窗口']
  const reasons: string[] = []
  if (availability.trading_bars < availability.required_bars) reasons.push(`当前只有 ${availability.trading_bars} 个交易日，本档至少需要 ${availability.required_bars} 个`)
  if (availability.outer_folds < 1) reasons.push('当前区间不能形成独立的训练、内选和外测窗口')
  const universe = availability.catalog.datasets.historical_universe
  if (universe && !universe.ready) reasons.push(...universe.reasons.map(translateReason))
  if (!availability.engines.some(item => item.ready)) reasons.push('当前数据下没有可运行的发现引擎')
  if (!engineIds.length) reasons.push('请至少选择一个可运行的发现引擎')
  return [...new Set(reasons)]
}

function RunPicker({ runs, selectedId, loading, onRefresh, onSelect }: { runs: AlphaRun[]; selectedId: string; loading: boolean; onRefresh: () => void; onSelect: (id: string) => void }) {
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <History className="hidden h-3.5 w-3.5 text-muted sm:block" />
      <select className="h-8 max-w-[18rem] rounded-input border border-border bg-surface px-2 text-[10px] text-secondary outline-none" value={selectedId} onChange={event => onSelect(event.target.value)}>
        <option value="">新建研究</option>
        {runs.map(run => <option key={run.run_id} value={run.run_id}>{run.run_id.slice(0, 14)} · {runLabel(run)}</option>)}
      </select>
      <button type="button" title="刷新运行历史" onClick={onRefresh} className="grid h-8 w-8 place-items-center rounded-btn border border-border text-muted hover:text-accent"><RefreshCw className={cn('h-3.5 w-3.5', loading && 'animate-spin')} /></button>
    </div>
  )
}

function ResearchSidebar({
  draft, hypothesis, availability, availabilityLoading, blockers, factorGroups, unavailableFactorReasons, factorSearch, engineSearch, engines, readyEngineIds,
  active, launching, locked, onFactorSearch, onEngineSearch, onDraft, onPreset, onProfile, onStart, onCancel,
}: {
  draft: AlphaWorkbenchDraft
  hypothesis: AlphaHypothesis | null
  availability?: AlphaAvailability
  availabilityLoading: boolean
  blockers: string[]
  factorGroups: Record<string, FactorColumn[]>
  unavailableFactorReasons: Map<string, string>
  factorSearch: string
  engineSearch: string
  engines: AlphaEngineManifest[]
  readyEngineIds: Set<string>
  active: boolean
  launching: boolean
  locked: boolean
  onFactorSearch: (value: string) => void
  onEngineSearch: (value: string) => void
  onDraft: <K extends keyof AlphaWorkbenchDraft>(key: K, value: AlphaWorkbenchDraft[K]) => void
  onPreset: (value: '1y' | '3y' | 'all') => void
  onProfile: (value: AlphaWorkbenchDraft['profile']) => void
  onStart: () => void
  onCancel: () => void
}) {
  const allFactorIds = Object.values(factorGroups).flat().map(item => item.id).filter(id => !unavailableFactorReasons.has(id))
  return (
    <aside className="border-b border-border bg-base/25 xl:max-h-[calc(100vh-7rem)] xl:overflow-y-auto xl:border-b-0 xl:border-r">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div><div className="text-xs font-semibold text-foreground">{locked ? '已冻结实验合同' : '下一轮研究草稿'}</div><div className="mt-0.5 text-[9px] text-muted">{locked ? '当前显示所选历史运行；不可原地修改' : '全市场独立发现 · 收盘决策 · 次日开盘成交'}</div></div>
        <span className="font-mono text-[9px] text-muted">{draft.factorNames.length}因子 / {draft.engineIds.length}引擎</span>
      </div>
      <div className="space-y-4 p-3">
        {hypothesis && <div className="rounded-md border border-accent/30 bg-accent/5 p-3"><div className="text-[8px] text-accent">当前研究假设</div><div className="mt-1 text-[10px] font-medium text-foreground">{hypothesis.title}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">因子、方向和预测期限已预注册；只允许调整研究区间、强度与成本。</div></div>}
        <fieldset disabled={locked} className={cn('space-y-4', locked && 'opacity-70')}>
        <section>
          <SectionLabel icon={Database} title="研究目标与数据范围" />
          <div className="mb-2 flex rounded-btn border border-border bg-base p-0.5">
            {(['1y', '3y', 'all'] as const).map(preset => <button key={preset} type="button" onClick={() => onPreset(preset)} className={cn('h-6 flex-1 rounded px-2 text-[9px]', draft.datePreset === preset ? 'bg-accent text-white' : 'text-muted hover:text-foreground')}>{preset === '1y' ? '近1年' : preset === '3y' ? '近3年' : '全部历史'}</button>)}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <label><span className={LABEL}>资产</span><select disabled={!!hypothesis} className={INPUT} value={draft.assetType} onChange={event => onDraft('assetType', event.target.value as 'stock' | 'etf')}><option value="stock">A股</option><option value="etf">ETF</option></select></label>
            <label><span className={LABEL}>预测目标</span><select disabled={!!hypothesis} className={INPUT} value={draft.horizon} onChange={event => onDraft('horizon', Number(event.target.value) as AlphaWorkbenchDraft['horizon'])}>{[1, 3, 5, 10, 20, 60].map(value => <option key={value} value={value}>未来{value}日净收益</option>)}</select></label>
            <label><span className={LABEL}>开始日期</span><input type="date" className={INPUT} value={draft.start} onChange={event => { onDraft('start', event.target.value); onDraft('datePreset', 'custom') }} /></label>
            <label><span className={LABEL}>结束日期</span><input type="date" className={INPUT} value={draft.end} onChange={event => { onDraft('end', event.target.value); onDraft('datePreset', 'custom') }} /></label>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-1">
            {(Object.entries(PROFILE_META) as [AlphaWorkbenchDraft['profile'], typeof PROFILE_META[AlphaWorkbenchDraft['profile']]][]).map(([id, meta]) => <button key={id} type="button" title={meta.description} onClick={() => onProfile(id)} className={cn('rounded-md border px-1.5 py-2 text-left', draft.profile === id ? 'border-accent/50 bg-accent/10 text-accent' : 'border-border text-muted')}><div className="text-[9px] font-medium">{meta.title}</div><div className="mt-0.5 font-mono text-[7px]">{meta.trials}次/引擎</div></button>)}
          </div>
          {locked ? <div className="mt-2 rounded-md border border-border bg-base/40 p-2 text-[8px] leading-relaxed text-muted">该合同来自不可变历史运行。要修改配置，请在右侧失败闭环中创建一份新草稿。</div> : <Readiness availability={availability} loading={availabilityLoading} blockers={blockers} />}
        </section>

        <section className="border-t border-border pt-3">
          <div className="mb-2 flex items-center justify-between"><SectionLabel icon={FlaskConical} title="因子与特征目录" /><button type="button" disabled={!!hypothesis} className="text-[9px] text-accent disabled:text-muted" onClick={() => onDraft('factorNames', draft.factorNames.length ? [] : allFactorIds)}>{hypothesis ? '方向已冻结' : draft.factorNames.length ? '清空' : '全选'}</button></div>
          <SearchBox value={factorSearch} placeholder="搜索因子名称、ID或含义" onChange={onFactorSearch} />
          <div className="mt-2 max-h-60 space-y-2 overflow-y-auto pr-1">
            {Object.entries(factorGroups).map(([group, factors]) => <div key={group}><div className="mb-1 text-[9px] font-medium text-muted">{group}</div><div className="grid grid-cols-2 gap-1">{factors.map(factor => { const unavailable = unavailableFactorReasons.get(factor.id); return <label key={factor.id} title={unavailable || `${factor.label} · ${factor.desc}`} className={cn('flex min-w-0 items-center gap-1.5 text-[9px]', unavailable || hypothesis ? 'cursor-not-allowed text-muted opacity-60' : 'cursor-pointer text-secondary')}><input type="checkbox" className="h-3 w-3 accent-accent" disabled={!!unavailable || !!hypothesis} checked={draft.factorNames.includes(factor.id)} onChange={() => onDraft('factorNames', draft.factorNames.includes(factor.id) ? draft.factorNames.filter(id => id !== factor.id) : [...draft.factorNames, factor.id])} /><span className="truncate">{factor.label}</span></label> })}</div>{group === '财务' && factors.some(factor => unavailableFactorReasons.has(factor.id)) && <div className="mt-1 text-[8px] leading-relaxed text-warning">公告时点财务数据未就绪，本组已禁用且不会被提交。</div>}</div>)}
            {!Object.keys(factorGroups).length && <div className="text-[9px] text-muted">没有匹配的因子</div>}
          </div>
        </section>

        <section className="border-t border-border pt-3">
          <div className="mb-2 flex items-center justify-between"><SectionLabel icon={GitBranch} title="发现引擎" /><span className="font-mono text-[9px] text-muted">{draft.engineIds.length}个已选</span></div>
          <SearchBox value={engineSearch} placeholder="搜索信息域、机制或引擎" onChange={onEngineSearch} />
          <div className="mt-2 max-h-56 space-y-1.5 overflow-y-auto pr-1">
            {engines.map(engine => {
              const ready = readyEngineIds.has(engine.engine_id)
              const selected = draft.engineIds.includes(engine.engine_id)
              return <label key={engine.engine_id} className={cn('block rounded-md border p-2', selected ? 'border-accent/40 bg-accent/5' : 'border-border', (!ready || hypothesis) && 'cursor-not-allowed opacity-60')}><div className="flex items-start gap-2"><input type="checkbox" className="mt-0.5 h-3 w-3 accent-accent" disabled={!ready || !!hypothesis} checked={selected} onChange={() => onDraft('engineIds', selected ? draft.engineIds.filter(id => id !== engine.engine_id) : [...draft.engineIds, engine.engine_id])} /><div className="min-w-0"><div className="truncate text-[9px] font-medium text-foreground">{engine.name}</div><div className="mt-0.5 line-clamp-2 text-[8px] leading-relaxed text-muted">{ready ? `${engine.discovery_method} · ${engine.economic_mechanism}` : '当前数据或实现条件不满足，详情见实验合同'}</div></div></div></label>
            })}
          </div>
        </section>

        <details className="border-t border-border pt-3">
          <summary className="flex cursor-pointer list-none items-center justify-between"><SectionLabel icon={Settings2} title="成交口径与成本" /><ChevronDown className="h-3 w-3 text-muted" /></summary>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <label><span className={LABEL}>佣金 bp</span><input className={INPUT} value={draft.commissionBps} onChange={event => onDraft('commissionBps', event.target.value)} /></label>
            <label><span className={LABEL}>印花税 bp</span><input className={INPUT} value={draft.stampTaxBps} onChange={event => onDraft('stampTaxBps', event.target.value)} /></label>
            <label><span className={LABEL}>滑点 bp</span><input className={INPUT} value={draft.slippageBps} onChange={event => onDraft('slippageBps', event.target.value)} /></label>
            <label><span className={LABEL}>最大持仓</span><input className={INPUT} value={draft.maxPositions} onChange={event => onDraft('maxPositions', event.target.value)} /></label>
          </div>
          <div className="mt-2 text-[8px] leading-relaxed text-muted">收盘后计算信号，次日开盘成交；等权、A股T+1、涨跌停和停牌不可成交。评分门槛、止损和最长持仓随冻结候选一并记录。</div>
        </details>
        </fieldset>

        <div className="sticky bottom-0 space-y-2 border-t border-border bg-base/95 py-2 backdrop-blur">
          <button type="button" disabled={locked || active || launching || !hypothesis || hypothesis.readiness?.ready === false || !!blockers.length || !draft.factorNames.length || !draft.engineIds.length} onClick={onStart} className="inline-flex h-9 w-full items-center justify-center gap-1.5 rounded-btn bg-accent text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">{launching || active ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}{locked ? '历史运行合同不可编辑' : active ? '研究正在运行' : launching ? '正在冻结实验合同' : !hypothesis ? '先选择一个Alpha假设' : '检验这个假设'}</button>
          {active && <button type="button" onClick={onCancel} className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-btn border border-danger/40 text-[10px] text-danger"><Pause className="h-3 w-3" />取消并保留证据</button>}
        </div>
      </div>
    </aside>
  )
}

function SectionLabel({ icon: Icon, title }: { icon: typeof Database; title: string }) {
  return <div className="flex items-center gap-1.5 text-[10px] font-semibold text-secondary"><Icon className="h-3 w-3" />{title}</div>
}

function SearchBox({ value, placeholder, onChange }: { value: string; placeholder: string; onChange: (value: string) => void }) {
  return <label className="relative block"><Search className="absolute left-2 top-2 h-3 w-3 text-muted" /><input className="h-7 w-full rounded-input border border-border bg-surface pl-7 pr-2 text-[9px] text-foreground outline-none focus:border-accent" value={value} placeholder={placeholder} onChange={event => onChange(event.target.value)} /></label>
}

function Readiness({ availability, loading, blockers }: { availability?: AlphaAvailability; loading: boolean; blockers: string[] }) {
  if (loading && !availability) return <div className="mt-2 flex items-center gap-1 text-[8px] text-muted"><LoaderCircle className="h-3 w-3 animate-spin" />正在核验数据和外测窗口</div>
  if (blockers.length) return <div className="mt-2 rounded-md border border-danger/20 bg-danger/5 p-2 text-[8px] leading-relaxed text-danger"><div className="font-medium">当前不能启动</div><ul className="mt-1 list-disc pl-4">{blockers.map(item => <li key={item}>{item}</li>)}</ul></div>
  return <div className="mt-2 rounded-md border border-success/20 bg-success/5 p-2 text-[8px] text-success"><div className="flex items-center gap-1 font-medium"><CheckCircle2 className="h-3 w-3" />实验合同可冻结</div><div className="mt-1 text-success/80">{availability?.trading_bars}个交易日 · {availability?.outer_folds}个独立外测窗口</div></div>
}

function WorkspaceHeader({ run, result }: { run: AlphaRun | null; result?: AlphaResult }) {
  const continuing = result?.candidates.filter(candidate => ['validation_candidate', 'research_candidate', 'shadow', 'challenger', 'champion'].includes(candidate.state)).length || 0
  return <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-b border-border bg-base/20 px-3 py-2"><div className="min-w-0"><div className="flex items-center gap-2"><span className={cn('h-2 w-2 rounded-full', !run ? 'bg-muted' : ACTIVE_RUN_STATUSES.has(run.status) ? 'animate-pulse bg-accent' : run.status === 'failed' ? 'bg-danger' : 'bg-success')} /><span className="text-[11px] font-medium text-foreground">{run ? statusLabel(run.status) : '准备新研究'}{run?.progress?.label ? ` · ${run.progress.label}` : ''}</span></div>{run && <div className="mt-0.5 truncate font-mono text-[8px] text-muted">{run.run_id}</div>}</div>{result && <div className={cn('rounded-full px-2 py-1 text-[9px]', continuing ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger')}>{continuing ? `${continuing}个候选可继续` : '0个候选通过硬门槛'}</div>}</div>
}

function HypothesisWorkbench({ items, selected, factors, loading, creating, generatingAI, onSelect, onCreate, onGenerateAI }: {
  items: AlphaHypothesis[]
  selected: AlphaHypothesis | null
  factors: FactorColumn[]
  loading: boolean
  creating: boolean
  generatingAI: boolean
  onSelect: (hypothesis: AlphaHypothesis) => void
  onCreate: (payload: AlphaHypothesisCreate) => void
  onGenerateAI: () => void
}) {
  const [showCreate, setShowCreate] = useState(false)
  const factorMap = new Map(factors.map(item => [item.id, item]))
  const sourceOrder: Record<AlphaHypothesis['source_kind'], number> = { ai: 0, manual: 1, failure: 2, prior: 3 }
  const orderedItems = [...items].sort((left, right) => sourceOrder[left.source_kind] - sourceOrder[right.source_kind])
  return <div className="space-y-3">
    <Panel title="第一步：生成或选择要验证的Alpha假设" subtitle="DeepSeek根据当前可用数据、A股机制和未覆盖空间提出假设；确定性代码校验并冻结后才允许研究。">
      <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-2">
        <div className="text-[9px] text-muted">AI只负责提出可证伪问题，不读取样本外答案；这里的任何条目都不是已证明策略。</div>
        <div className="flex shrink-0 gap-2">
          <button type="button" disabled={generatingAI} onClick={onGenerateAI} className="inline-flex h-7 items-center gap-1 rounded-btn bg-accent px-2 text-[9px] font-medium text-white disabled:opacity-50">{generatingAI ? <LoaderCircle className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}{generatingAI ? 'DeepSeek正在提出假设' : '让DeepSeek提出新假设'}</button>
          <button type="button" onClick={() => setShowCreate(value => !value)} className="inline-flex h-7 items-center gap-1 rounded-btn border border-accent/40 px-2 text-[9px] text-accent"><Plus className="h-3 w-3" />提出我的假设</button>
        </div>
      </div>
      {showCreate && <CustomHypothesisForm factors={factors} creating={creating} onCancel={() => setShowCreate(false)} onCreate={payload => { onCreate(payload); setShowCreate(false) }} />}
      {loading ? <div className="flex items-center gap-2 p-4 text-[9px] text-muted"><LoaderCircle className="h-3 w-3 animate-spin" />正在读取假设库</div> : <div className="grid gap-2 p-3 lg:grid-cols-2">{orderedItems.map(item => {
        const active = selected?.hypothesis_id === item.hypothesis_id
        const ready = item.readiness?.ready !== false
        const source = item.source_kind === 'ai' ? 'DeepSeek提出' : item.source_kind === 'prior' ? '内置研究先验' : item.source_kind === 'failure' ? '失败证据生成' : '研究者提出'
        return <div key={item.hypothesis_id} className={cn('rounded-md border p-3', active ? 'border-accent bg-accent/5' : 'border-border bg-base/30')}>
          <div className="flex items-start justify-between gap-3"><div><div className="flex items-center gap-1.5"><span className={cn('rounded px-1.5 py-0.5 text-[7px]', item.source_kind === 'ai' ? 'bg-accent/10 text-accent' : item.source_kind === 'failure' ? 'bg-warning/10 text-warning' : item.source_kind === 'manual' ? 'bg-success/10 text-success' : 'bg-muted/10 text-muted')}>{source}</span><span className={cn('rounded px-1.5 py-0.5 text-[7px]', ready ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger')}>{ready ? '当前可运行' : '数据阻断'}</span></div><div className="mt-2 text-[11px] font-semibold text-foreground">{item.title}</div>{item.source_kind === 'ai' && <div className="mt-0.5 text-[7px] text-muted">模型：{item.provenance?.model || 'DeepSeek'} · 提案凭证已冻结</div>}</div><span className="shrink-0 text-[8px] text-muted">未来{item.forward_horizon}日</span></div>
          <div className="mt-2 text-[9px] leading-relaxed text-secondary">{item.thesis}</div>
          <div className="mt-2 rounded bg-base/60 p-2 text-[8px] leading-relaxed text-muted"><span className="text-foreground">为什么可能成立：</span>{item.mechanism}</div>
          <div className="mt-2 flex flex-wrap gap-1">{item.test_spec.factor_names.map(id => <span key={id} className="rounded border border-border px-1.5 py-1 text-[8px] text-muted">{factorMap.get(id)?.label || id}{item.test_spec.expected_directions[id] > 0 ? ' ↑' : ' ↓'}</span>)}</div>
          {!ready && <div className="mt-2 text-[8px] leading-relaxed text-danger">阻断：{item.readiness?.reasons.join('；')}</div>}
          {item.results?.length ? <div className="mt-2 text-[8px] text-muted">已运行{item.run_ids.length}次 · 最近结论：{item.results.at(-1)?.conclusion}</div> : <div className="mt-2 text-[8px] text-muted">尚未运行；结果可能支持，也可能直接证伪。</div>}
          <button type="button" onClick={() => onSelect(item)} className={cn('mt-3 h-8 w-full rounded-btn border text-[9px] font-medium', active ? 'border-accent bg-accent text-white' : ready ? 'border-accent/40 text-accent hover:bg-accent/5' : 'border-border text-muted')}>{active ? '已选择，检查合同后启动' : ready ? '选择并检验这个假设' : '查看阻断与所需数据'}</button>
        </div>
      })}</div>}
    </Panel>
    {selected && <Panel title="这个假设如何被判定" subtitle="方向、权重和证伪条件会随Run冻结；失败不会被改写成成功。"><div className="grid gap-3 p-3 lg:grid-cols-2"><div><div className="text-[9px] font-medium text-secondary">预期方向</div><div className="mt-2 space-y-1">{selected.test_spec.factor_names.map(id => <div key={id} className="flex items-center justify-between rounded bg-base/40 px-2 py-1.5 text-[8px]"><span className="text-muted">{factorMap.get(id)?.label || id}</span><span className={selected.test_spec.expected_directions[id] > 0 ? 'text-success' : 'text-warning'}>{selected.test_spec.expected_directions[id] > 0 ? '越高越有利' : '越低越有利'} · 权重{Math.round((selected.test_spec.weights[id] || 0) * 100)}%</span></div>)}</div></div><div><div className="text-[9px] font-medium text-secondary">什么情况下承认失败</div><ul className="mt-2 list-disc space-y-1.5 pl-4 text-[8px] leading-relaxed text-muted">{selected.falsification.map(item => <li key={item}>{item}</li>)}</ul></div></div></Panel>}
  </div>
}

function CustomHypothesisForm({ factors, creating, onCancel, onCreate }: { factors: FactorColumn[]; creating: boolean; onCancel: () => void; onCreate: (payload: AlphaHypothesisCreate) => void }) {
  const [title, setTitle] = useState('')
  const [thesis, setThesis] = useState('')
  const [mechanism, setMechanism] = useState('')
  const [horizon, setHorizon] = useState<1 | 3 | 5 | 10 | 20 | 60>(5)
  const [search, setSearch] = useState('')
  const [directions, setDirections] = useState<Record<string, -1 | 1>>({})
  const factorMap = new Map(factors.map(item => [item.id, item]))
  const visible = factors.filter(item => !search.trim() || `${item.label} ${item.id} ${item.desc}`.toLowerCase().includes(search.trim().toLowerCase())).slice(0, 18)
  const selectedIds = Object.keys(directions)
  function submit() {
    if (title.trim().length < 3) return toast('请写清楚假设名称', 'error')
    if (thesis.trim().length < 10) return toast('请写清楚预期什么股票在什么期限内表现更好或更差', 'error')
    if (mechanism.trim().length < 10) return toast('请写清楚这个规律可能存在的A股机制', 'error')
    if (!selectedIds.length) return toast('请至少选择一个事前可观测因子并冻结方向', 'error')
    const weight = 1 / selectedIds.length
    const hasFinancial = selectedIds.some(id => factorMap.get(id)?.group === '财务')
    onCreate({
      title: title.trim(), thesis: thesis.trim(), mechanism: mechanism.trim(), asset_type: 'stock', forward_horizon: horizon,
      information_domains: hasFinancial ? ['price_volume', 'fundamentals'] : ['price_volume'],
      test_spec: { engine_ids: ['cross_sectional_rank'], factor_names: selectedIds, expected_directions: directions, weights: Object.fromEntries(selectedIds.map(id => [id, weight])) },
      falsification: ['训练窗预注册组合没有达到最低效应与方向一致性', '独立样本外净收益、夏普、回撤、多窗口或成本压力任一硬门槛失败'],
      data_requirements: hasFinancial ? ['daily_enriched', 'historical_universe', 'financial_pit'] : ['daily_enriched', 'historical_universe'],
    })
  }
  return <div className="border-b border-border bg-base/30 p-3"><div className="grid gap-3 lg:grid-cols-2"><div className="space-y-2"><label><span className={LABEL}>假设名称</span><input className={INPUT} value={title} onChange={event => setTitle(event.target.value)} placeholder="例如：缩量止跌后的5日修复" /></label><label><span className={LABEL}>可证伪预测</span><textarea className="min-h-16 w-full rounded-input border border-border bg-surface p-2 text-[9px] text-foreground outline-none focus:border-accent" value={thesis} onChange={event => setThesis(event.target.value)} placeholder="哪类股票、预期方向、未来多久" /></label><label><span className={LABEL}>为什么可能成立</span><textarea className="min-h-16 w-full rounded-input border border-border bg-surface p-2 text-[9px] text-foreground outline-none focus:border-accent" value={mechanism} onChange={event => setMechanism(event.target.value)} placeholder="A股制度、行为、信息扩散、流动性或风险补偿机制" /></label><label><span className={LABEL}>预测期限</span><select className={INPUT} value={horizon} onChange={event => setHorizon(Number(event.target.value) as typeof horizon)}>{[1, 3, 5, 10, 20, 60].map(value => <option key={value} value={value}>未来{value}日净收益</option>)}</select></label></div><div><span className={LABEL}>事前因子与预期方向</span><SearchBox value={search} placeholder="搜索因子" onChange={setSearch} /><div className="mt-2 max-h-52 space-y-1 overflow-y-auto">{visible.map(factor => { const direction = directions[factor.id]; return <div key={factor.id} className={cn('flex items-center gap-2 rounded border p-2', direction ? 'border-accent/30 bg-accent/5' : 'border-border')}><button type="button" onClick={() => setDirections(current => { const next = { ...current }; if (next[factor.id]) delete next[factor.id]; else next[factor.id] = 1; return next })} className="min-w-0 flex-1 text-left"><div className="truncate text-[9px] text-foreground">{factor.label}</div><div className="truncate text-[7px] text-muted">{factor.desc}</div></button>{direction && <div className="flex shrink-0 rounded border border-border"><button type="button" onClick={() => setDirections(current => ({ ...current, [factor.id]: 1 }))} className={cn('px-2 py-1 text-[8px]', direction > 0 && 'bg-success/10 text-success')}>高值有利</button><button type="button" onClick={() => setDirections(current => ({ ...current, [factor.id]: -1 }))} className={cn('px-2 py-1 text-[8px]', direction < 0 && 'bg-warning/10 text-warning')}>低值有利</button></div>}</div> })}</div><div className="mt-2 text-[8px] text-muted">已冻结{selectedIds.length}个因子。系统不会根据回测结果反向修改方向。</div></div></div><div className="mt-3 flex justify-end gap-2"><button type="button" onClick={onCancel} className="h-8 rounded-btn border border-border px-3 text-[9px] text-muted">取消</button><button type="button" disabled={creating} onClick={submit} className="h-8 rounded-btn bg-accent px-3 text-[9px] font-medium text-white disabled:opacity-50">{creating ? '正在冻结假设' : '冻结并选择此假设'}</button></div></div>
}

function ReadyWorkspace({ draft, origin, availability, engines, blockers }: { draft: AlphaWorkbenchDraft; origin: AlphaDraftOrigin | null; availability?: AlphaAvailability; engines: AlphaEngineManifest[]; blockers: string[] }) {
  const selected = engines.filter(engine => draft.engineIds.includes(engine.engine_id))
  const engineNames = new Map(engines.map(engine => [engine.engine_id, engine.name]))
  if (origin) return <div className="space-y-3 p-3 lg:p-4"><FailureDraftOrigin origin={origin} engineNames={engineNames} /><ReadyWorkspace draft={draft} origin={null} availability={availability} engines={engines} blockers={blockers} /></div>
  return <div className="space-y-3 p-3 lg:p-4"><Panel title="本次实验合同" subtitle="点击启动后以下内容将冻结，任何变化都会形成新的实验运行。"><div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-3 xl:grid-cols-6"><Metric label="研究对象" value={draft.assetType === 'stock' ? 'A股全市场' : 'ETF全市场'} /><Metric label="预测目标" value={`未来${draft.horizon}日净收益`} /><Metric label="研究区间" value={`${draft.start || '—'} 至 ${draft.end || '—'}`} /><Metric label="研究强度" value={PROFILE_META[draft.profile].title} /><Metric label="因子" value={`${draft.factorNames.length}个`} /><Metric label="发现引擎" value={`${draft.engineIds.length}个`} /></div><div className="grid gap-3 border-t border-border p-3 lg:grid-cols-2"><ContractBlock title="研究流程" rows={[['训练窗发现', '只允许引擎查看训练区'], ['隐藏窗内选', '确定方向、公式和冻结参数'], ['独立样本外', `${availability?.outer_folds || 0}个未见窗口`], ['统一压力测试', '双倍成本、延迟、扰动、容量、集中度']]} /><ContractBlock title="交易与预算" rows={[['成交时钟', '收盘信号 → 次日开盘'], ['费用', `佣金${draft.commissionBps}bp / 印花税${draft.stampTaxBps}bp / 滑点${draft.slippageBps}bp`], ['预算', `每引擎最多${PROFILE_META[draft.profile].trials}次尝试 / ${PROFILE_META[draft.profile].candidates}个候选`], ['现有策略', '不进入发现样本，不作为首任冠军底座']]} /></div></Panel><Panel title="本轮搜索空间" subtitle="发现引擎从完整合格股票池和所选信息域独立寻找规律。"><div className="grid gap-2 p-3 md:grid-cols-2 xl:grid-cols-3">{selected.map(engine => <div key={engine.engine_id} className="rounded-md border border-border bg-base/30 p-3"><div className="text-[10px] font-medium text-foreground">{engine.name}</div><div className="mt-1 text-[8px] text-secondary">{engine.discovery_method}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">信息：{engine.information_domains.map(termLabel).join('、')}<br />机制：{engine.mechanism_classes.map(termLabel).join('、')}</div></div>)}</div></Panel>{blockers.length > 0 && <Panel title="阻断原因" subtitle="解决以下问题后才能冻结实验合同。"><ul className="list-disc space-y-1 p-4 pl-8 text-[10px] text-danger">{blockers.map(item => <li key={item}>{item}</li>)}</ul></Panel>}<DatasetStrip availability={availability} /></div>
}

function FailureDraftOrigin({ origin, engineNames }: { origin: AlphaDraftOrigin; engineNames: Map<string, string> }) {
  return <Panel title="基于失败创建的新研究草稿" subtitle="这是未启动的新合同；来源运行保持不可变，确认差异后再启动。"><div className="grid gap-3 p-3 xl:grid-cols-[18rem_minmax(0,1fr)]"><div className="rounded-md border border-border bg-base/30 p-3"><div className="text-[8px] text-muted">来源运行</div><div className="mt-1 break-all font-mono text-[9px] text-foreground">{origin.sourceRunId}</div><div className="mt-2 text-[10px] font-medium text-accent">{origin.title}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">{origin.why}</div></div><div><div className="text-[9px] font-medium text-secondary">保留不变</div><ul className="mt-1 list-disc space-y-1 pl-4 text-[8px] text-muted">{origin.keep.map(item => <li key={item}>{item}</li>)}</ul><div className="mt-3 text-[9px] font-medium text-secondary">明确改变</div><div className="mt-1 space-y-1.5">{origin.changes.map(change => <div key={change.field} className="rounded-md border border-accent/20 bg-accent/5 p-2 text-[8px]"><div className="font-medium text-foreground">{change.label}</div><div className="mt-1 text-muted"><span className="line-through">{formatDiffValue(change.before, engineNames)}</span><span className="mx-1.5 text-accent">→</span><span className="text-secondary">{formatDiffValue(change.after, engineNames)}</span></div><div className="mt-1 text-accent">原因：{change.reason}</div></div>)}</div></div></div></Panel>
}

function RunningWorkspace({ run, engines }: { run: AlphaRun; engines: AlphaEngineManifest[] }) {
  const progress = run.progress
  const percent = typeof progress?.percent === 'number' ? progress.percent : progress?.total ? ((progress.done || 0) / progress.total) * 100 : 0
  const phases = ['研究面板', '撮合矩阵', '滚动发现与样本外', '压力测试', '冻结证据']
  const phaseIndex = progress?.phase === 'panel' ? 0 : progress?.phase === 'matrix' ? 1 : progress?.phase === 'validation' ? 2 : progress?.phase === 'stress' ? 3 : progress?.phase === 'evidence' || progress?.phase === 'completed' ? 4 : 0
  const engineMap = new Map(engines.map(engine => [engine.engine_id, engine]))
  const engineRows = progress?.engines || []
  const engineStatus = {
    waiting: '等待调度',
    running: '正在发现与外测',
    stress: '正在压力测试',
    completed: '本引擎已完成',
    failed: '本引擎执行失败',
  } as const
  return <div className="space-y-3 p-3 lg:p-4">
    <Panel title="研究任务正在执行" subtitle="任务在独立 worker 中运行；切换页面或刷新不会取消，重新打开同一运行编号会恢复最新事件。">
      <div className="p-4">
        <div className="h-1.5 overflow-hidden rounded-full bg-elevated"><div className="h-full bg-accent transition-all" style={{ width: `${Math.max(2, Math.min(percent, 100))}%` }} /></div>
        <div className="mt-2 flex justify-between gap-3 text-[9px]"><span className="text-secondary">{progress?.label || '正在准备研究环境'}</span><span className="shrink-0 font-mono text-muted">{progress?.total ? `${progress.done || 0}/${progress.total}` : statusLabel(run.status)}</span></div>
        <div className="mt-5 grid grid-cols-2 gap-2 md:grid-cols-5">{phases.map((phase, index) => <div key={phase} className={cn('rounded-md border p-2', index < phaseIndex ? 'border-success/20 bg-success/5' : index === phaseIndex ? 'border-accent/40 bg-accent/5' : 'border-border bg-base/30')}><div className="text-[8px] text-muted">0{index + 1}</div><div className="mt-1 text-[9px] font-medium text-foreground">{phase}</div></div>)}</div>
      </div>
    </Panel>
    <Panel title="真实研究预算" subtitle="以下数字由 worker 在每次发现、冻结、回测和异常发生后写入，不是前端估算。">
      <div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4">
        <Metric label="已用尝试 / 上限" value={`${progress?.trials_used || 0} / ${progress?.trial_limit || '—'}`} />
        <Metric label="已冻结 / 上限" value={`${progress?.frozen_candidates || 0} / ${progress?.candidate_limit || '—'}`} />
        <Metric label="真实回测" value={progress?.backtests || 0} />
        <Metric label="引擎异常" value={progress?.engine_errors || 0} tone={(progress?.engine_errors || 0) > 0 ? 'bad' : undefined} />
      </div>
    </Panel>
    <Panel title="逐引擎进度" subtitle="每个引擎独立运行；单引擎失败会显示原因，但不会中断其他引擎。">
      {engineRows.length ? <div className="grid gap-2 p-3 md:grid-cols-2 xl:grid-cols-3">{engineRows.map(row => {
        const engine = engineMap.get(row.engine_id)
        const active = row.status === 'running' || row.status === 'stress'
        return <div key={row.engine_id} className={cn('rounded-md border p-3', row.status === 'failed' ? 'border-danger/30 bg-danger/5' : active ? 'border-accent/40 bg-accent/5' : row.status === 'completed' ? 'border-success/20 bg-success/5' : 'border-border')}>
          <div className="flex items-center justify-between gap-2"><span className="truncate text-[10px] font-medium text-foreground">{engine?.name || row.engine_id}</span><span className={cn('flex shrink-0 items-center gap-1 text-[8px]', row.status === 'failed' ? 'text-danger' : row.status === 'completed' ? 'text-success' : active ? 'text-accent' : 'text-muted')}>{active && <LoaderCircle className="h-3 w-3 animate-spin" />}{engineStatus[row.status]}</span></div>
          <div className="mt-2 grid grid-cols-4 gap-1"><TinyMetric label="外测窗" value={`${row.folds_done}/${row.folds_total}`} /><TinyMetric label="尝试" value={String(row.trials)} /><TinyMetric label="冻结" value={String(row.selected)} /><TinyMetric label="回测" value={String(row.backtests)} /></div>
          <div className={cn('mt-2 min-h-6 text-[8px] leading-relaxed', row.errors ? 'text-danger' : 'text-muted')}>{row.message || (row.errors ? `${row.errors}次异常，原因已写入证据` : engine?.discovery_method || '等待运行事件')}</div>
        </div>
      })}</div> : <div className="p-4 text-[9px] text-muted">当前仍处于数据面板或撮合矩阵准备阶段；逐引擎事件尚未开始。</div>}
    </Panel>
  </div>
}

function FailedWorkspace({ run, onRetry }: { run: AlphaRun; onRetry: () => void }) {
  return <div className="space-y-3 p-3 lg:p-4"><Panel title="本次运行没有形成研究结果" subtitle="这是执行失败，不是策略被证伪；已经产生的技术证据仍然保留。"><div className="p-4"><div className="rounded-md border border-danger/30 bg-danger/5 p-3 text-[10px] leading-relaxed text-danger"><div className="font-medium">停止阶段：{run.progress?.label || '未记录'}</div><div className="mt-1">{translateRunError(run.error)}</div></div><button type="button" onClick={onRetry} className="mt-3 h-8 rounded-btn border border-border px-3 text-[10px] text-secondary hover:border-accent/40 hover:text-accent">返回配置并创建新实验</button><details className="mt-3 text-[8px] text-muted"><summary className="cursor-pointer">查看技术详情</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded bg-base p-2 font-mono">{run.error || '无'}</pre></details></div></Panel></div>
}

function TerminalWorkspace({ run }: { run: AlphaRun }) {
  return <Centered icon={CircleDashed} title={`任务${statusLabel(run.status)}`} hint="该运行没有可展示的成功结果；可以保留证据后创建新实验。" />
}

function CompletedWorkspace({ view, result, run, availability, factors, engines, selectedCandidate, candidateEvidence, evidenceCandidate, shadow, leaderboard, experiments, onView, onCandidate, onOpenCandidate, onStrictValidation, onShadowStart, onShadowEvaluate, onPromote, onCreateFromFailure }: {
  view: AlphaWorkbenchView
  result: AlphaResult
  run: AlphaRun
  availability?: AlphaAvailability
  factors: Map<string, FactorColumn>
  engines: Map<string, AlphaEngineManifest>
  selectedCandidate: AlphaCandidateResult | null
  candidateEvidence?: { candidate: AlphaEvidenceCandidate; events: Record<string, unknown>[] }
  evidenceCandidate?: AlphaEvidenceCandidate
  shadow?: AlphaShadowStatus
  leaderboard?: AlphaLeaderboard
  experiments: Record<string, unknown>[]
  onView: (view: AlphaWorkbenchView) => void
  onCandidate: (id: string) => void
  onOpenCandidate: (id: string) => void
  onStrictValidation: () => void
  onShadowStart: () => void
  onShadowEvaluate: () => void
  onPromote: () => void
  onCreateFromFailure: (suggestion: AlphaNextResearchSuggestion) => void
}) {
  return <div className="min-w-0"><nav className="sticky top-0 z-10 flex overflow-x-auto border-b border-border bg-surface/95 px-2 backdrop-blur">{VIEW_META.map(item => <button key={item.id} type="button" onClick={() => onView(item.id)} className={cn('h-9 shrink-0 border-b-2 px-3 text-[10px]', view === item.id ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-foreground')}>{item.label}</button>)}</nav>{view === 'overview' && <ResultOverview result={result} run={run} factors={factors} engines={engines} onCandidate={onOpenCandidate} onCreateFromFailure={onCreateFromFailure} />}{view === 'discovery' && <DiscoveryEvidence result={result} factors={factors} engines={engines} />}{view === 'oos' && <OosEvidence result={result} selected={selectedCandidate} factors={factors} onCandidate={onCandidate} />}{view === 'market' && <MarketEvidence result={result} selected={selectedCandidate} factors={factors} onCandidate={onCandidate} />}{view === 'robustness' && <RobustnessEvidence result={result} selected={selectedCandidate} factors={factors} onCandidate={onCandidate} />}{view === 'candidates' && <CandidateWorkbench result={result} selected={selectedCandidate} factors={factors} engines={engines} evidence={candidateEvidence} onCandidate={onCandidate} />}{view === 'forward' && <ForwardWorkbench candidate={evidenceCandidate} shadow={shadow} leaderboard={leaderboard} onStrictValidation={onStrictValidation} onStart={onShadowStart} onEvaluate={onShadowEvaluate} onPromote={onPromote} />}{view === 'audit' && <AuditWorkbench run={run} result={result} availability={availability} experiments={experiments} candidateEvidence={candidateEvidence} />}</div>
}

function ResultOverview({ result, run, factors, engines, onCandidate, onCreateFromFailure }: { result: AlphaResult; run: AlphaRun; factors: Map<string, FactorColumn>; engines: Map<string, AlphaEngineManifest>; onCandidate: (id: string) => void; onCreateFromFailure: (suggestion: AlphaNextResearchSuggestion) => void }) {
  const frozen = result.candidates.filter(item => item.frozen_candidate).length
  const continuing = result.candidates.filter(item => ['validation_candidate', 'research_candidate', 'shadow', 'challenger', 'champion'].includes(item.state)).length
  const best = result.candidates.find(item => item.frozen_candidate)
  if (result.failure_analysis?.zero_pass) return <div><ResultOverview result={{ ...result, failure_analysis: undefined, next_research_suggestions: [] }} run={run} factors={factors} engines={engines} onCandidate={onCandidate} onCreateFromFailure={onCreateFromFailure} /><FailureClosure result={result} engines={engines} onCreate={onCreateFromFailure} /></div>
  return <div className="space-y-3 p-3 lg:p-4"><RunLineagePanel run={run} engines={engines} /><div className={cn('rounded-card border p-4', continuing ? 'border-success/30 bg-success/5' : 'border-danger/30 bg-danger/5')}><div className="text-sm font-semibold text-foreground">{continuing ? `本轮产生 ${continuing} 个可继续验证的研究候选` : '本轮没有候选通过全部历史硬门槛'}</div><div className="mt-1 text-[9px] leading-relaxed text-muted">任务已经完整结束，但“运行完成”不等于找到 Alpha。所有失败尝试、候选和样本外结果已冻结。</div></div><div className="grid grid-cols-2 gap-px overflow-hidden rounded-card border border-border bg-border sm:grid-cols-3 xl:grid-cols-6"><Metric label="发现引擎" value={result.summary.candidate_engine_count} /><Metric label="记录尝试" value={result.summary.trial_count} /><Metric label="冻结候选" value={frozen} /><Metric label="真实回测" value={result.summary.backtest_count} /><Metric label="独立外测窗口" value={result.summary.outer_fold_count} /><Metric label="通过硬门槛" value={continuing} tone={continuing ? 'good' : 'bad'} /></div><div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_22rem]"><Panel title="研究漏斗" subtitle="每一层都使用冻结证据，不把任务成功当作策略有效。"><div className="grid grid-cols-2 gap-2 p-3 md:grid-cols-5">{[['所选引擎', run.request.engine_ids.length], ['记录尝试', result.summary.trial_count], ['冻结方案', frozen], ['进入外测', result.candidates.filter(item => item.metrics.oos_days).length], ['通过门槛', continuing]].map(([label, value], index) => <div key={label} className="relative rounded-md border border-border bg-base/30 p-3"><div className="text-[8px] text-muted">0{index + 1}</div><div className="mt-1 text-lg font-semibold text-foreground">{value}</div><div className="text-[9px] text-secondary">{label}</div></div>)}</div></Panel><Panel title="当前最佳候选" subtitle="最佳不等于合格；必须查看失败门槛。">{best ? <button type="button" disabled={!best.candidate_id} onClick={() => best.candidate_id && onCandidate(best.candidate_id)} className="block w-full p-3 text-left hover:bg-elevated/40"><div className="text-[11px] font-medium text-foreground">{engines.get(best.engine_id)?.name || best.engine_name}</div><div className="mt-1 text-[9px] text-secondary">{factorRule(best, factors)}</div><div className="mt-3 grid grid-cols-3 gap-1"><TinyMetric label="样本外" value={fmtPct(best.metrics.stitched_oos_return)} /><TinyMetric label="夏普" value={fmtNumber(best.metrics.stitched_oos_sharpe)} /><TinyMetric label="最大回撤" value={fmtPct(best.metrics.max_drawdown)} /></div></button> : <Empty text="训练和内选没有形成冻结候选" />}</Panel></div>{result.engine_failures.length > 0 && <Panel title="引擎异常" subtitle="单引擎异常不会污染其他引擎，但必须进入失败证据。"><div className="space-y-1 p-3">{result.engine_failures.map((item, index) => <div key={`${item.engine_id}-${index}`} className="rounded-md border border-danger/20 bg-danger/5 p-2 text-[9px] text-danger">{engines.get(item.engine_id)?.name || item.engine_id} · {item.stage}：{translateRunError(item.error)}</div>)}</div></Panel>}</div>
}

function RunLineagePanel({ run, engines }: { run: AlphaRun; engines: Map<string, AlphaEngineManifest> }) {
  const sourceRunId = run.request.source_run_id
  const sourceSuggestionId = run.request.source_suggestion_id
  const sourceCandidateId = run.request.source_candidate_id
  const sourceDiff = run.request.source_diff
  if (!sourceRunId || (!sourceSuggestionId && !sourceCandidateId) || !sourceDiff) return null
  const engineNames = new Map(Array.from(engines.values()).map(engine => [engine.engine_id, engine.name]))
  return <Panel title="研究血缘" subtitle={sourceCandidateId ? '本运行用于验证上一轮冻结候选的同一发现路径；来源候选和来源结果均未覆盖。' : '本运行由上一轮失败建议人工确认后创建；来源结果未被覆盖。'}><div className="grid gap-3 p-3 lg:grid-cols-[18rem_minmax(0,1fr)]"><div className="rounded-md border border-border bg-base/30 p-3"><div className="text-[8px] text-muted">来源运行</div><div className="mt-1 break-all font-mono text-[9px] text-foreground">{sourceRunId}</div><div className="mt-2 text-[8px] text-muted">{sourceCandidateId ? '来源候选' : '来源建议'}</div><div className="mt-1 break-all font-mono text-[9px] text-secondary">{sourceCandidateId || sourceSuggestionId}</div></div><div><div className="text-[9px] font-medium text-secondary">本次明确差异</div><div className="mt-2 space-y-1.5">{Object.entries(sourceDiff).map(([field, diff]) => <div key={field} className="rounded-md border border-accent/20 bg-accent/5 p-2 text-[8px]"><div className="font-medium text-foreground">{field === 'engine_ids' ? '发现引擎' : field === 'factor_names' ? '研究因子' : field === 'budget_profile' ? '研究强度' : field === 'forward_horizon' ? '预测周期' : field === 'start' ? '开始日期' : field === 'end' ? '结束日期' : field === 'max_trials_per_engine' ? '单引擎试验上限' : field === 'max_candidates_per_engine' ? '单引擎候选上限' : field}</div><div className="mt-1 text-muted"><span className="line-through">{formatDiffValue(diff.before, engineNames)}</span><span className="mx-1.5 text-accent">→</span><span className="text-secondary">{formatDiffValue(diff.after, engineNames)}</span></div></div>)}</div></div></div></Panel>
}

function FailureClosure({ result, engines, onCreate }: { result: AlphaResult; engines: Map<string, AlphaEngineManifest>; onCreate: (suggestion: AlphaNextResearchSuggestion) => void }) {
  const analysis = result.failure_analysis!
  const best = analysis.best_failed_candidate
  const suggestions = result.next_research_suggestions || []
  return <div className="space-y-3 px-3 pb-4 lg:px-4"><Panel title="失败研究结论" subtitle="零通过不是空结果：失败分类、证据和下一轮差异已随本次运行冻结。"><div className="p-3"><div className="text-[11px] font-medium text-foreground">{analysis.conclusion}</div><div className="mt-2 grid grid-cols-2 gap-px overflow-hidden rounded-md bg-border sm:grid-cols-4"><Metric label="所选引擎" value={analysis.funnel.selected_engines} /><Metric label="冻结候选" value={analysis.funnel.frozen_candidates} /><Metric label="完成外测" value={analysis.funnel.outer_evaluated} /><Metric label="通过门槛" value={analysis.funnel.historical_gate_passed} tone="bad" /></div></div></Panel>{best && <Panel title="本轮最接近但仍失败的候选" subtitle="排序只用于定位最值得复盘的失败，不改变淘汰结论。"><div className="grid gap-3 p-3 md:grid-cols-[minmax(0,1fr)_18rem]"><div><div className="text-[11px] font-medium text-foreground">{engines.get(best.engine_id)?.name || best.engine_name}</div><div className="mt-1 break-all font-mono text-[8px] text-muted">冻结方案 {best.recipe_id || '—'}</div><div className="mt-2 flex flex-wrap gap-1">{best.failed_gate_ids.map(id => <span key={id} className="rounded border border-danger/20 bg-danger/5 px-2 py-1 text-[8px] text-danger">{GATE_LABELS[id] || id}</span>)}{best.pending_gate_ids.map(id => <span key={id} className="rounded border border-border px-2 py-1 text-[8px] text-muted">{GATE_LABELS[id] || id}待验证</span>)}</div></div><div className="grid grid-cols-3 gap-1"><TinyMetric label="样本外" value={fmtPct(best.stitched_oos_return)} /><TinyMetric label="夏普" value={fmtNumber(best.stitched_oos_sharpe)} /><TinyMetric label="最大回撤" value={fmtPct(best.max_drawdown)} /></div></div></Panel>}<Panel title="失败归因" subtitle="每类都回答：证据是什么、保留什么、改变什么、为什么改变。"><div className="grid gap-2 p-3 xl:grid-cols-2">{analysis.categories.map(category => <div key={category.id} className={cn('rounded-md border p-3', category.id === analysis.primary_category_id ? 'border-danger/40 bg-danger/5' : 'border-border bg-base/30')}><div className="flex items-start justify-between gap-3"><div><div className="text-[10px] font-medium text-foreground">{category.label}</div><div className="mt-0.5 text-[8px] text-muted">影响 {category.count} 条路径{category.id === analysis.primary_category_id ? ' · 首要失败' : ''}</div></div><span className={cn('rounded px-1.5 py-0.5 text-[7px]', category.severity === 'high' ? 'bg-danger/10 text-danger' : 'bg-warning/10 text-warning')}>{category.severity === 'high' ? '高' : '中'}优先级</span></div><div className="mt-2 text-[8px] leading-relaxed text-muted">{category.why}</div><div className="mt-2 rounded bg-base/60 p-2"><div className="text-[8px] font-medium text-secondary">证据</div><ul className="mt-1 list-disc space-y-1 pl-4 text-[8px] text-muted">{category.evidence.map(item => <li key={item}>{item}</li>)}</ul></div><div className="mt-2 grid gap-2 sm:grid-cols-2"><div><div className="text-[8px] font-medium text-success">保留</div><ul className="mt-1 list-disc space-y-1 pl-4 text-[8px] text-muted">{category.keep.map(item => <li key={item}>{item}</li>)}</ul></div><div><div className="text-[8px] font-medium text-accent">改变</div><ul className="mt-1 list-disc space-y-1 pl-4 text-[8px] text-muted">{category.change.map(item => <li key={item}>{item}</li>)}</ul></div></div></div>)}</div></Panel><Panel title="下一轮研究建议" subtitle="系统不会自动调参或覆盖旧运行；点击后只创建一份未启动草稿，必须人工确认差异。">{suggestions.length ? <div className="grid gap-2 p-3 xl:grid-cols-2">{suggestions.map(suggestion => <div key={suggestion.suggestion_id} className="rounded-md border border-accent/20 bg-accent/5 p-3"><div className="text-[10px] font-medium text-foreground">{suggestion.title}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">{suggestion.why}</div><div className="mt-2 space-y-1">{suggestion.changes.map(change => <div key={change.field} className="text-[8px] text-secondary">改变：{change.label} · {change.reason}</div>)}</div><button type="button" onClick={() => onCreate(suggestion)} className="mt-3 h-8 rounded-btn bg-accent px-3 text-[9px] font-medium text-white">基于失败创建新研究</button></div>)}</div> : <div className="p-4 text-[9px] leading-relaxed text-warning">本轮失败中没有能够仅靠配置安全修复的项。旧证据已冻结；应先补足数据或新增发现引擎，不能用放宽门槛替代。</div>}</Panel>{analysis.excluded_recipe_ids.length > 0 && <details className="rounded-card border border-border p-3 text-[8px] text-muted"><summary className="cursor-pointer text-secondary">本轮已证伪且禁止原地复用的冻结方案</summary><div className="mt-2 space-y-1 font-mono">{analysis.excluded_recipe_ids.map(id => <div key={id}>{id}</div>)}</div></details>}</div>
}

function DiscoveryEvidence({ result, factors, engines }: { result: AlphaResult; factors: Map<string, FactorColumn>; engines: Map<string, AlphaEngineManifest> }) {
  const summaries = result.discovery_summary || result.candidates.map(candidate => ({
    engine_id: candidate.engine_id, engine_name: candidate.engine_name,
    discovery_trials: result.trial_ledger.filter(row => row.engine_id === candidate.engine_id && row.stage === 'discovery').length,
    selection_trials: result.trial_ledger.filter(row => row.engine_id === candidate.engine_id && 'penalized_score' in row).length,
    finite_selection_trials: 0, selected_folds: candidate.folds.filter(row => !!row.recipe_id).length,
    outer_folds: candidate.folds.length, selection_stability: candidate.folds.length ? candidate.folds.filter(row => !!row.recipe_id).length / candidate.folds.length : null,
    best_penalized_score: null, recipes_considered: 0, selected_recipe_id: candidate.frozen_candidate?.recipe_id || null,
    selected_recipe_fold_count: 0, errors: candidate.folds.filter(row => !!row.error).length,
  }))
  return <div className="space-y-3 p-3 lg:p-4"><Panel title="发现漏斗与训练证据" subtitle="只展示训练区发现和隐藏内选的统计；样本外收益从未参与方案选择。"><div className="overflow-x-auto"><table className="w-full min-w-[980px] text-left text-[9px]"><thead className="border-b border-border text-muted"><tr>{['发现引擎', '经济机制与方法', '训练尝试', '隐藏内选', '进入外测', '选择稳定度', '惩罚后最佳分', '执行异常'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody>{summaries.map(row => { const manifest = engines.get(row.engine_id); return <tr key={row.engine_id} className="border-b border-border/60 align-top"><td className="px-3 py-2"><div className="font-medium text-foreground">{manifest?.name || row.engine_name}</div><div className="mt-1 text-[8px] text-muted">{manifest?.family || row.engine_id}</div></td><td className="max-w-sm px-3 py-2"><div className="text-secondary">{manifest?.economic_mechanism || '机制说明未登记'}</div><div className="mt-1 text-muted">方法：{manifest?.discovery_method || '—'}</div></td><td className="px-3 py-2 font-mono">{row.discovery_trials}<div className="text-[8px] text-muted">{row.recipes_considered}种方案</div></td><td className="px-3 py-2 font-mono">{row.finite_selection_trials}/{row.selection_trials}<div className="text-[8px] text-muted">有限分/总尝试</div></td><td className="px-3 py-2 font-mono">{row.selected_folds}/{row.outer_folds}</td><td className="px-3 py-2 font-mono">{fmtPct(row.selection_stability)}</td><td className="px-3 py-2 font-mono">{fmtNumber(row.best_penalized_score, 4)}</td><td className={cn('px-3 py-2 font-mono', row.errors ? 'text-danger' : 'text-success')}>{row.errors}</td></tr>})}</tbody></table></div></Panel><TrialEvidenceLedger rows={result.trial_ledger} engines={engines} /><Panel title="冻结方案" subtitle="公式、方向和交易规则一旦进入外层验证即冻结，失败方案同样保留。"><div className="grid gap-2 p-3 md:grid-cols-2">{result.candidates.map(candidate => <div key={candidate.engine_id} className="rounded-md border border-border bg-base/30 p-3"><div className="flex items-center justify-between gap-2"><span className="text-[10px] font-medium text-foreground">{engines.get(candidate.engine_id)?.name || candidate.engine_name}</span><span className={candidate.frozen_candidate ? 'text-[8px] text-success' : 'text-[8px] text-danger'}>{candidate.frozen_candidate ? '已冻结并外测' : '训练/内选未形成方案'}</span></div><div className="mt-2 text-[9px] leading-relaxed text-secondary">{factorRule(candidate, factors)}</div>{candidate.frozen_candidate && <div className="mt-2 text-[8px] leading-relaxed text-muted">入场分≥{String(candidate.frozen_candidate.parameters.entry_score ?? 70)} · 排名前{String(candidate.frozen_candidate.parameters.top_rank ?? 20)} · 退出分≤{String(candidate.frozen_candidate.parameters.exit_score ?? 40)}</div>}</div>)}</div></Panel></div>
}

function TrialEvidenceLedger({ rows, engines }: { rows: Record<string, unknown>[]; engines: Map<string, AlphaEngineManifest> }) {
  const visible = rows.filter(row => row.stage === 'discovery' || 'penalized_score' in row).slice(0, 100)
  if (!visible.length) return <Panel title="逐次试验证据" subtitle="每次尝试的训练指标和淘汰原因会在这里展示。"><Empty text="本次运行没有产生试验记录" /></Panel>
  return <Panel title="逐次试验证据" subtitle="这些指标只来自训练/内选窗口；状态“未通过”是研究结论，不是系统异常。"><div className="overflow-x-auto"><table className="w-full min-w-[900px] text-left text-[9px]"><thead className="border-b border-border text-muted"><tr>{['引擎', '方案', '阶段', '训练IC', 'ICIR', '方向一致率', '有效交易日', '结论'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody>{visible.map((row, index) => { const evidence = row.evidence && typeof row.evidence === 'object' ? row.evidence as Record<string, unknown> : {}; const metric = evidence.metric && typeof evidence.metric === 'object' ? evidence.metric as Record<string, unknown> : {}; const passed = row.status === 'passed' || row.status === 'selected'; const reason = evidence.reason === 'preregistered_effect_floor' ? '预注册效应未达到训练门槛' : evidence.reason === 'selected' ? '通过训练门槛' : String(evidence.reason || row.status || '已记录'); return <tr key={`${String(row.engine_id)}-${String(row.recipe_id)}-${index}`} className="border-b border-border/60"><td className="px-3 py-2 text-foreground">{engines.get(String(row.engine_id))?.name || String(row.engine_id || '—')}</td><td className="px-3 py-2 font-mono text-muted">{String(row.recipe_id || '—')}</td><td className="px-3 py-2 text-secondary">{row.stage === 'discovery' ? '训练发现' : '隐藏内选'}</td><td className="px-3 py-2 font-mono">{fmtNumber(metric.ic_mean as number | null | undefined, 4)}</td><td className="px-3 py-2 font-mono">{fmtNumber(metric.ic_ir as number | null | undefined, 3)}</td><td className="px-3 py-2 font-mono">{fmtPct(metric.positive_date_ratio as number | null | undefined)}</td><td className="px-3 py-2 font-mono">{String(metric.valid_dates ?? '—')}</td><td className={cn('px-3 py-2', passed ? 'text-success' : 'text-danger')}>{passed ? '通过' : '未通过'} · {reason}</td></tr>})}</tbody></table></div></Panel>
}

function OosEvidence({ result, selected, factors, onCandidate }: { result: AlphaResult; selected: AlphaCandidateResult | null; factors: Map<string, FactorColumn>; onCandidate: (id: string) => void }) {
  if (!selected) return <div className="grid min-h-[32rem] grid-cols-1 lg:grid-cols-[18rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} /><Empty text="选择候选查看样本外证据" /></div>
  const folds = selected.folds.map((fold, index) => ({ label: `外测${fold.outer_index ?? index + 1}`, return: fold.metrics?.total_return ?? null, sharpe: fold.metrics?.sharpe ?? null, row: fold }))
  const trades = selected.folds.flatMap(fold => fold.metrics?.trades || [])
  return <div className="grid min-h-[32rem] grid-cols-1 border-b border-border lg:grid-cols-[18rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} /><div className="min-w-0 space-y-3 p-4"><div><div className="text-xs font-semibold text-foreground">{factorRule(selected, factors)}</div><div className="mt-1 text-[8px] text-muted">训练和隐藏内选结束后，按时间顺序拼接每个从未被引擎看到的外层窗口。</div></div><div className="grid grid-cols-2 gap-px overflow-hidden rounded-md bg-border sm:grid-cols-4"><Metric label="真实外测区间" value={selected.metrics.oos_start && selected.metrics.oos_end ? `${selected.metrics.oos_start} 至 ${selected.metrics.oos_end}` : '未形成'} /><Metric label="拼接样本外" value={fmtPct(selected.metrics.stitched_oos_return)} /><Metric label="样本外夏普" value={fmtNumber(selected.metrics.stitched_oos_sharpe)} /><Metric label="最大回撤" value={fmtPct(selected.metrics.max_drawdown)} /></div><div className="grid gap-3 xl:grid-cols-2"><Panel title="拼接样本外净值" subtitle="起点为1；每段只包含对应外层测试窗。"><EquityCurveChart points={selected.metrics.equity_curve || []} /></Panel><Panel title="逐折样本外收益" subtitle="折间不调参；空折明确显示错误原因。"><EvidenceBarChart points={folds.map(fold => ({ label: fold.label, value: fold.return }))} label="逐折收益" /></Panel></div><Panel title="外层窗口明细" subtitle="每个窗口的训练截止、测试区间、收益、夏普和交易数。"><div className="overflow-x-auto"><table className="w-full min-w-[720px] text-left text-[9px]"><thead className="border-b border-border text-muted"><tr>{['窗口', '训练截止', '测试区间', '收益', '夏普', '最大回撤', '交易', '证据状态'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody>{folds.map(fold => <tr key={fold.label} className="border-b border-border/60"><td className="px-3 py-2 text-foreground">{fold.label}</td><td className="px-3 py-2 font-mono text-muted">{fold.row.train_end || '—'}</td><td className="px-3 py-2 font-mono text-muted">{fold.row.test_start || '—'} 至 {fold.row.test_end || '—'}</td><td className="px-3 py-2 font-mono">{fmtPct(fold.return)}</td><td className="px-3 py-2 font-mono">{fmtNumber(fold.sharpe)}</td><td className="px-3 py-2 font-mono">{fmtPct(fold.row.metrics?.max_drawdown)}</td><td className="px-3 py-2 font-mono">{fold.row.metrics?.n_trades ?? '—'}</td><td className={cn('px-3 py-2', fold.row.error ? 'text-danger' : 'text-success')}>{fold.row.error ? translateRunError(fold.row.error) : '外测完成'}</td></tr>)}</tbody></table></div></Panel><Panel title="成交样本" subtitle="用于核对信号如何变成次日开盘成交，不用单笔案例代替组合证据。">{trades.length ? <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-[9px]"><thead className="border-b border-border text-muted"><tr>{['股票', '入场', '出场', '持有', '单笔收益', '退出原因'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody>{trades.slice(0, 20).map((trade, index) => <tr key={`${trade.symbol}-${trade.entry_date}-${index}`} className="border-b border-border/60"><td className="px-3 py-2 font-mono text-foreground">{trade.symbol}</td><td className="px-3 py-2 font-mono text-muted">{trade.entry_date}</td><td className="px-3 py-2 font-mono text-muted">{trade.exit_date}</td><td className="px-3 py-2">{trade.duration}日</td><td className="px-3 py-2 font-mono">{fmtPct(trade.pnl_pct)}</td><td className="px-3 py-2 text-muted">{trade.exit_reason}</td></tr>)}</tbody></table></div> : <div className="p-4 text-[9px] text-muted">统一撮合没有产生已完成交易，因此不会展示伪造的胜率或收益样本。</div>}</Panel></div></div>
}

function MarketEvidence({ result, selected, factors, onCandidate }: { result: AlphaResult; selected: AlphaCandidateResult | null; factors: Map<string, FactorColumn>; onCandidate: (id: string) => void }) {
  const evidence: AlphaMarketAttribution | undefined = selected ? result.market_attribution?.[selected.engine_id] : undefined
  return <div className="grid min-h-[32rem] grid-cols-1 border-b border-border lg:grid-cols-[18rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} />{!selected ? <Empty text="选择候选查看市场环境与收益来源" /> : !evidence ? <Empty text="这份旧运行不包含市场归因证据；请创建新研究，不会用当前数据倒填旧结果" /> : !evidence.available ? <Empty text={evidence.reason || '没有可归因的样本外收益'} /> : <div className="min-w-0 space-y-3 p-4"><div><div className="text-xs font-semibold text-foreground">{factorRule(selected, factors)}</div><div className="mt-1 text-[8px] text-muted">市场状态按当日全市场等权收益和上涨家数占比定义，再与候选样本外日收益按日期对齐。</div></div><div className="grid grid-cols-2 gap-px overflow-hidden rounded-md bg-border sm:grid-cols-4">{evidence.regimes.map(row => <Metric key={row.state} label={`${row.label} · ${row.days}日`} value={fmtPct(row.return)} tone={row.return == null ? undefined : row.return >= 0 ? 'good' : 'bad'} />)}</div><div className="grid gap-3 xl:grid-cols-2"><Panel title="市场状态收益" subtitle="回答策略究竟在上涨、下跌还是过渡行情赚钱。"><EvidenceBarChart points={evidence.regimes.map(row => ({ label: `${row.label} ${row.days}日`, value: row.return }))} label="市场状态收益" /></Panel><Panel title="年度收益分布" subtitle="识别收益是否只集中在某一年。"><EvidenceBarChart points={evidence.years.map(row => ({ label: `${row.year} ${row.days}日`, value: row.return }))} label="年度收益" /></Panel></div><div className="grid gap-3 xl:grid-cols-2"><Panel title="行业贡献" subtitle="仅使用交易发生当时有效的行业归属。">{evidence.industries.available ? <div className="overflow-x-auto"><table className="w-full min-w-[480px] text-left text-[9px]"><thead className="border-b border-border text-muted"><tr>{['行业', '交易', '盈亏贡献', '胜率'].map(label => <th key={label} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody>{evidence.industries.rows.map(row => <tr key={row.industry} className="border-b border-border/60"><td className="px-3 py-2 text-foreground">{row.industry}</td><td className="px-3 py-2 font-mono">{row.trades}</td><td className="px-3 py-2 font-mono">¥{row.pnl.toFixed(2)}</td><td className="px-3 py-2 font-mono">{fmtPct(row.win_rate)}</td></tr>)}</tbody></table></div> : <div className="p-4 text-[9px] leading-relaxed text-warning">{evidence.industries.reason}</div>}</Panel><Panel title="概念贡献" subtitle="概念成员关系必须具备历史有效期和可用时点。"><div className="p-4"><div className="rounded-md border border-warning/30 bg-warning/5 p-3 text-[9px] leading-relaxed text-warning">{evidence.concepts.reason || '概念历史证据不可用'}</div><div className="mt-2 text-[8px] text-muted">系统没有把当前概念分析页的成员快照用于历史研究，因此这里的空白是数据门禁结果，不是漏画。</div></div></Panel></div>{(result.candidate_correlations || []).length > 0 && <Panel title="候选收益相关性" subtitle="用于判断多个候选是否只是同一个风险暴露的不同写法。"><div className="grid gap-2 p-3 md:grid-cols-2 xl:grid-cols-3">{result.candidate_correlations!.map(row => <div key={`${row.left_engine_id}-${row.right_engine_id}`} className="rounded-md border border-border bg-base/30 p-3"><div className="text-[8px] text-muted">{row.left_engine_id} × {row.right_engine_id}</div><div className="mt-1 font-mono text-sm text-foreground">{fmtNumber(row.correlation)}</div><div className="text-[8px] text-muted">共同样本外 {row.overlap_days} 日</div></div>)}</div></Panel>}</div>}</div>
}

function RobustnessEvidence({ result, selected, factors, onCandidate }: { result: AlphaResult; selected: AlphaCandidateResult | null; factors: Map<string, FactorColumn>; onCandidate: (id: string) => void }) {
  if (!selected) return <div className="grid min-h-[32rem] grid-cols-1 lg:grid-cols-[18rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} /><Empty text="选择候选查看压力测试" /></div>
  const concentration = selected.metrics.stress?.concentration
  const counts = gateCounts(selected)
  const stressPoints = [
    { label: '原始外测', value: selected.metrics.stitched_oos_return ?? null },
    { label: '双倍成本', value: selected.metrics.double_cost_return ?? null },
    { label: '延迟成交', value: selected.metrics.delayed_entry_return ?? null },
    { label: '最差参数扰动', value: selected.metrics.worst_parameter_return ?? null },
    { label: '扩大持仓', value: selected.metrics.capacity_return ?? null },
  ]
  return <div className="grid min-h-[32rem] grid-cols-1 border-b border-border lg:grid-cols-[18rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} /><div className="min-w-0 space-y-3 p-4"><div><div className="text-xs font-semibold text-foreground">{factorRule(selected, factors)}</div><div className="mt-1 text-[8px] text-muted">所有压力情景复用相同外层测试日期和统一撮合，只改变被检验的一个条件。</div></div><div className="grid grid-cols-2 gap-px overflow-hidden rounded-md bg-border sm:grid-cols-5"><Metric label="双倍成本" value={fmtPct(selected.metrics.double_cost_return)} /><Metric label="延迟成交" value={fmtPct(selected.metrics.delayed_entry_return)} /><Metric label="最差参数扰动" value={fmtPct(selected.metrics.worst_parameter_return)} /><Metric label="容量" value={selected.metrics.capacity_passed == null ? '证据不足' : selected.metrics.capacity_passed ? '通过' : '失败'} /><Metric label="集中度" value={selected.metrics.concentration_passed == null ? '证据不足' : selected.metrics.concentration_passed ? '通过' : '失败'} /></div><Panel title="压力情景对照" subtitle="原始外测不是门槛替代物；任何压力测试失败都保留为硬证据。"><EvidenceBarChart points={stressPoints} label="压力情景收益" /></Panel><div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_22rem]"><Panel title="收益集中度" subtitle="检查是否由少数股票、年份或行业贡献。">{concentration ? <div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4"><Metric label="最大单股贡献" value={fmtPct(concentration.top_symbol_share)} /><Metric label="前5股贡献" value={fmtPct(concentration.top5_symbol_share)} /><Metric label="最大单年贡献" value={fmtPct(concentration.top_year_share)} /><Metric label="最大行业贡献" value={fmtPct(concentration.top_industry_share)} /></div> : <div className="p-4 text-[9px] text-muted">没有形成可计算的正收益交易贡献。</div>}</Panel><Panel title="稳健性结论" subtitle="通过数不掩盖任何硬门槛失败。"><div className="p-3"><div className="text-lg font-semibold text-foreground">{counts.passed}过 · <span className="text-danger">{counts.failed}败</span> · <span className="text-muted">{counts.pending}待验证</span></div><div className="mt-1 text-[8px] leading-relaxed text-muted">{counts.failed ? '当前候选已经被历史证据证伪，不进入前向模拟。' : counts.pending ? '历史证据尚不完整，不能晋级。' : '历史硬门槛通过，仍需独立前向模拟。'}</div></div></Panel></div><GateGrid candidate={selected} /></div></div>
}

function CandidateWorkbench({ result, selected, factors, engines, evidence, onCandidate }: { result: AlphaResult; selected: AlphaCandidateResult | null; factors: Map<string, FactorColumn>; engines: Map<string, AlphaEngineManifest>; evidence?: { candidate: AlphaEvidenceCandidate; events: Record<string, unknown>[] }; onCandidate: (id: string) => void }) {
  return <div className="grid min-h-[38rem] grid-cols-1 border-b border-border lg:grid-cols-[20rem_minmax(0,1fr)]"><CandidateList candidates={result.candidates} selected={selected} factors={factors} onCandidate={onCandidate} />{selected?.frozen_candidate ? <CandidateDetail candidate={selected} factorMap={factors} engine={engines.get(selected.engine_id)} evidence={evidence} /> : <Empty text="选择已冻结候选查看完整定义" />}</div>
}

function CandidateList({ candidates, selected, factors, onCandidate }: { candidates: AlphaCandidateResult[]; selected: AlphaCandidateResult | null; factors: Map<string, FactorColumn>; onCandidate: (id: string) => void }) {
  return <div className="border-b border-border lg:border-b-0 lg:border-r"><div className="border-b border-border px-3 py-2 text-[10px] font-medium text-secondary">候选列表</div>{candidates.map(candidate => { const counts = gateCounts(candidate); return <button key={candidate.engine_id} type="button" disabled={!candidate.candidate_id} onClick={() => candidate.candidate_id && onCandidate(candidate.candidate_id)} className={cn('block w-full border-b border-border/60 px-3 py-2 text-left hover:bg-elevated/40 disabled:cursor-default', selected?.engine_id === candidate.engine_id && 'bg-accent/5')}><div className="flex items-center justify-between gap-2"><span className="truncate text-[10px] font-medium text-foreground">{candidate.engine_name}</span><span className={cn('shrink-0 text-[8px]', candidate.state === 'rejected' ? 'text-danger' : 'text-success')}>{statusLabel(candidate.state)}</span></div><div className="mt-1 line-clamp-2 text-[8px] leading-relaxed text-muted">{factorRule(candidate, factors)}</div><div className="mt-1 text-[8px]"><span className="text-success">{counts.passed}过</span> <span className="text-danger">{counts.failed}败</span> <span className="text-muted">{counts.pending}待</span></div></button>})}</div>
}

function CandidateDetail({ candidate, factorMap, engine, evidence }: { candidate: AlphaCandidateResult; factorMap: Map<string, FactorColumn>; engine?: AlphaEngineManifest; evidence?: { candidate: AlphaEvidenceCandidate; events: Record<string, unknown>[] } }) {
  const frozen = candidate.frozen_candidate!
  const range = successfulOosRange(candidate)
  const failed = candidate.gates.filter(gate => gate.status === 'failed')
  return <div className="space-y-3 p-4"><div><div className="flex flex-wrap items-center gap-2"><h2 className="text-sm font-semibold text-foreground">{engine?.name || candidate.engine_name}</h2><span className={cn('rounded-full px-2 py-1 text-[8px]', candidate.state === 'rejected' ? 'bg-danger/10 text-danger' : 'bg-success/10 text-success')}>{statusLabel(candidate.state)}</span></div><div className="mt-1 text-[9px] leading-relaxed text-muted">{frozen.thesis || engine?.economic_mechanism}</div></div><div className="grid gap-3 xl:grid-cols-3"><EvidenceCard title="因子与方向">{frozen.features.map((feature, index) => <div key={feature} className="mb-2 rounded-md bg-base/50 p-2"><div className="text-[9px] font-medium text-secondary">{factorMap.get(feature)?.label || feature} · {(frozen.directions[index] || 1) > 0 ? '高值优先' : '低值优先'}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">{factorMap.get(feature)?.desc || '缺少中文定义'} · 权重{Math.round(Math.abs(frozen.weights[index] || 0) * 100)}%</div></div>)}</EvidenceCard><EvidenceCard title="信号与交易规则"><ol className="list-decimal space-y-1.5 pl-4 text-[8px] leading-relaxed text-muted"><li>每个交易日收盘后，在当时全部合格股票中计算横截面得分。</li><li>得分≥{String(frozen.parameters.entry_score ?? 70)}且排名前{String(frozen.parameters.top_rank ?? 20)}进入买入候选。</li><li>次日开盘成交，等权且最多持有10只；得分≤{String(frozen.parameters.exit_score ?? 40)}退出。</li><li>统一计入佣金、印花税、滑点、T+1、停牌和涨跌停不可成交。</li></ol></EvidenceCard><EvidenceCard title="真实样本外"><div className="grid grid-cols-2 gap-1"><TinyMetric label="区间" value={range ? `${range.start} 至 ${range.end}` : '未形成'} /><TinyMetric label="交易日" value={`${candidate.metrics.oos_days || 0}日`} /><TinyMetric label="拼接收益" value={fmtPct(candidate.metrics.stitched_oos_return)} /><TinyMetric label="夏普" value={fmtNumber(candidate.metrics.stitched_oos_sharpe)} /><TinyMetric label="近1年" value={hasPeriodCoverage(candidate, 'year') ? fmtPct(candidate.metrics.recent_1y_return) : '覆盖不足'} /><TinyMetric label="近3个月" value={hasPeriodCoverage(candidate, 'quarter') ? fmtPct(candidate.metrics.recent_3m_return) : '覆盖不足'} /></div></EvidenceCard></div><div className="rounded-md border border-danger/20 bg-danger/5 p-3"><div className="text-[10px] font-medium text-foreground">裁决原因</div>{failed.length ? <div className="mt-2 flex flex-wrap gap-1.5">{failed.map(gate => <span key={gate.id} className="rounded border border-danger/20 px-2 py-1 text-[8px] text-danger">{GATE_LABELS[gate.id] || gate.id}：{formatGateActual(gate)}</span>)}</div> : <div className="mt-2 text-[9px] text-success">当前已经计算的历史门槛没有失败项。</div>}</div><details className="rounded-md border border-border p-3 text-[8px] text-muted"><summary className="cursor-pointer text-secondary">审计标识与不可变状态</summary><div className="mt-2 break-all font-mono">候选：{candidate.candidate_id || '—'}</div><div className="mt-1 break-all font-mono">方案：{frozen.recipe_id}</div>{evidence && <><div className="mt-1 break-all font-mono">内容指纹：{evidence.candidate.content_sha256}</div><div className="mt-1">状态事件：{evidence.events.length}条</div></>}</details></div>
}

function ForwardWorkbench({ candidate, shadow, leaderboard, onStrictValidation, onStart, onEvaluate, onPromote }: { candidate?: AlphaEvidenceCandidate; shadow?: AlphaShadowStatus; leaderboard?: AlphaLeaderboard; onStrictValidation: () => void; onStart: () => void; onEvaluate: () => void; onPromote: () => void }) {
  const [promotionConfirming, setPromotionConfirming] = useState(false)
  if (!candidate) return <Empty text="先在候选方案中选择一个冻结候选" />
  const state = candidate.state.state
  const stageIndex = ({ validation_candidate: 0, research_candidate: 1, shadow: 2, challenger: 3, champion: 4, retired: 4 } as Record<string, number>)[state] ?? -1
  const evaluation = shadow?.evaluation
  const account = shadow?.account as { positions?: unknown[]; status?: string } | undefined
  const action = state === 'validation_candidate' ? { label: '创建全历史严格验证运行', run: onStrictValidation } : state === 'research_candidate' ? { label: '创建独立前向账户', run: onStart } : state === 'shadow' ? { label: '核对前向证据并判定', run: onEvaluate } : state === 'challenger' ? { label: '准备发布并晋级冠军', run: () => setPromotionConfirming(true) } : null
  const stateHint = state === 'validation_candidate' ? '历史指标初筛通过，但研究不是全历史严格档，不能进入前向。' : state === 'research_candidate' ? '全历史严格验证与全部压力门槛通过，可以建立独立前向账户。' : state === 'shadow' ? '正在积累真实信号、订单、成交、对账与因子衰减证据；不足时保持原状态。' : state === 'challenger' ? '前向门槛已经通过；发布前会重新核对当前动态冠军，过期比较不能晋级。' : state === 'champion' ? '该候选是当前正式冠军，后续新研究只在公共验证末端与它同口径比较。' : state === 'retired' ? '该候选是历史冠军，已被证据更强的新冠军替代。' : '该候选已证伪或尚未完成严格验证，不能进入下一阶段。'
  return <div className="space-y-3 p-3 lg:p-4"><Panel title="候选生命周期" subtitle="短窗口发现、全历史严格验证、前向、挑战者和冠军逐级分离；任何变化都生成不可变事件。"><div className="grid grid-cols-2 gap-2 p-3 md:grid-cols-5">{['待严格验证', '严格验证通过', '前向模拟', '挑战者', '正式冠军'].map((label, index) => <div key={label} className={cn('rounded-md border p-3', stageIndex >= index ? 'border-accent/40 bg-accent/5' : 'border-border')}><div className="text-[8px] text-muted">0{index + 1}</div><div className="mt-1 text-[9px] font-medium text-foreground">{label}</div></div>)}</div><div className={cn('mx-3 mb-3 rounded-md border p-3 text-[9px] leading-relaxed', action ? 'border-accent/20 bg-accent/5 text-secondary' : 'border-border text-muted')}>{stateHint}{action && <button type="button" onClick={action.run} className="mt-3 block h-8 rounded-btn bg-accent px-3 text-[9px] font-medium text-white">{action.label}</button>}{state === 'challenger' && promotionConfirming && <div className="mt-3 rounded-md border border-warning/40 bg-warning/5 p-3"><div className="font-medium text-warning">确认发布并替换当前冠军</div><div className="mt-1 text-[8px] leading-relaxed text-muted">系统将重新核对当前动态冠军，生成可交易策略文件，并写入不可变晋级记录。当前冠军：{leaderboard?.champion.strategy_id || '无（将产生首任冠军）'}。</div><div className="mt-3 flex flex-wrap gap-2"><button type="button" onClick={() => { setPromotionConfirming(false); onPromote() }} className="h-8 rounded-btn bg-warning px-3 text-[9px] font-medium text-base">确认发布</button><button type="button" onClick={() => setPromotionConfirming(false)} className="h-8 rounded-btn border border-border px-3 text-[9px] text-secondary">取消</button></div></div>}</div></Panel><Panel title="前向证据" subtitle="没有真实成交时收益保持为零；证据不足不升级，发生漂移则暂停且不自动调参。">{evaluation ? <><div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4 xl:grid-cols-8"><Metric label="交易日" value={`${evaluation.trading_days}/${evaluation.required_trading_days}`} /><Metric label="信号/订单/成交" value={`${evaluation.signals}/${evaluation.orders}/${evaluation.fills}`} /><Metric label="最低成交" value={evaluation.required_fills} /><Metric label="当前持仓" value={Array.isArray(account?.positions) ? account.positions.length : 0} /><Metric label="前向净收益" value={fmtPct(evaluation.total_return)} /><Metric label="最大回撤" value={fmtPct(evaluation.max_drawdown)} /><Metric label="实际滑点" value={evaluation.average_slippage_bps == null ? '尚无成交' : `${evaluation.average_slippage_bps.toFixed(2)}bp`} /><Metric label="因子Rank IC" value={fmtNumber(evaluation.factor_decay.rank_ic)} /></div><div className="grid gap-2 border-t border-border p-3 sm:grid-cols-2 xl:grid-cols-4">{[['账实对账', evaluation.reconcile_ok], ['信号订单闭环', evaluation.signal_order_parity], ['无伪造收益', evaluation.no_synthetic_profit], ['漂移检测', !evaluation.drift_detected]].map(([label, passed]) => <div key={String(label)} className={cn('rounded-md border p-2 text-[8px]', passed ? 'border-success/20 bg-success/5 text-success' : 'border-danger/20 bg-danger/5 text-danger')}>{label}：{passed ? '通过' : '失败'}</div>)}</div><div className="border-t border-border px-3 py-2 text-[8px] text-muted">因子衰减：完成 {evaluation.factor_decay.completed_round_trips}/{evaluation.required_factor_round_trips} 个闭环，状态为{evaluation.factor_decay.status === 'passed' ? '通过' : evaluation.factor_decay.status === 'failed' ? '失败' : '证据不足'}；账户状态：{account?.status || '运行中'}。</div></> : <div className="p-4 text-[10px] text-muted">尚未建立独立前向账户；页面不会用回测收益代替前向证据。</div>}</Panel><Panel title="动态冠军" subtitle="发现阶段没有固定策略底座；首任冠军看绝对门槛，后续挑战者必须使用冻结时的同一动态冠军并在发布前复核。"><div className="grid gap-3 p-3 lg:grid-cols-[minmax(0,1fr)_20rem]"><div><div className="text-[9px] text-muted">当前正式冠军</div><div className="mt-1 text-xs font-semibold text-foreground">{leaderboard?.champion.strategy_id || '尚无正式冠军'}</div><div className="mt-1 text-[8px] leading-relaxed text-muted">{leaderboard?.champion.reason}</div>{leaderboard?.champion.metrics && <div className="mt-3 grid grid-cols-3 gap-1"><TinyMetric label="样本外" value={fmtPct(leaderboard.champion.metrics.stitched_oos_return)} /><TinyMetric label="夏普" value={fmtNumber(leaderboard.champion.metrics.stitched_oos_sharpe)} /><TinyMetric label="最大回撤" value={fmtPct(leaderboard.champion.metrics.max_drawdown)} /></div>}</div><div className="rounded-md border border-border bg-base/30 p-3"><div className="text-[9px] font-medium text-secondary">所选候选</div><div className="mt-1 break-all font-mono text-[8px] text-muted">{candidate.candidate_id}</div><div className="mt-2 text-[8px] text-muted">当前状态：{statusLabel(state)}</div><div className="mt-1 text-[8px] text-muted">已替换冠军：{leaderboard?.history?.filter(item => item.candidate_id).length || 0} 任</div></div></div></Panel></div>
}

function AuditWorkbench({ run, result, availability, experiments, candidateEvidence }: { run: AlphaRun; result: AlphaResult; availability?: AlphaAvailability; experiments: Record<string, unknown>[]; candidateEvidence?: { candidate: AlphaEvidenceCandidate; events: Record<string, unknown>[] } }) {
  return <div className="space-y-3 p-3 lg:p-4"><Panel title="实验审计" subtitle="技术标识与原始状态集中在此，不污染研究主结论。"><div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4"><Metric label="运行编号" value={run.run_id} /><Metric label="算法版本" value={result.algorithm_version} /><Metric label="数据截至" value={result.data_as_of} /><Metric label="数据指纹" value={availability?.catalog.fingerprint?.slice(0, 16) || '—'} /></div><div className="grid gap-3 border-t border-border p-3 lg:grid-cols-2"><ContractBlock title="冻结请求" rows={Object.entries(result.request_summary).map(([key, value]) => [key, typeof value === 'object' ? JSON.stringify(value) : String(value ?? '—')])} /><ContractBlock title="证据留存" rows={[['实验总数', String(experiments.length)], ['本次试验记录', String(result.trial_ledger.length)], ['引擎异常', String(result.engine_failures.length)], ['候选状态事件', String(candidateEvidence?.events.length || 0)]]} /></div></Panel><details className="rounded-card border border-border bg-surface p-3 text-[8px] text-muted"><summary className="cursor-pointer text-secondary">查看技术请求与失败记录</summary><pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-all rounded bg-base p-3 font-mono">{JSON.stringify({ request: result.request_summary, failures: result.engine_failures }, null, 2)}</pre></details></div>
}

function DatasetStrip({ availability }: { availability?: AlphaAvailability }) {
  return <Panel title="数据资格" subtitle="缺少时点证据的数据必须失败闭合；当前概念快照不进入历史研究。"><div className="grid gap-2 p-3 sm:grid-cols-2 xl:grid-cols-3">{Object.entries(availability?.catalog.datasets || {}).map(([id, item]) => { const meta = DATASET_LABELS[id] || { title: id, description: '' }; return <div key={id} className={cn('rounded-md border p-3', item.ready ? 'border-success/20 bg-success/5' : 'border-danger/20 bg-danger/5')}><div className="flex items-start justify-between gap-2"><div><div className="text-[10px] font-medium text-foreground">{meta.title}</div><div className="mt-1 text-[8px] text-muted">{meta.description}</div></div>{item.ready ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <AlertTriangle className="h-3.5 w-3.5 text-danger" />}</div><div className={cn('mt-2 text-[8px] leading-relaxed', item.ready ? 'text-success' : 'text-danger')}>{item.ready ? '时点防泄漏检查通过' : item.reasons.map(translateReason).join('；')}</div></div>})}</div></Panel>
}

function GateGrid({ candidate }: { candidate: AlphaCandidateResult }) {
  return <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{candidate.gates.map(gate => <div key={gate.id} className={cn('rounded-md border p-3', gate.status === 'passed' ? 'border-success/20 bg-success/5' : gate.status === 'failed' ? 'border-danger/20 bg-danger/5' : 'border-border')}><div className="flex items-center justify-between gap-2"><span className="text-[9px] font-medium text-foreground">{GATE_LABELS[gate.id] || gate.id}</span><span className={cn('text-[8px]', gate.status === 'passed' ? 'text-success' : gate.status === 'failed' ? 'text-danger' : 'text-muted')}>{gate.status === 'passed' ? '通过' : gate.status === 'failed' ? '失败' : '待验证'}</span></div><div className="mt-1 text-[8px] text-muted">实际：{formatGateActual(gate)} · 门槛：{formatGateRequired(gate)}</div></div>)}</div>
}

function InvalidRun({ onLatest }: { onLatest: () => void }) {
  return <div className="p-4"><div className="rounded-card border border-danger/30 bg-danger/5 p-4"><div className="text-sm font-semibold text-danger">地址中的研究运行不存在或无法读取</div><div className="mt-1 text-[9px] text-muted">系统不会静默切换到另一份结果。</div><button type="button" onClick={onLatest} className="mt-3 h-8 rounded-btn border border-border px-3 text-[10px] text-secondary">打开最近运行</button></div></div>
}

function Panel({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return <section className="overflow-hidden rounded-card border border-border bg-surface"><div className="border-b border-border px-3 py-2"><div className="text-[11px] font-semibold text-foreground">{title}</div>{subtitle && <div className="mt-0.5 text-[8px] leading-relaxed text-muted">{subtitle}</div>}</div>{children}</section>
}

function ContractBlock({ title, rows }: { title: string; rows: (string | number)[][] }) {
  return <div><div className="mb-2 text-[10px] font-medium text-foreground">{title}</div><div className="space-y-1.5">{rows.map(([label, value]) => <div key={String(label)} className="grid grid-cols-[7rem_minmax(0,1fr)] gap-2 text-[8px]"><span className="text-muted">{label}</span><span className="break-words text-secondary">{value}</span></div>)}</div></div>
}

function EvidenceCard({ title, children }: { title: string; children: React.ReactNode }) {
  return <div className="rounded-md border border-border bg-surface p-3"><div className="mb-2 text-[10px] font-medium text-foreground">{title}</div>{children}</div>
}

function formatDiffValue(value: unknown, engineNames: Map<string, string>) {
  if (Array.isArray(value)) return value.map(item => engineNames.get(String(item)) || String(item)).join('、') || '无'
  if (value === 'exploratory') return '快速研究'
  if (value === 'balanced') return '标准研究'
  if (value === 'strict') return '严格研究'
  return value == null ? '无' : String(value)
}

function Metric({ label, value, tone }: { label: string; value: string | number; tone?: 'good' | 'bad' }) {
  return <div className="min-w-0 bg-surface px-3 py-2"><div className="text-[8px] text-muted">{label}</div><div className={cn('mt-0.5 truncate font-mono text-[11px] font-semibold text-foreground', tone === 'good' && 'text-success', tone === 'bad' && 'text-danger')} title={String(value)}>{value}</div></div>
}

function TinyMetric({ label, value }: { label: string; value: string }) {
  return <div className="rounded bg-base/60 p-2"><div className="text-[7px] text-muted">{label}</div><div className="mt-0.5 truncate text-[9px] font-medium text-foreground" title={value}>{value}</div></div>
}

function Empty({ text }: { text: string }) {
  return <div className="grid min-h-40 place-items-center p-6 text-center"><div><CircleDashed className="mx-auto h-6 w-6 text-muted" /><div className="mt-2 text-[10px] text-muted">{text}</div></div></div>
}

function Centered({ icon: Icon, title, hint, spin = false, danger = false }: { icon: typeof FileSearch; title: string; hint: string; spin?: boolean; danger?: boolean }) {
  return <div className="grid min-h-[32rem] place-items-center p-6 text-center"><div><Icon className={cn('mx-auto h-8 w-8', danger ? 'text-danger' : 'text-muted', spin && 'animate-spin')} /><div className="mt-3 text-sm font-semibold text-foreground">{title}</div><div className="mt-1 max-w-lg text-[10px] leading-relaxed text-muted">{hint}</div></div></div>
}
