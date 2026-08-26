import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Calendar, Download, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

type Dataset = 'auction' | 'irm_qa'
const PRESETS: readonly (readonly [number, string])[] = [
  [30, '1 个月'],
  [90, '3 个月'],
  [365, '1 年'],
]

function parseDate(value: string): Date {
  return new Date(`${value}T12:00:00`)
}

function dateText(value: Date): string {
  const year = value.getFullYear()
  const month = String(value.getMonth() + 1).padStart(2, '0')
  const day = String(value.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function shiftDays(value: string, amount: number): string {
  const next = parseDate(value)
  next.setDate(next.getDate() + amount)
  return dateText(next)
}

function defaultRange(earliestDate: string | null, days = 30): [string, string] {
  const today = dateText(new Date())
  const end = earliestDate ? shiftDays(earliestDate, -1) : today
  return [shiftDays(end, -(days - 1)), end]
}

export function TushareSupplementalBackfill({
  dataset,
  earliestDate,
  isRunning,
  onJobStart,
}: {
  dataset: Dataset
  earliestDate: string | null
  isRunning: boolean
  onJobStart: (jobId: string) => void
}) {
  const qc = useQueryClient()
  const label = dataset === 'auction' ? '集合竞价' : '董秘问答'
  const prefs = useQuery({ queryKey: QK.preferences, queryFn: api.preferences })
  const autoEnabled = prefs.data?.tushare_supplemental_sync_enabled ?? false
  const toggleAuto = useMutation({
    mutationFn: () => api.updateTushareSupplementalSync(!autoEnabled),
    onSuccess: () => qc.invalidateQueries({ queryKey: QK.preferences }),
  })

  const initial = defaultRange(earliestDate)
  const [startDate, setStartDate] = useState(initial[0])
  const [endDate, setEndDate] = useState(initial[1])
  useEffect(() => {
    const [start, end] = defaultRange(earliestDate)
    setStartDate(start)
    setEndDate(end)
  }, [earliestDate])

  const dayCount = useMemo(() => {
    if (!startDate || !endDate || startDate > endDate) return 0
    return Math.floor((parseDate(endDate).getTime() - parseDate(startDate).getTime()) / 86_400_000) + 1
  }, [startDate, endDate])
  const estimatedRequests = dayCount * 2
  const today = dateText(new Date())
  const invalid = dayCount < 1 || dayCount > 3660 || endDate > today

  const backfill = useMutation({
    mutationFn: () => api.backfillTushareSupplemental(dataset, startDate, endDate),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: QK.pipelineJobs })
      onJobStart(result.job_id)
    },
  })

  const applyPreset = (days: number) => {
    const [start, end] = defaultRange(earliestDate, days)
    setStartDate(start)
    setEndDate(end)
  }

  return (
    <div className="space-y-4">
      <div className="rounded-card border border-border bg-base/30 p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-foreground">每日盘后自动同步</div>
            <div className="text-[10px] text-muted mt-0.5">每天回看近 3 日；集合竞价与董秘问答共用此开关。</div>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={autoEnabled}
            onClick={() => toggleAuto.mutate()}
            disabled={toggleAuto.isPending}
            className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${autoEnabled ? 'bg-accent' : 'bg-elevated'} disabled:opacity-40`}
          >
            <span className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform ${autoEnabled ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
          </button>
        </div>
      </div>

      <div className="rounded-card border border-border bg-base/30 p-4 space-y-3">
        <div>
          <div className="flex items-center gap-1.5 text-sm font-medium text-foreground">
            <Calendar className="h-4 w-4 text-accent" />
            历史补采
          </div>
          <div className="text-[10px] text-muted mt-1 leading-relaxed">
            {earliestDate ? <>本地最早日期为 <span className="font-mono text-secondary">{earliestDate}</span>，快捷选项会从它的前一天继续向前补。</> : '本地暂无数据，快捷选项以今天为结束日期。'}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {PRESETS.map(([days, text]) => (
            <button
              key={days}
              type="button"
              onClick={() => applyPreset(days)}
              disabled={isRunning || backfill.isPending}
              className="rounded-btn border border-border bg-elevated px-2 py-1.5 text-[11px] text-secondary hover:border-accent/40 hover:text-foreground disabled:opacity-40 transition-colors"
            >
              向前补 {text}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1">
            <span className="text-[10px] text-muted">开始日期</span>
            <input
              type="date"
              value={startDate}
              max={endDate || today}
              onChange={(event) => setStartDate(event.target.value)}
              disabled={isRunning || backfill.isPending}
              className="w-full rounded-btn border border-border bg-elevated px-2.5 py-1.5 text-xs font-mono text-foreground outline-none focus:border-accent disabled:opacity-40"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-muted">结束日期</span>
            <input
              type="date"
              value={endDate}
              min={startDate}
              max={today}
              onChange={(event) => setEndDate(event.target.value)}
              disabled={isRunning || backfill.isPending}
              className="w-full rounded-btn border border-border bg-elevated px-2.5 py-1.5 text-xs font-mono text-foreground outline-none focus:border-accent disabled:opacity-40"
            />
          </label>
        </div>

        <div className="rounded-btn border border-accent/20 bg-accent/5 px-3 py-2 text-[10px] text-secondary leading-relaxed">
          共 {dayCount || 0} 个自然日，预计拆成 {estimatedRequests || 0} 个小请求。按日落盘并增量合并，重复日期会去重；不会触发分钟 K 或另一个特色数据集。
          {dataset === 'irm_qa' && <div className="mt-1 text-muted">上证历史自 2023-06 起，深证历史自 2010-10 起；按回复发布日期补采。</div>}
        </div>

        <button
          type="button"
          onClick={() => backfill.mutate()}
          disabled={isRunning || backfill.isPending || invalid}
          className="w-full inline-flex items-center justify-center gap-1.5 rounded-btn bg-accent px-3 py-2 text-xs font-medium text-base hover:bg-accent/90 disabled:opacity-40 transition-colors"
        >
          {backfill.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
          {backfill.isPending ? '正在创建任务…' : `开始补采${label}`}
        </button>
        {dayCount > 3660 && <div className="text-[10px] text-danger">单次最多 10 年，请分段补采。</div>}
        {backfill.isError && <div className="text-[10px] text-danger">{String((backfill.error as Error)?.message ?? backfill.error)}</div>}
      </div>
    </div>
  )
}
