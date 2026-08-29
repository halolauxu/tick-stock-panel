import { useMemo } from 'react'
import type { EChartsOption } from 'echarts'
import { useECharts } from '@/pages/backtest/charts/useECharts'
import { useChartTheme } from '@/lib/theme'

interface EquityPoint { date: string; value: number }
interface BarPoint { label: string; value: number | null }

function valid(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function ChartFrame({ option, empty, label, height = 260 }: { option: EChartsOption | null; empty: string; label: string; height?: number }) {
  const ref = useECharts(option, [option])
  return (
    <div className="relative w-full min-w-0 overflow-hidden" style={{ height }}>
      <div ref={ref} className="h-full w-full" role="img" aria-label={label} />
      {!option && <div className="absolute inset-0 grid place-items-center bg-surface text-[10px] text-muted">{empty}</div>}
    </div>
  )
}

export function EquityCurveChart({ points }: { points: EquityPoint[] }) {
  const ct = useChartTheme()
  const data = useMemo(() => points.filter(point => valid(point.value)), [points])
  const option = useMemo<EChartsOption | null>(() => {
    if (!data.length) return null
    return {
      animation: false,
      grid: { left: 52, right: 18, top: 26, bottom: 44 },
      tooltip: {
        trigger: 'axis', confine: true, backgroundColor: ct.tooltipBg,
        borderColor: ct.tooltipBorder, textStyle: { color: ct.tooltipText, fontSize: 11 },
        valueFormatter: value => `${((Number(value) - 1) * 100).toFixed(2)}%`,
      },
      xAxis: {
        type: 'category', data: data.map(point => point.date), boundaryGap: false,
        axisLine: { lineStyle: { color: ct.border } }, axisTick: { show: false },
        axisLabel: { color: ct.text, fontSize: 9, hideOverlap: true },
      },
      yAxis: {
        type: 'value', scale: true,
        axisLabel: { color: ct.text, fontSize: 9, formatter: value => `${((Number(value) - 1) * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: ct.grid } },
      },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 5, borderColor: ct.border, fillerColor: ct.zoomFill, showDetail: false }],
      series: [{
        name: '样本外累计收益', type: 'line', data: data.map(point => point.value),
        showSymbol: false, smooth: false, lineStyle: { width: 1.5, color: '#3b82f6' },
        areaStyle: { color: 'rgba(59,130,246,.10)' },
        markLine: { silent: true, symbol: 'none', lineStyle: { color: ct.border }, data: [{ yAxis: 1 }] },
      }],
    }
  }, [ct, data])
  return <ChartFrame option={option} empty="没有形成可绘制的样本外净值" label="拼接样本外累计收益曲线" />
}

export function EvidenceBarChart({ points, label, valueKind = 'percent', height = 240 }: { points: BarPoint[]; label: string; valueKind?: 'percent' | 'number'; height?: number }) {
  const ct = useChartTheme()
  const display = useMemo(() => points.filter(point => valid(point.value)), [points])
  const option = useMemo<EChartsOption | null>(() => {
    if (!display.length) return null
    const multiplier = valueKind === 'percent' ? 100 : 1
    return {
      animation: false,
      grid: { left: 48, right: 18, top: 22, bottom: 46 },
      tooltip: {
        trigger: 'axis', confine: true, axisPointer: { type: 'shadow' },
        backgroundColor: ct.tooltipBg, borderColor: ct.tooltipBorder,
        textStyle: { color: ct.tooltipText, fontSize: 11 },
        valueFormatter: value => valueKind === 'percent' ? `${Number(value).toFixed(2)}%` : Number(value).toFixed(2),
      },
      xAxis: {
        type: 'category', data: display.map(point => point.label),
        axisLine: { lineStyle: { color: ct.border } }, axisTick: { show: false },
        axisLabel: { color: ct.text, fontSize: 9, interval: 0, width: 72, overflow: 'truncate' },
      },
      yAxis: {
        type: 'value', axisLabel: { color: ct.text, fontSize: 9, formatter: value => valueKind === 'percent' ? `${Number(value).toFixed(0)}%` : Number(value).toFixed(1) },
        splitLine: { lineStyle: { color: ct.grid } },
      },
      series: [{
        name: label, type: 'bar', barMaxWidth: 34,
        data: display.map(point => ({
          value: point.value! * multiplier,
          itemStyle: { color: point.value! >= 0 ? '#ef4444' : '#22c55e' },
        })),
      }],
    }
  }, [ct, display, label, valueKind])
  return <ChartFrame option={option} empty={`没有可展示的${label}`} label={`${label}柱状图`} height={height} />
}
