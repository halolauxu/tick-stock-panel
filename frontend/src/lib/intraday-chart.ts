import type { MinuteKlineRow } from '@/lib/api'

export function formatMinuteTime(datetime: string): string {
  const match = datetime.match(/(\d{2}):(\d{2})/)
  if (!match) return datetime.slice(11, 16)

  // 分钟 K 的 canonical datetime 是北京时间墙钟，但旧版 TickFlow 数据由
  // Unix timestamp 直接落成了 naive UTC。兼容这两种已存在的数据：A 股盘中
  // 的 UTC 小时只会落在 01:30-07:00，北京时间数据则已经是 09:30-15:00。
  const rawHour = parseInt(match[1])
  const hour = rawHour < 8 ? rawHour + 8 : rawHour
  return `${String(hour).padStart(2, '0')}:${match[2]}`
}

export function computeIntradayAverage(data: MinuteKlineRow[]): number[] {
  const result: number[] = []
  let amount = 0
  let volume = 0
  for (const row of data) {
    amount += row.amount
    volume += row.volume * 100
    result.push(volume > 0 ? amount / volume : row.close)
  }
  return result
}

function generateFullDayTimes(): string[] {
  const times: string[] = []
  for (let hour = 9; hour <= 11; hour++) {
    const startMinute = hour === 9 ? 30 : 0
    const endMinute = hour === 11 ? 30 : 59
    for (let minute = startMinute; minute <= endMinute; minute++) {
      times.push(`${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`)
    }
  }
  for (let hour = 13; hour <= 15; hour++) {
    const endMinute = hour === 15 ? 0 : 59
    for (let minute = 0; minute <= endMinute; minute++) {
      times.push(`${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`)
    }
  }
  return times
}

export const FULL_DAY_TIMES = generateFullDayTimes()
