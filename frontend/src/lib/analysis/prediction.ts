import type { ScenarioRecord } from '../../types/ipc'
import { getDatePlayed, getScenarioName } from '../utils'

export type HighscorePrediction = {
  // Legacy time-based ETA derived from recommended cadence and expected runs
  etaTs: number | null
  etaHuman: string
  // New runs-based forecast
  runsExpected: number | null
  runsLo: number | null
  runsHi: number | null
  optPauseHours: number | null
  // Diagnostics
  confidence: 'low' | 'med' | 'high'
  sample: number
  best: number
  lastScore: number
  lastPlayedDays: number
  slopePerDay: number
  slopePerRun: number
  reason?: string
}

const DAY = 24 * 3600 * 1000

function pad(n: number, w = 2) { return n.toString().padStart(w, '0') }

function humanizeETA(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return 'soon'
  const totalMin = Math.round(ms / (60 * 1000))
  const days = Math.floor(totalMin / (60 * 24))
  const hours = Math.floor((totalMin % (60 * 24)) / 60)
  const mins = totalMin % 60
  if (days <= 0) {
    if (hours <= 0) return `${mins}m`
    // under a day, show Hh Mm when there are minutes
    return mins ? `${hours}h ${mins}m` : `${hours}h`
  }
  if (days < 7) return mins ? `${days}d ${hours}h ${mins}m` : (hours ? `${days}d ${hours}h` : `${days}d`)
  const weeks = Math.floor(days / 7)
  const remDays = days % 7
  return remDays ? `${weeks}w ${remDays}d` : `${weeks}w`
}

// Try to get a precise timestamp from fileName, else fall back to Date Played + Challenge Start
export function parseRecordTimestamp(it: ScenarioRecord): number {
  const fn = String(it.fileName || '')
  const m = fn.match(/(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})/)
  if (m) {
    const y = parseInt(m[1], 10)
    const mo = parseInt(m[2], 10) - 1
    const d = parseInt(m[3], 10)
    const hh = parseInt(m[4], 10)
    const mm = parseInt(m[5], 10)
    const ss = parseInt(m[6], 10)
    return new Date(y, mo, d, hh, mm, ss).getTime()
  }
  const dateStr = getDatePlayed(it.stats)
  const timeStr = String((it.stats as any)?.['Challenge Start'] ?? '')
  if (dateStr) {
    // Try to parse common formats; build ISO-ish when possible
    const d1 = dateStr.match(/(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})/)
    let ts = NaN
    if (d1) {
      const y = parseInt(d1[1], 10)
      const mo = parseInt(d1[2], 10) - 1
      const d = parseInt(d1[3], 10)
      if (timeStr) {
        const t = timeStr.match(/(\d{1,2}):(\d{2}):(\d{2})/)
        if (t) {
          ts = new Date(y, mo, d, parseInt(t[1], 10), parseInt(t[2], 10), parseInt(t[3], 10)).getTime()
        }
      }
      if (!Number.isFinite(ts)) ts = new Date(y, mo, d).getTime()
      return ts
    }
    // MM/DD/YYYY or DD/MM/YYYY – ambiguous; use Date.parse and hope environment locale handles it
    const parsed = Date.parse(dateStr)
    if (Number.isFinite(parsed)) return parsed
  }
  return Date.now()
}

function weightedLinReg(xs: number[], ys: number[], ws: number[]): { a: number; b: number; r2: number } {
  const n = Math.min(xs.length, ys.length, ws.length)
  if (n < 2) return { a: 0, b: 0, r2: 0 }
  let Sw = 0, Swx = 0, Swy = 0, Swxx = 0, Swxy = 0
  for (let i = 0; i < n; i++) {
    const w = ws[i]
    const x = xs[i]
    const y = ys[i]
    Sw += w
    Swx += w * x
    Swy += w * y
    Swxx += w * x * x
    Swxy += w * x * y
  }
  const denom = Sw * Swxx - Swx * Swx
  const b = denom !== 0 ? (Sw * Swxy - Swx * Swy) / denom : 0
  const a = Sw !== 0 ? (Swy - b * Swx) / Sw : 0
  // weighted R^2
  const yhat = xs.map(x => a + b * x)
  const ymean = Sw !== 0 ? (Swy / Sw) : 0
  let ssRes = 0, ssTot = 0
  for (let i = 0; i < n; i++) {
    const w = ws[i]
    const e = ys[i] - yhat[i]
    ssRes += w * e * e
    const d = ys[i] - ymean
    ssTot += w * d * d
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0
  return { a, b, r2 }
}

export function collectScenarioHistory(items: ScenarioRecord[], name: string): Array<{ t: number; score: number }> {
  const out: Array<{ t: number; score: number }> = []
  for (const it of items) {
    if (getScenarioName(it) !== name) continue
    const score = Number(it.stats['Score'] ?? 0)
    if (!Number.isFinite(score)) continue
    out.push({ t: parseRecordTimestamp(it), score })
  }
  // oldest -> newest
  out.sort((a, b) => a.t - b.t)
  return out
}

function clamp(n: number, a: number, b: number) { return Math.max(a, Math.min(b, n)) }

function linRegWeighted(xs: number[], ys: number[], ws: number[]): { a: number; b: number; r2: number } {
  return weightedLinReg(xs, ys, ws)
}

function quantile(arr: number[], q: number): number {
  if (arr.length === 0) return 0
  const a = [...arr].sort((x, y) => x - y)
  const pos = clamp(q, 0, 1) * (a.length - 1)
  const lo = Math.floor(pos), hi = Math.ceil(pos), f = pos - lo
  if (hi === lo) return a[lo]
  return a[lo] * (1 - f) + a[hi] * f
}

// Fit deltaScore per run as a function of pause between runs: delta ~= a * (1 - exp(-dt/tau)) + b
function fitDeltaVsPause(dtDays: number[], deltas: number[], pairWeights: number[]) {
  const taus = [] as number[]
  for (let x = -1.2; x <= 1.2; x += 0.1) { // log10 grid ~ 0.06 to ~15 days
    const tau = Math.pow(10, x)
    taus.push(tau)
  }
  let best = { tau: 1, a: 0, b: 0, r2: 0 }
  for (const tau of taus) {
    const x = dtDays.map(d => 1 - Math.exp(-Math.max(0, d) / tau))
    const { a, b, r2 } = linRegWeighted(x, deltas, pairWeights)
    if (r2 > best.r2) best = { tau, a, b, r2 }
  }
  return best
}

function expectedDeltaAtPause(a: number, b: number, tau: number, dtDays: number): number {
  const x = 1 - Math.exp(-Math.max(0, dtDays) / Math.max(1e-6, tau))
  return a * x + b
}

function findOptimalPauseAndRuns(deficit: number, fit: { a: number; b: number; tau: number },
  dtMinDays = 0.08, dtMaxDays = 7, dtStepDays = 0.04) {
  // Minimize total time = runs * dt, with runs = deficit / expectedDelta(dt)
  let best = { dtDays: 1, runs: Number.POSITIVE_INFINITY, delta: 0 }
  for (let d = dtMinDays; d <= dtMaxDays + 1e-9; d += dtStepDays) {
    const delta = Math.max(0, expectedDeltaAtPause(fit.a, fit.b, fit.tau, d))
    if (delta <= 1e-6) continue
    const runs = deficit / delta
    const time = runs * d
    if (!Number.isFinite(time)) continue
    if (time < best.runs * best.dtDays) {
      best = { dtDays: d, runs, delta }
    }
  }
  if (!Number.isFinite(best.runs)) return null
  return best
}

export function predictNextHighscore(items: ScenarioRecord[], name: string): HighscorePrediction {
  const now = Date.now()
  const hist = collectScenarioHistory(items, name)
  const n = hist.length
  if (n < 4) {
    return { etaTs: null, etaHuman: 'unknown', runsExpected: null, runsLo: null, runsHi: null, optPauseHours: null, confidence: 'low', sample: n, best: 0, lastScore: 0, lastPlayedDays: 0, slopePerDay: 0, slopePerRun: 0, reason: 'Need at least 4 runs' }
  }
  const best = Math.max(...hist.map(h => h.score))
  const last = hist[n - 1]
  const lastPlayedDays = (now - last.t) / DAY
  // Build inputs
  const t0 = hist[0].t
  const xsDays = hist.map(h => (h.t - t0) / DAY)
  const ys = hist.map(h => h.score)
  // recency weighting half-life (days, a bit shorter than before)
  const halfLifeDays = 21
  const wsDays = hist.map(h => Math.pow(0.5, (now - h.t) / (halfLifeDays * DAY)))
  // Also slope per run with runs-referenced recency
  const idx = ys.map((_, i) => i)
  const halfLifeRuns = 12
  const wsRuns = idx.map(i => Math.pow(0.5, (n - 1 - i) / halfLifeRuns))
  // drop very old low-weight points if we have many (time-weighted)
  const MIN_W = 0.04
  const xs2: number[] = []
  const ys2: number[] = []
  const ws2: number[] = []
  for (let i = 0; i < xsDays.length; i++) {
    if (n > 40 && wsDays[i] < MIN_W) continue
    xs2.push(xsDays[i]); ys2.push(ys[i]); ws2.push(wsDays[i])
  }
  const { a, b, r2 } = weightedLinReg(xs2, ys2, ws2)
  let slopePerDay = b

  // Slope per run (runs as x)
  const { b: bRun } = weightedLinReg(idx, ys, wsRuns)
  const slopePerRun = bRun
  // idle penalty: if not played recently, decay slope towards zero
  const idleHalf = 7
  const idleDecay = Math.pow(0.5, Math.max(0, lastPlayedDays) / idleHalf)
  slopePerDay *= idleDecay

  // Prepare runs-based delta model across observed adjacent runs
  const dtDays: number[] = []
  const deltas: number[] = []
  const pairWeights: number[] = []
  for (let i = 1; i < n; i++) {
    const d = (hist[i].t - hist[i - 1].t) / DAY
    const ds = ys[i] - ys[i - 1]
    dtDays.push(Math.max(0.01, d))
    deltas.push(ds)
    // weight more recent pairs and moderate by magnitude to reduce outlier impact
    const w = Math.pow(0.5, (n - 1 - i) / halfLifeRuns)
    pairWeights.push(w)
  }
  // Robustify deltas: clip to IQR
  if (deltas.length >= 6) {
    const q1 = quantile(deltas, 0.25)
    const q3 = quantile(deltas, 0.75)
    const iqr = Math.max(1, q3 - q1)
    for (let i = 0; i < deltas.length; i++) {
      deltas[i] = clamp(deltas[i], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    }
  }

  const fit = fitDeltaVsPause(dtDays, deltas, pairWeights)
  const deficitMargin = Math.max(1, Math.round(best * 0.003))
  const target = best + deficitMargin
  const deficit = target - last.score

  // If no upward trend from time regression and delta model poor, bail
  const modelWeak = (!Number.isFinite(slopePerDay) || slopePerDay <= 0.00001 || r2 < 0.05)
  if (deficit <= 0) {
    const soonTs = now + (2 * DAY) * (1 - Math.min(1, r2)) * (lastPlayedDays > 2 ? 1.5 : 1)
    return { etaTs: soonTs, etaHuman: humanizeETA(soonTs - now), runsExpected: 1, runsLo: 1, runsHi: 2, optPauseHours: 24, confidence: r2 > 0.35 ? 'high' : (r2 > 0.15 ? 'med' : 'low'), sample: xs2.length, best, lastScore: last.score, lastPlayedDays, slopePerDay, slopePerRun }
  }

  const opt = findOptimalPauseAndRuns(deficit, fit)
  // Reasonable caps and guards
  const MAX_RUNS_REASONABLE = 500
  const epsImprovement = Math.max(1, Math.round(best * 0.0002)) // ~0.02% of best, at least 1 point

  if (!opt || modelWeak || !Number.isFinite(opt.runs) || opt.runs > MAX_RUNS_REASONABLE || opt.delta < epsImprovement) {
    // Fallback to runs from slope per run and pause from median observed gap
    const medDt = quantile(dtDays, 0.5) || 1
    const spr = Math.max(1e-6, slopePerRun)
    const runsRaw = deficit / spr
    if (!Number.isFinite(runsRaw) || runsRaw > MAX_RUNS_REASONABLE || spr < epsImprovement * 0.25) {
      // Too many runs or essentially flat improvement: report unknown like the old model
      return { etaTs: null, etaHuman: 'unknown', runsExpected: null, runsLo: null, runsHi: null, optPauseHours: null, confidence: 'low', sample: n, best, lastScore: last.score, lastPlayedDays, slopePerDay, slopePerRun, reason: 'No upward trend detected yet' }
    }
    const runs = Math.ceil(runsRaw)
    const confScore = Math.max(0, Math.min(1, 0.6 * r2 + 0.4 * Math.min(1, n / 20)))
    const widen = 0.35 + 0.35 * (1 - confScore)
    const lo = Math.max(1, Math.floor(runs * (1 - widen)))
    const hi = Math.max(lo + 1, Math.ceil(runs * (1 + widen)))
    const eta = now + Math.max(1, runs) * medDt * DAY
    const confidence: HighscorePrediction['confidence'] = confScore > 0.55 ? 'high' : (confScore > 0.28 ? 'med' : 'low')
    return { etaTs: eta, etaHuman: humanizeETA(eta - now), runsExpected: runs, runsLo: lo, runsHi: hi, optPauseHours: Math.round(medDt * 24), confidence, sample: n, best, lastScore: last.score, lastPlayedDays, slopePerDay, slopePerRun, reason: modelWeak ? 'Weak trend; using fallback' : undefined }
  }

  const runs = Math.max(1, Math.ceil(opt.runs))
  const eta = now + runs * opt.dtDays * DAY
  // confidence combines fit.r2 and sample size of pairs
  const pairSample = dtDays.length
  const sizeFactor = Math.min(1, pairSample / 24)
  const confScore = Math.max(0, Math.min(1, 0.65 * fit.r2 + 0.35 * sizeFactor))
  const confidence: HighscorePrediction['confidence'] = confScore > 0.6 ? 'high' : (confScore > 0.3 ? 'med' : 'low')
  // Range based on uncertainty
  const widen = 0.25 + 0.35 * (1 - confScore)
  const lo = Math.max(1, Math.floor(runs * (1 - widen)))
  const hi = Math.max(lo + 1, Math.ceil(runs * (1 + widen)))
  const optPauseHours = Math.max(1, Math.round(opt.dtDays * 24))
  return { etaTs: eta, etaHuman: humanizeETA(eta - now), runsExpected: runs, runsLo: lo, runsHi: hi, optPauseHours, confidence, sample: n, best, lastScore: last.score, lastPlayedDays, slopePerDay, slopePerRun }
}

export function formatEtaDate(ts: number | null): string {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
