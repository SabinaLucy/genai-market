export function fmt(n, d = 2) {
  if (n == null) return '—'
  return Number(n).toFixed(d)
}

export function pct(n, d = 1) {
  if (n == null) return '—'
  return `${(Number(n) * 100).toFixed(d)}%`
}

export function timeAgo(date) {
  if (!date) return null
  const diff = Math.floor((Date.now() - date.getTime()) / 1000)
  if (diff < 10)   return 'just now'
  if (diff < 60)   return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

export const REGIME_COLOR = {
  LOW:      '#22c55e',
  ELEVATED: '#f59e0b',
  CRISIS:   '#ef4444',
}

export const REGIME_TEXT = {
  LOW:      '#4ade80',
  ELEVATED: '#fbbf24',
  CRISIS:   '#f87171',
}
