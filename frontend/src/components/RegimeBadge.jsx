const CFG = {
  LOW:      { color: '#22c55e', text: '#4ade80', bg: 'rgba(34,197,94,0.1)',   border: 'rgba(34,197,94,0.3)',   glow: 'rgba(34,197,94,0.15)'   },
  ELEVATED: { color: '#f59e0b', text: '#fbbf24', bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.3)',  glow: 'rgba(245,158,11,0.15)'  },
  CRISIS:   { color: '#ef4444', text: '#f87171', bg: 'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.3)',   glow: 'rgba(239,68,68,0.15)'   },
}

export default function RegimeBadge({ regime, regimeLabel, size = 'md' }) {
  const c = CFG[regime] || CFG.ELEVATED
  const pad = size === 'sm' ? '3px 10px' : size === 'lg' ? '6px 16px' : '4px 12px'
  const fs  = size === 'sm' ? 11 : size === 'lg' ? 13 : 12
  const dot = size === 'sm' ? 6  : 8

  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: pad, borderRadius: 999,
      background: c.bg, border: `1px solid ${c.border}`,
      color: c.text, fontSize: fs,
      fontFamily: 'JetBrains Mono, monospace', fontWeight: 600,
      letterSpacing: '0.05em', whiteSpace: 'nowrap',
      boxShadow: `0 0 10px ${c.glow}`,
    }}>
      <span style={{
        width: dot, height: dot, borderRadius: '50%',
        background: c.color, display: 'inline-block',
        position: 'relative', flexShrink: 0,
      }} className={regime === 'CRISIS' ? 'pulse-ring' : ''} />
      {regimeLabel || `[${regime}]`}
    </span>
  )
}
