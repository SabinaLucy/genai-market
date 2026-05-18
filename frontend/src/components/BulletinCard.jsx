import { useState } from 'react'
import RegimeBadge from './RegimeBadge'
import { useTheme } from '../ThemeContext'
import { fmt } from '../utils'

const TILES = [
  { key: 'SUMMARY',            label: 'Summary',           icon: '◈', dark: { color: '#a78bfa', bg: 'rgba(139,92,246,0.12)', border: 'rgba(139,92,246,0.3)', top: 'linear-gradient(90deg,#8b5cf6,#a78bfa)' }, light: { color: '#6d28d9', bg: 'rgba(109,40,217,0.08)', border: 'rgba(109,40,217,0.25)', top: 'linear-gradient(90deg,#8b5cf6,#a78bfa)' } },
  { key: 'KEY DRIVERS',        label: 'Key Drivers',       icon: '↑', dark: { color: '#fbbf24', bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.3)', top: 'linear-gradient(90deg,#f59e0b,#fbbf24)' }, light: { color: '#b45309', bg: 'rgba(180,83,9,0.08)',    border: 'rgba(180,83,9,0.25)',    top: 'linear-gradient(90deg,#f59e0b,#fbbf24)' } },
  { key: 'HISTORICAL CONTEXT', label: 'Historical Context',icon: '◎', dark: { color: '#4ade80', bg: 'rgba(34,197,94,0.10)',  border: 'rgba(34,197,94,0.3)',  top: 'linear-gradient(90deg,#22c55e,#4ade80)' }, light: { color: '#15803d', bg: 'rgba(21,128,61,0.08)',  border: 'rgba(21,128,61,0.25)',   top: 'linear-gradient(90deg,#22c55e,#4ade80)' } },
  { key: 'SIGNAL',             label: 'Signal',             icon: '▲', dark: { color: '#c4b5fd', bg: 'rgba(124,58,237,0.10)', border: 'rgba(124,58,237,0.3)', top: 'linear-gradient(90deg,#7c3aed,#c4b5fd)' }, light: { color: '#5b21b6', bg: 'rgba(91,33,182,0.08)',  border: 'rgba(91,33,182,0.25)',   top: 'linear-gradient(90deg,#7c3aed,#c4b5fd)' } },
]

const FULL_ROWS = [
  { key: 'UNCERTAINTY', label: 'Uncertainty', icon: '⚡', dark: { color: '#f87171', bg: 'rgba(239,68,68,0.08)',    border: 'rgba(239,68,68,0.25)',   rowBg: 'linear-gradient(90deg,rgba(239,68,68,0.06),rgba(239,68,68,0.02))'  }, light: { color: '#dc2626', bg: 'rgba(239,68,68,0.10)',   border: 'rgba(239,68,68,0.3)',   rowBg: 'rgba(239,68,68,0.04)' } },
  { key: 'DISCLAIMER',  label: 'Disclaimer',  icon: '◇', dark: { color: '#94a3b8', bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.2)',  rowBg: 'transparent' }, light: { color: '#6b7280', bg: 'rgba(107,114,128,0.08)', border: 'rgba(107,114,128,0.2)', rowBg: 'transparent' } },
]

const ALL_KEYS = [...TILES, ...FULL_ROWS].map(t => t.key)

function styledText(text, isLight) {
  if (!text) return text
  const processed = text
    .replace(/(\d+\.?\d*)\s*%/g, '__PCT__$1__PCT__')
    .replace(/(ELEVATED|CRISIS|STABLE|WATCH|ALERT)/g, '__REG__$1__REG__')
    .replace(/\b(\d+\.?\d*)\b/g, '__NUM__$1__NUM__')

  return processed.split(/(__PCT__.*?__PCT__|__REG__.*?__REG__|__NUM__.*?__NUM__)/g).map((part, i) => {
    if (part.startsWith('__PCT__')) {
      const val = part.replace(/__PCT__/g, '')
      return <span key={i} style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: 14, background: isLight ? 'rgba(91,33,182,0.1)' : 'rgba(99,102,241,0.18)', color: isLight ? '#5b21b6' : '#a5b4fc', padding: '1px 6px', borderRadius: 4, border: `1px solid ${isLight ? 'rgba(91,33,182,0.25)' : 'rgba(99,102,241,0.35)'}` }}>{val}%</span>
    }
    if (part.startsWith('__REG__')) {
      const val = part.replace(/__REG__/g, '')
      const colors = {
        ELEVATED: isLight ? { bg: 'rgba(180,83,9,0.12)',   color: '#b45309', border: 'rgba(180,83,9,0.3)'   } : { bg: 'rgba(245,158,11,0.15)', color: '#fbbf24', border: 'rgba(245,158,11,0.35)' },
        CRISIS:   isLight ? { bg: 'rgba(220,38,38,0.12)',  color: '#dc2626', border: 'rgba(220,38,38,0.3)'  } : { bg: 'rgba(239,68,68,0.15)',  color: '#f87171', border: 'rgba(239,68,68,0.35)'  },
        STABLE:   isLight ? { bg: 'rgba(21,128,61,0.12)',  color: '#15803d', border: 'rgba(21,128,61,0.3)'  } : { bg: 'rgba(34,197,94,0.15)',  color: '#4ade80', border: 'rgba(34,197,94,0.35)'  },
        WATCH:    isLight ? { bg: 'rgba(180,83,9,0.12)',   color: '#b45309', border: 'rgba(180,83,9,0.3)'   } : { bg: 'rgba(245,158,11,0.15)', color: '#fbbf24', border: 'rgba(245,158,11,0.35)' },
        ALERT:    isLight ? { bg: 'rgba(220,38,38,0.12)',  color: '#dc2626', border: 'rgba(220,38,38,0.3)'  } : { bg: 'rgba(239,68,68,0.15)',  color: '#f87171', border: 'rgba(239,68,68,0.35)'  },
      }
      const c = colors[val] || colors.ELEVATED
      return <span key={i} style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: 12, background: c.bg, color: c.color, padding: '1px 7px', borderRadius: 4, border: `1px solid ${c.border}`, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{val}</span>
    }
    if (part.startsWith('__NUM__')) {
      const val = part.replace(/__NUM__/g, '')
      return <span key={i} style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: 14, background: isLight ? 'rgba(180,83,9,0.1)' : 'rgba(245,158,11,0.14)', color: isLight ? '#b45309' : '#fbbf24', padding: '1px 6px', borderRadius: 4, border: `1px solid ${isLight ? 'rgba(180,83,9,0.25)' : 'rgba(245,158,11,0.3)'}` }}>{val}</span>
    }
    return part
  })
}

function parseBulletin(raw) {
  if (!raw) return {}
  const text    = raw.replace(/[-─]{4,}/g, '').replace(/\[CACHED\]/g, '').trim()
  const pattern = new RegExp(`(${ALL_KEYS.join('|')})`, 'g')
  const parts   = text.split(pattern).filter(Boolean)
  const map     = {}
  let i = 0
  if (parts[0] && !ALL_KEYS.includes(parts[0].trim())) {
    map['HEADER'] = parts[0].replace(/WEEKLY MARKET STRESS BULLETIN[^|]*\|/i, '').trim()
    i = 1
  }
  while (i < parts.length) {
    const key = parts[i]?.trim(); const body = parts[i + 1]?.trim()
    if (key && body && ALL_KEYS.includes(key)) { map[key] = body; i += 2 } else { i++ }
  }
  return map
}

export default function BulletinCard({ data, loading, error, onRetry }) {
  const [copied, setCopied] = useState(false)
  const isDark   = useTheme()
  const isLight  = !isDark
  const isMobile = window.innerWidth <= 768
  const sections = parseBulletin(data?.bulletin)

  const handleCopy = () => {
    navigator.clipboard.writeText(data?.bulletin || '').then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 2000)
    })
  }

  if (loading) return (
    <div className="card" style={{ padding: 24 }}>
      {[...Array(4)].map((_, i) => (
        <div key={i} style={{ marginBottom: 16 }}>
          <div className="skeleton" style={{ height: 12, width: 80, marginBottom: 10 }} />
          <div className="skeleton" style={{ height: 14, marginBottom: 4 }} />
          <div className="skeleton" style={{ height: 14, width: '85%' }} />
        </div>
      ))}
    </div>
  )

  if (error) return (
    <div className="card" style={{ padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, paddingTop: 48, paddingBottom: 48 }}>
      <span style={{ fontSize: 24 }}>⚠</span>
      <p style={{ fontSize: 13, color: 'var(--text-2)' }}>{error}</p>
      {onRetry && <button onClick={onRetry} style={{ fontSize: 12, padding: '6px 14px', borderRadius: 8, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'transparent', cursor: 'pointer' }}>Retry</button>}
    </div>
  )

  if (!data) return null

  const borderColor = isLight ? 'rgba(0,0,0,0.09)'  : 'rgba(255,255,255,0.08)'
  const gridBg      = isLight ? '#f4f4fb'            : '#09090f'
  const hdrBg       = isLight ? 'linear-gradient(90deg,rgba(245,158,11,0.14),rgba(245,158,11,0.05))' : 'linear-gradient(90deg,rgba(245,158,11,0.12),rgba(245,158,11,0.04))'
  const hdrBorder   = isLight ? 'rgba(245,158,11,0.3)' : 'rgba(245,158,11,0.2)'
  const statBg      = isLight ? '#ffffff'            : '#0d0d1c'
  const statRowBg   = isLight ? '#f0f0f8'            : '#1a1a2e'
  const statBorder  = isLight ? 'rgba(0,0,0,0.08)'  : '#1e1e2e'

  return (
    <div style={{ borderRadius: 20, overflow: 'hidden', border: `1px solid ${borderColor}`, boxShadow: isLight ? '0 8px 40px rgba(0,0,0,0.10)' : '0 8px 40px rgba(0,0,0,0.5)' }}>

      {/* Header */}
      <div style={{ padding: isMobile ? '16px 18px' : '22px 28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: hdrBg, borderBottom: `2px solid ${hdrBorder}`, flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ fontSize: 9, color: isLight ? '#b45309' : '#fbbf24', fontFamily: 'JetBrains Mono, monospace', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>
            Volarix Intelligence · {data.date}
          </div>
          <div style={{ fontSize: isMobile ? 15 : 18, fontWeight: 700, color: 'var(--text-1)', letterSpacing: '-0.02em' }}>Market Stress Report</div>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <RegimeBadge regime={data.regime} regimeLabel={data.regime_label} size="md" />
          <button onClick={handleCopy} style={{ fontSize: 11, padding: '5px 12px', borderRadius: 8, border: '1px solid var(--border-light)', color: copied ? '#4ade80' : 'var(--text-1)', background: isLight ? '#fff' : 'rgba(255,255,255,0.05)', cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace', transition: 'color 0.2s' }}>
            {copied ? '✓' : 'Copy'}
          </button>
        </div>
      </div>

      {/* 4-stat strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', background: statRowBg }}>
        {[
          { label: 'Current VIX', val: fmt(data.current_vix),          color: isLight ? '#b45309' : '#fbbf24' },
          { label: 'Forecast 5d', val: `${fmt(data.predicted_vix)} ↓`, color: isLight ? '#15803d' : '#4ade80' },
          { label: 'Regime',      val: data.regime || '—',              color: data.regime === 'LOW' ? (isLight ? '#15803d' : '#4ade80') : data.regime === 'CRISIS' ? (isLight ? '#dc2626' : '#f87171') : (isLight ? '#b45309' : '#fbbf24') },
          { label: 'Confidence',  val: `${data.confidence_pct || 90}%`, color: isLight ? '#5b21b6' : '#c4b5fd' },
        ].map(({ label, val, color }, i, arr) => (
          <div key={label} style={{ padding: isMobile ? '10px 12px' : '13px 20px', borderRight: i < arr.length - 1 ? `1px solid ${statBorder}` : 'none', background: statBg }}>
            <div style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: isMobile ? 16 : 22, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color }}>{val}</div>
          </div>
        ))}
      </div>

      {/* 2x2 tile grid — single column on mobile */}
      <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr', gap: isMobile ? 10 : 16, padding: isMobile ? '12px' : '20px', background: gridBg }}>
        {TILES.map(tile => {
          const c = isLight ? tile.light : tile.dark
          return (
            <div key={tile.key} style={{
              borderRadius: 16,
              padding: isMobile ? '16px 18px' : '28px 28px',
              position: 'relative',
              overflow: 'hidden',
              border: `1px solid ${c.border}`,
              background: isLight ? '#ffffff' : 'linear-gradient(145deg,#13131f,#0e0e1a)',
              boxShadow: isLight ? '0 4px 20px rgba(0,0,0,0.07), 0 1px 0 rgba(255,255,255,0.9) inset' : '0 4px 20px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.06) inset',
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: c.top, borderRadius: '16px 16px 0 0' }} />
              <div style={{ position: 'absolute', top: -40, right: -40, width: 100, height: 100, borderRadius: '50%', background: c.color, opacity: isLight ? 0.05 : 0.06 }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <div style={{ width: 30, height: 30, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, flexShrink: 0, background: c.bg, border: `1px solid ${c.border}`, color: c.color, boxShadow: isLight ? 'none' : `0 0 10px ${c.border}` }}>
                  {tile.icon}
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', color: c.color, textShadow: isLight ? 'none' : `0 0 20px ${c.border}` }}>
                  {tile.label}
                </div>
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.75 }}>
                {styledText(sections[tile.key] || 'Loading analysis…', isLight)}
              </p>
            </div>
          )
        })}
      </div>

      {/* Full-width rows */}
      {FULL_ROWS.map(tile => {
        if (!sections[tile.key]) return null
        const c = isLight ? tile.light : tile.dark
        return (
          <div key={tile.key} style={{ padding: isMobile ? '14px 18px' : '18px 28px', borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.06)'}`, display: 'flex', gap: 14, background: c.rowBg }}>
            <div style={{ width: 30, height: 30, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, flexShrink: 0, background: c.bg, border: `1px solid ${c.border}`, color: c.color }}>
              {tile.icon}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', marginBottom: 8, color: c.color }}>
                {tile.label}
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.75 }}>
                {styledText(sections[tile.key], isLight)}
              </p>
            </div>
          </div>
        )
      })}

      <div style={{ padding: '10px 28px', fontSize: 10, color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.05)'}`, background: isLight ? '#f9f9ff' : 'rgba(0,0,0,0.2)' }}>
        Generated {data.generated_at ? new Date(data.generated_at).toLocaleTimeString() : '—'} · Automated quantitative system
      </div>
    </div>
  )
}
