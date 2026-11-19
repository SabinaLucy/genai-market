import { useState, useEffect } from 'react'
import RegimeBadge from './RegimeBadge'
import { useTheme } from '../ThemeContext'
import { fetchHeadlines } from '../api'
import { fmt } from '../utils'

const TILES = [
  { key: 'SUMMARY',            label: 'Market Summary',    icon: '✧', dark: { color: '#a78bfa', bg: 'rgba(139,92,246,0.12)', border: 'rgba(139,92,246,0.3)', top: 'linear-gradient(90deg,#8b5cf6,#a78bfa)' }, light: { color: '#6d28d9', bg: 'rgba(109,40,217,0.08)', border: 'rgba(109,40,217,0.25)', top: 'linear-gradient(90deg,#8b5cf6,#a78bfa)' } },
  { key: 'KEY DRIVERS',        label: 'Key Market Drivers', icon: '↗', dark: { color: '#fbbf24', bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.3)', top: 'linear-gradient(90deg,#f59e0b,#fbbf24)' }, light: { color: '#b45309', bg: 'rgba(180,83,9,0.08)',    border: 'rgba(180,83,9,0.25)',    top: 'linear-gradient(90deg,#f59e0b,#fbbf24)' } },
  { key: 'HISTORICAL CONTEXT', label: 'Historical Context',icon: '◎', dark: { color: '#4ade80', bg: 'rgba(34,197,94,0.10)',  border: 'rgba(34,197,94,0.3)',  top: 'linear-gradient(90deg,#22c55e,#4ade80)' }, light: { color: '#15803d', bg: 'rgba(21,128,61,0.08)',  border: 'rgba(21,128,61,0.25)',   top: 'linear-gradient(90deg,#22c55e,#4ade80)' } },
  { key: 'SIGNAL',             label: 'Market Signal',     icon: '◎', dark: { color: '#c4b5fd', bg: 'rgba(124,58,237,0.10)', border: 'rgba(124,58,237,0.3)', top: 'linear-gradient(90deg,#7c3aed,#c4b5fd)' }, light: { color: '#5b21b6', bg: 'rgba(91,33,182,0.08)',  border: 'rgba(91,33,182,0.25)',   top: 'linear-gradient(90deg,#7c3aed,#c4b5fd)' } },
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

export default function BulletinCard({ data, loading, error, onRetry, rateLimited = false, retryAfter = 0, cachedAt = null }) {
  const [copied,    setCopied]    = useState(false)
  const [headlines, setHeadlines] = useState([])
  const [headlinesLoading, setHeadlinesLoading] = useState(true)
  const isDark   = useTheme()
  const isLight  = !isDark
  const isMobile = window.innerWidth <= 768
  const sections = parseBulletin(data?.bulletin)

  useEffect(() => {
    fetchHeadlines()
      .then(res => setHeadlines(res.data?.headlines || []))
      .catch(() => setHeadlines([]))
      .finally(() => setHeadlinesLoading(false))
  }, [])

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

  if (error && !data) return (
    <div className="card" style={{ padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, paddingTop: 48, paddingBottom: 48 }}>
      <span style={{ fontSize: 24 }}>⚠</span>
      <p style={{ fontSize: 13, color: 'var(--text-2)' }}>{rateLimited ? 'AI report limit reached. Please try again after the cooldown. No previous report is cached in this browser yet.' : error}</p>
      {onRetry && !rateLimited && <button onClick={onRetry} style={{ fontSize: 12, padding: '6px 14px', borderRadius: 8, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'transparent', cursor: 'pointer' }}>Retry</button>}
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

      {(rateLimited || error) && <div className="vi-cache-notice" role="status">{rateLimited ? 'AI report limit reached. Displaying the latest saved report' : 'Unable to refresh the AI report. Displaying the latest saved report'}{cachedAt ? ` · Saved ${new Date(cachedAt).toLocaleString()}` : ''}. {rateLimited ? 'Please try again after the request limit resets.' : 'You can retry when the service is available.'}</div>}
      {/* Purple glass market stress report, live figures and existing actions */}
      <div className="vi-report-top">
        <div className="vi-report-heading">
          <span className="vi-report-icon" aria-hidden="true">✧</span>
          <div className="vi-report-heading-text"><span className="vi-report-eyebrow">VOLARIX INTELLIGENCE · {data.date}</span><h2>Market Stress Report</h2><p>AI-powered market assessment</p></div>
          <div className="vi-report-actions"><RegimeBadge regime={data.regime} regimeLabel={data.regime_label} size="md"/><button onClick={handleCopy} className="vi-copy-btn">{copied ? '✓ Copied' : 'Copy'}</button></div>
        </div>
        <div className="vi-report-metrics">
          {[
            {label:'Current VIX',val:fmt(data.current_vix),note:'Latest volatility reading',icon:'◈',kind:'vix'},
            {label:'5-Day Forecast',val:fmt(data.predicted_vix),note:'Expected volatility',icon:'↘',kind:'forecast'},
            {label:'Market Regime',val:data.regime_label || data.regime || '—',note:'Current stress classification',icon:'◇',kind:'regime'},
            {label:'Confidence',val:`${data.confidence_pct ?? 90}%`,note:'Forecast interval level',icon:'◎',kind:'confidence'},
          ].map(m=><div key={m.kind} className={`vi-report-metric vi-metric-${m.kind}`}><div className="vi-metric-label"><span>{m.label}</span><span aria-hidden="true">{m.icon}</span></div><strong>{m.val}</strong><span className="vi-metric-note">{m.note}</span></div>)}
        </div>
        <p className="vi-report-footnote">Forecasts reflect model estimates and may change as new market data becomes available.</p>
      </div>

      {/* Decorative intelligence cards — original AI bulletin sections retained */}
      <div className="vi-card-grid">
        {TILES.map((tile, i) => {
          const c = isLight ? tile.light : tile.dark
          const subtitles = ['THE BIG PICTURE', "WHAT'S MOVING MARKETS", 'LOOKING BACK', 'AI INTERPRETATION']
          const footers = ['Market overview', 'Influencing factors', 'Past comparisons', 'Model insights']
          const lines = [
            'M0 19 Q12 2 24 14 T48 9 T75 3',
            'M0 22 L12 17 L23 20 L37 7 L51 12 L64 3 L75 1',
            'M0 18 Q10 2 20 18 T40 18 T60 18 T75 8',
            'M0 22 Q14 20 24 13 T47 16 T75 2',
          ]
          return <article className={`vi-intelligence-card vi-card-${i}`} key={tile.key} style={{ '--vi-accent': c.color, '--vi-glow': c.bg, '--vi-border': c.border }}>
            <div className="vi-card-stripe" style={{background:c.top}} />
            <div className="vi-card-head">
              <span className="vi-card-icon" aria-hidden="true">{tile.icon}</span>
              <div className="vi-card-ornament" aria-hidden="true">
                <span>0{i+1}</span>
                <svg viewBox="0 0 75 26" width="75" height="26"><path d={lines[i]} fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round"/><circle cx="73" cy={i===0?3:i===1?1:i===2?8:2} r="3" fill="currentColor"/></svg>
              </div>
            </div>
            <div className="vi-card-eyebrow">{subtitles[i]}</div>
            <h3 className="vi-card-title">{tile.label}</h3>
            <p className="vi-card-body">{styledText(sections[tile.key] || 'Analysis unavailable for this section.', isLight)}</p>
            <div className="vi-card-footer"><span>{footers[i]}</span><span className="vi-card-arrow" aria-hidden="true">↗</span></div>
          </article>
        })}
      </div>

      {sections.UNCERTAINTY && <section className="vi-lower-panel vi-uncertainty">
        <div className="vi-section-heading"><span className="vi-section-icon">◇</span><div><h2>Forecast Uncertainty</h2><p>Understanding the possible range of future volatility</p></div></div>
        <p className="vi-lower-body">{styledText(sections.UNCERTAINTY, isLight)}</p>
      </section>}

      {(headlinesLoading || headlines.length > 0) && <section className="vi-news-section">
        <div className="vi-section-heading"><span className="vi-section-icon">▤</span><div><h2>Market News &amp; Headlines</h2><p>Financial news and market developments</p></div>{!headlinesLoading&&<span className="vi-news-count">{headlines.length} stories</span>}</div>
        {headlinesLoading ? <div className="vi-news-grid">{[1,2,3].map(i=><div className="vi-news-card" key={i}><div className="skeleton" style={{height:130}}/><div className="skeleton" style={{height:16,margin:16}}/></div>)}</div> : <div className="vi-news-grid">{headlines.map((h,i)=><a className="vi-news-card" key={h.url||i} href={h.url} target="_blank" rel="noopener noreferrer">
          {h.urlToImage||h.image_url||h.image ? <img className="vi-news-image" src={h.urlToImage||h.image_url||h.image} alt="" loading="lazy" onError={e=>{e.currentTarget.style.display='none'}}/> : <div className="vi-news-placeholder" aria-hidden="true">✧</div>}
          <div className="vi-news-content"><span className="vi-news-source">{h.source||'Financial news'}</span><h3>{h.title}</h3><div className="vi-news-meta"><span>{h.publishedAt&&!Number.isNaN(new Date(h.publishedAt).getTime())?new Date(h.publishedAt).toLocaleString('en-US',{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}):'Latest report'}</span><span aria-hidden="true">↗</span></div></div>
        </a>)}</div>}
      </section>}

      {sections.DISCLAIMER && <section className="vi-lower-panel vi-disclaimer"><div className="vi-section-heading"><span className="vi-section-icon">ⓘ</span><div><h2>About This Analysis</h2></div></div><p className="vi-lower-body">{styledText(sections.DISCLAIMER,isLight)}</p></section>}

  <div style={{ padding: '10px 28px', fontSize: 10, color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.05)'}`, background: isLight ? '#f9f9ff' : 'rgba(0,0,0,0.2)' }}>
        Generated {data.generated_at ? new Date(data.generated_at).toLocaleTimeString() : '—'} · Automated quantitative system
      </div>
    </div>
  )
}