import RegimeBadge from './RegimeBadge'
import useIsMobile from '../useIsMobile'
import { useTheme } from '../ThemeContext'
import { timeAgo } from '../utils'

const PAGE_CONFIG = {
  overview: { color: '#a78bfa', glow: 'rgba(167,139,250,0.5)', bar: 'linear-gradient(180deg,#a78bfa,#6366f1)', title: 'linear-gradient(90deg,#22d3ee 0%,#a78bfa 40%,#22d3ee 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#a78bfa)' },
  regime:   { color: '#fbbf24', glow: 'rgba(245,158,11,0.4)',  bar: 'linear-gradient(180deg,#fbbf24,#f59e0b)', title: 'linear-gradient(90deg,#f97316 0%,#fbbf24 40%,#f97316 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#fbbf24)' },
  bulletin: { color: '#f87171', glow: 'rgba(239,68,68,0.4)',   bar: 'linear-gradient(180deg,#f87171,#ef4444)', title: 'linear-gradient(90deg,#fb923c 0%,#f87171 40%,#fb923c 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#f87171)' },
  ask:      { color: '#22d3ee', glow: 'rgba(6,182,212,0.4)',   bar: 'linear-gradient(180deg,#22d3ee,#06b6d4)', title: 'linear-gradient(90deg,#818cf8 0%,#22d3ee 40%,#818cf8 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#22d3ee)' },
  backtest: { color: '#4ade80', glow: 'rgba(34,197,94,0.4)',   bar: 'linear-gradient(180deg,#4ade80,#22c55e)', title: 'linear-gradient(90deg,#34d399 0%,#4ade80 40%,#34d399 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#4ade80)' },
  about:    { color: '#c4b5fd', glow: 'rgba(139,92,246,0.4)',  bar: 'linear-gradient(180deg,#c4b5fd,#8b5cf6)', title: 'linear-gradient(90deg,#818cf8 0%,#c4b5fd 40%,#818cf8 100%)', subtitle: 'linear-gradient(90deg,#c0c0e0,#c4b5fd)' },
}

export default function PageHeader({ title, subtitle, page, lastUpdated, regime, regimeLabel, children }) {
  const cfg      = PAGE_CONFIG[page] || PAGE_CONFIG.overview
  const isMobile = useIsMobile()
  const isDark   = useTheme()
  const isLight  = !isDark

  const titleStyle = {
    fontSize:      isMobile ? 20 : 30,
    fontWeight:    900,
    letterSpacing: isMobile ? '-0.02em' : '-0.04em',
    lineHeight:    1.2,
    paddingBottom: 4,
    display:       'block',
  }

  const subtitleStyle = {
    fontSize:      isMobile ? 10 : 12,
    color:         'var(--text-2)',
    marginTop:     isMobile ? 2 : 4,
    fontFamily:    'var(--font-mono)',
    letterSpacing: '0.02em',
    fontWeight:    500,
    display:       'block',
    overflow:      isMobile ? 'hidden' : 'visible',
    textOverflow:  isMobile ? 'ellipsis' : 'unset',
    whiteSpace:    isMobile ? 'nowrap' : 'normal',
    maxWidth:      isMobile ? 'calc(100vw - 130px)' : 'none',
  }

  return (
    <div style={{
      padding:       isMobile ? '10px 14px' : '18px 32px',
      borderBottom:  '1px solid var(--border)',
      display:       'flex',
      alignItems:    'center',
      justifyContent:'space-between',
      background:    isLight ? 'linear-gradient(180deg,#ffffff,#f8f8fd)' : 'linear-gradient(180deg,#0e0e1c,#0a0a18)',
      position:      'sticky',
      top:           0,
      zIndex:        10,
      boxShadow:     '0 1px 0 var(--border)',
      gap:           8,
    }}>

      <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? 10 : 14, minWidth: 0, flex: 1 }}>
        <div style={{
          width:       3,
          height:      isMobile ? 38 : 52,
          borderRadius:2,
          background:  cfg.bar,
          boxShadow:   isLight ? 'none' : `0 0 14px ${cfg.glow}`,
          flexShrink:  0,
        }} />
        <div style={{ minWidth: 0 }}>
          {isLight ? (
            <span style={{ ...titleStyle, backgroundImage: cfg.title, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>{title}</span>
          ) : (
            <span style={{ ...titleStyle, backgroundImage: cfg.title, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>{title}</span>
          )}
          {subtitle && (
            isLight ? (
              <span style={{ ...subtitleStyle, backgroundImage: cfg.subtitle, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', color: undefined }}>{subtitle}</span>
            ) : (
              <span style={{ ...subtitleStyle, backgroundImage: cfg.subtitle, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', color: undefined }}>{subtitle}</span>
            )
          )}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? 5 : 10, flexShrink: 0 }}>
        {lastUpdated && (
          <span style={{
            fontSize:     isMobile ? 9 : 11,
            color:        'var(--text-2)',
            fontFamily:   'var(--font-mono)',
            background:   'var(--bg-hover)',
            padding:      isMobile ? '3px 7px' : '4px 10px',
            borderRadius: 6,
            border:       '1px solid var(--border)',
            whiteSpace:   'nowrap',
          }}>
            {isMobile ? timeAgo(lastUpdated) : `Updated ${timeAgo(lastUpdated)}`}
          </span>
        )}
        {!isMobile && (
          <span style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', background: 'var(--bg-hover)', padding: '4px 10px', borderRadius: 6, border: '1px solid var(--border)', whiteSpace: 'nowrap' }}>
            🕐 {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        )}
        {regime && <RegimeBadge regime={regime} regimeLabel={regimeLabel} size="sm" />}
        {children}
      </div>
    </div>
  )
}
