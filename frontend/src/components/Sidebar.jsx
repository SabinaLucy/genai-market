import VolarixLogo from './VolarixLogo'
import RegimeBadge  from './RegimeBadge'
import ThemeToggle  from './ThemeToggle'

const NAV = [
  { id: 'overview',  label: 'Overview',              icon: '◈' },
  { id: 'regime',    label: 'Regime Analysis',        icon: '◉' },
  { id: 'bulletin',  label: 'Intelligence Bulletin',  icon: '▤' },
  { id: 'ask',       label: 'Ask Volarix',            icon: '⌘' },
  { id: 'backtest',  label: 'Strategy Backtest',      icon: '◎' },
  { id: 'about',     label: 'About',                  icon: '◇' },
]

export default function Sidebar({ active, onNav, latestData, isDark, onThemeToggle }) {
  const regime      = latestData?.regime
  const regimeColor = latestData?.regime_color

  return (
    <div className="sidebar">
      <div style={{ padding: '22px 18px 16px', borderBottom: '1px solid var(--border)' }}>
        <VolarixLogo regimeColor={regimeColor} />
        {regime && (
          <div style={{ marginTop: 14 }}>
            <RegimeBadge regime={regime} regimeLabel={latestData?.regime_label} size="sm" />
          </div>
        )}
      </div>

      <nav style={{ flex: 1, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ padding: '8px 8px 4px', fontSize: 9, fontWeight: 600, color: 'var(--text-3)', letterSpacing: '0.14em', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace' }}>
          Menu
        </div>
        {NAV.map(item => (
          <button key={item.id} onClick={() => onNav(item.id)} className={`nav-item ${active === item.id ? 'active' : ''}`}>
            <span style={{ width: 18, height: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, opacity: active === item.id ? 1 : 0.6, fontSize: 13 }}>
              {item.icon}
            </span>
            {item.label}
            {active === item.id && (
              <span style={{ marginLeft: 'auto', width: 5, height: 5, borderRadius: '50%', background: regimeColor || '#a78bfa', flexShrink: 0 }} />
            )}
          </button>
        ))}
      </nav>

      <div style={{ padding: '14px 18px', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'JetBrains Mono, monospace' }}>
            {isDark ? 'Dark mode' : 'Light mode'}
          </span>
          <ThemeToggle isDark={isDark} onToggle={onThemeToggle} />
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {[
            { label: 'GitHub',   href: 'https://github.com/SabinaLucy/genai-market',              icon: '⌥' },
            { label: 'API Docs', href: 'https://sabinalucy-volarix.hf.space/docs',                icon: '⎋' },
            { label: 'HF Space', href: 'https://huggingface.co/spaces/SabinaLucy/volarix',       icon: '◈' },
          ].map(l => (
            <a key={l.label} href={l.href} target="_blank" rel="noopener noreferrer"
               style={{ flex: 1, textAlign: 'center', fontSize: 10, fontWeight: 600, color: 'var(--text-2)', fontFamily: 'JetBrains Mono, monospace', textDecoration: 'none', padding: '6px 4px', borderRadius: 8, border: '1px solid transparent', letterSpacing: '0.04em', textTransform: 'uppercase', transition: 'all 0.15s' }}
               onMouseEnter={e => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-hover)'; e.currentTarget.style.borderColor = 'var(--border)' }}
               onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent' }}
            >
              {l.label}
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}
