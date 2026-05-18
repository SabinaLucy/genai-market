import ThemeToggle from './ThemeToggle'

const NAV = [
  { id: 'overview',  label: 'Home',     icon: '◈' },
  { id: 'regime',    label: 'Regime',   icon: '◉' },
  { id: 'bulletin',  label: 'Bulletin', icon: '▤' },
  { id: 'ask',       label: 'Ask',      icon: '⌘' },
  { id: 'backtest',  label: 'Backtest', icon: '◎' },
  { id: 'about',     label: 'About',    icon: '◇' },
]

export default function MobileNav({ active, onNav, isDark, onThemeToggle }) {
  return (
    <div className="mobile-nav" style={{ flexDirection: 'column', padding: 0 }}>

      {/* Nav icons row */}
      <div style={{ display: 'flex', width: '100%', padding: '7px 0 10px', background: 'var(--bg-sidebar)' }}>
        {NAV.map(item => (
          <button key={item.id} onClick={() => onNav(item.id)} style={{
            flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
            padding: '6px 2px', border: 'none', cursor: 'pointer', margin: '0 2px',
            background: active === item.id ? 'rgba(167,139,250,0.10)' : 'transparent',
            borderRadius: 10,
          }}>
            <span style={{ fontSize: 17, lineHeight: 1, opacity: active === item.id ? 1 : 0.35 }}>{item.icon}</span>
            <span style={{
              fontSize: 8, fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase',
              letterSpacing: '0.04em', color: active === item.id ? '#a78bfa' : 'var(--text-3)',
              fontWeight: active === item.id ? 700 : 400,
            }}>{item.label}</span>
          </button>
        ))}
      </div>

      {/* Theme toggle strip — sits BELOW nav icons */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
        padding: '5px 16px 8px', borderTop: '1px solid var(--border)',
        background: 'var(--bg-sidebar)', width: '100%',
      }}>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace' }}>
          {isDark ? '🌙 Dark' : '☀️ Light'}
        </span>
        <ThemeToggle isDark={isDark} onToggle={onThemeToggle} />
      </div>

    </div>
  )
}
