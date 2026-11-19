import { useState } from 'react'
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
  const [more,setMore]=useState(false)
  return (
    <div className="mobile-nav" style={{ flexDirection: 'column', padding: 0 }}>

      {more && <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8,padding:12,background:'var(--bg-sidebar)'}}>{[['market','Market Explorer'],['research','Research Lab'],['workspace','My Workspace'],['settings','Settings']].map(([id,label])=><button key={id} onClick={()=>{onNav(id);setMore(false)}} style={{padding:12,border:'1px solid var(--border)',borderRadius:10,background:active===id?'var(--bg-hover)':'var(--bg-card)',color:'var(--text-1)'}}>{label}</button>)}</div>}
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
        <button onClick={()=>setMore(v=>!v)} aria-label="More Volarix pages" style={{flex:1,border:0,background:'transparent',color:'var(--text-2)',fontSize:11}}>☷<br/>More</button>
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
