import { useEffect, useState } from 'react'
import ThemeToggle from './ThemeToggle'

const NAV = [
  { id: 'overview', label: 'Home', icon: '◈' },
  { id: 'regime', label: 'Regime', icon: '◉' },
  { id: 'bulletin', label: 'Bulletin', icon: '▤' },
  { id: 'ask', label: 'Ask', icon: '⌘' },
]
const EXTRA = [
  ['market','Market Explorer','Historical VIX, stress calendar and events','◈'],
  ['research','Research Lab','Historical patterns and research','◇'],
  ['backtest','Backtesting','Strategy performance','◎'],
  ['workspace','My Workspace','Saved research tools','▦'],
  ['about','About Volarix','How the platform works','ⓘ'],
  ['settings','Settings','Preferences and appearance','⚙'],
]
export default function MobileNav({active,onNav,isDark,onThemeToggle}) {
  const [more,setMore]=useState(false)
  useEffect(()=>{if(!more)return;const onKey=e=>{if(e.key==='Escape')setMore(false)};document.addEventListener('keydown',onKey);return()=>document.removeEventListener('keydown',onKey)},[more])
  const go=id=>{setMore(false);onNav(id)}
  const moreActive=EXTRA.some(([id])=>id===active)
  return <div className="mobile-nav" style={{flexDirection:'column',padding:0}}>
    {more&&<><button aria-label="Close navigation menu" onClick={()=>setMore(false)} style={{position:'fixed',inset:0,bottom:0,background:'rgba(3,2,13,.68)',border:0,zIndex:-1}}/><div role="dialog" aria-label="Explore Volarix pages" className="vol-mobile-more">
      <div style={{width:38,height:4,borderRadius:10,background:'var(--text-3)',margin:'0 auto 14px'}}/>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}><div><strong style={{fontSize:19}}>Explore Volarix</strong><p style={{fontSize:12,color:'var(--text-2)',marginTop:3}}>All your market intelligence tools</p></div><button onClick={()=>setMore(false)} aria-label="Close menu" style={{border:'1px solid var(--border)',borderRadius:12,padding:'8px 12px',background:'var(--bg-card)',color:'var(--text-1)'}}>✕</button></div>
      {EXTRA.map(([id,label,description,icon])=><button key={id} onClick={()=>go(id)} style={{display:'flex',alignItems:'center',width:'100%',gap:13,textAlign:'left',padding:'12px 8px',border:0,borderBottom:'1px solid var(--border)',background:active===id?'rgba(139,92,246,.14)':'transparent',color:'var(--text-1)',borderRadius:8}}><span style={{width:36,height:36,borderRadius:10,background:'rgba(139,92,246,.17)',color:'#bc9aff',display:'grid',placeItems:'center',fontSize:20}}>{icon}</span><span style={{flex:1,minWidth:0}}><strong style={{fontSize:13}}>{label}</strong><small style={{display:'block',fontSize:11,color:'var(--text-2)',marginTop:3}}>{description}</small></span><span style={{color:'var(--text-3)'}}>›</span></button>)}
      <div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:12,paddingTop:14}}><span style={{fontSize:12,color:'var(--text-2)'}}>{isDark?'🌙 Dark':'☀️ Light'}</span><ThemeToggle isDark={isDark} onToggle={onThemeToggle}/></div>
    </div></>}
    <div style={{display:'flex',width:'100%',padding:'8px 4px max(10px,env(safe-area-inset-bottom))',background:'var(--bg-sidebar)'}}>
      {NAV.map(item=><button key={item.id} onClick={()=>go(item.id)} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:5,padding:'7px 2px',border:0,borderRadius:10,background:active===item.id?'rgba(167,139,250,.14)':'transparent',color:active===item.id?'#bc9aff':'var(--text-2)'}}><span style={{fontSize:21,lineHeight:1}}>{item.icon}</span><span style={{fontSize:10,fontWeight:active===item.id?700:500}}>{item.label}</span></button>)}
      <button aria-expanded={more} aria-label="More Volarix pages" onClick={()=>setMore(v=>!v)} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:5,padding:'7px 2px',border:0,borderRadius:10,background:more||moreActive?'rgba(167,139,250,.14)':'transparent',color:more||moreActive?'#bc9aff':'var(--text-2)'}}><span style={{fontSize:21,lineHeight:1}}>☷</span><span style={{fontSize:10,fontWeight:600}}>More</span></button>
    </div>
  </div>
}
