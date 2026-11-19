import useIsMobile from '../useIsMobile'
import { useTheme } from '../ThemeContext'
import { timeAgo } from '../utils'

/**
 * Shared visual language of the Regime Analysis heading.
 * Existing page content and actions are passed through unchanged.
 */
export default function PageHeader({title,subtitle,page,lastUpdated,regime,regimeLabel,children}) {
  const mobile=useIsMobile()
  const dark=useTheme()
  const label=page==='about'?'Platform information':page==='backtest'?'Historical analysis':page==='bulletin'?'Market intelligence':page==='ask'?'Market intelligence':'Market intelligence'
  return <header className={'volarix-unified-header'+(!dark?' is-light':'')} style={{
    position:'relative',isolation:'isolate',overflow:'hidden',
    display:'flex',alignItems:'center',justifyContent:'space-between',gap:16,flexWrap:'wrap',
    padding:mobile?'22px 16px 24px':'30px 30px 32px',minHeight:mobile?154:174,
    background:dark?'radial-gradient(ellipse at 80% 0%,rgba(130,94,214,.29),transparent 55%),linear-gradient(120deg,#111322,#25203e 80%,#17152a)':'radial-gradient(ellipse at 80% 0%,rgba(176,150,245,.32),transparent 55%),linear-gradient(120deg,#faf9ff,#eae3fa 80%,#f8f6ff)',
    borderBottom:dark?'1px solid rgba(201,188,244,.22)':'1px solid rgba(113,94,170,.18)',
    color:dark?'#f7f5ff':'#211b36',fontFamily:'var(--font-sans)'
  }}>
    <svg aria-hidden="true" viewBox="0 0 1100 210" preserveAspectRatio="none" style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none',zIndex:-1}}>
      <path d="M 475 -55 C 760 210 1050 195 1190 70" stroke={dark?'rgba(187,157,255,.67)':'rgba(139,98,214,.42)'} strokeWidth="1.4" fill="none"/>
    </svg>
    <div style={{minWidth:0,maxWidth:'min(750px,100%)'}}>
      <div style={{fontSize:mobile?11:13,fontWeight:650,color:dark?'#c19cff':'#7846b5',marginBottom:13}}>✧ Market Intelligence by Volarix</div>
      <h1 style={{fontSize:mobile?'clamp(25px,6vw,32px)':'clamp(30px,3vw,40px)',fontWeight:750,letterSpacing:'-.045em',lineHeight:1.14,margin:'0 0 10px',color:'inherit'}}>{title}</h1>
      <p style={{fontSize:mobile?12:13,lineHeight:1.55,margin:0,color:dark?'#c5bfd8':'#655e7a'}}>{subtitle}</p>
    </div>
    <div style={{display:'flex',alignItems:'center',gap:10,flexWrap:'wrap',position:'relative'}}>
      {lastUpdated&&<span style={{fontSize:11,color:dark?'#c9c0dd':'#655e7a'}}>Updated {timeAgo(lastUpdated)}</span>}
      {regime&&page!=='ask'&&<span style={{display:'inline-flex',alignItems:'center',gap:7,padding:'9px 12px',borderRadius:12,background:'rgba(115,104,160,.17)',border:'1px solid rgba(185,161,255,.3)',fontSize:12,color:'inherit'}}><span style={{height:8,width:8,borderRadius:99,background:regime==='LOW'?'#49e2a4':regime==='ELEVATED'?'#fbbf24':regime==='CRISIS'?'#fb7185':'#a78bfa'}}/>{regimeLabel||regime}</span>}
      {children||(!regime&&<span style={{display:'inline-flex',alignItems:'center',gap:8,padding:'10px 14px',borderRadius:13,background:'rgba(115,104,160,.17)',border:'1px solid rgba(185,161,255,.3)',fontSize:12,whiteSpace:'nowrap',color:'inherit'}}><span style={{width:8,height:8,borderRadius:99,background:'#bca0f8'}}/>{label}</span>)}
    </div>
  </header>
}
