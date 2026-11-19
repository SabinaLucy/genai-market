import '../styles/regime.css'
import RegimeBadge from '../components/RegimeBadge'
import { fmt } from '../utils'
import React, { useState } from 'react'
import ReactDOM from 'react-dom'

function InfoTip({ text, title }) {
  const [open, setOpen] = useState(false)
  const [pos,  setPos]  = useState({ top: 0, left: 0 })
  const btnRef          = React.useRef(null)

  const handleClick = () => {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom
      setPos({
        top:  spaceBelow > 200 ? rect.bottom + 10 : rect.top - 220,
        left: Math.max(12, Math.min(rect.left - 100, window.innerWidth - 280)),
      })
    }
    setOpen(o => !o)
  }

  return (
    <>
      <button ref={btnRef} onClick={handleClick} style={{ width: 18, height: 18, borderRadius: 6, background: open ? 'rgba(139,92,246,0.3)' : 'rgba(139,92,246,0.12)', border: `1px solid ${open ? 'rgba(139,92,246,0.7)' : 'rgba(139,92,246,0.4)'}`, color: '#a78bfa', fontSize: 10, fontWeight: 700, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0, lineHeight: 1, padding: 0, marginLeft: 6, boxShadow: open ? '0 0 10px rgba(139,92,246,0.4)' : 'none', transition: 'all 0.2s', fontFamily: 'var(--font-mono)' }}>i</button>
      {open && ReactDOM.createPortal(
        <>
          <div onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 9998 }} />
          <div style={{ position: 'fixed', top: pos.top, left: pos.left, width: 260, background: 'linear-gradient(135deg,#1c1c3a 0%,#12122a 100%)', border: '1px solid rgba(139,92,246,0.45)', borderRadius: 14, overflow: 'hidden', boxShadow: '0 20px 60px rgba(0,0,0,0.9)', zIndex: 9999 }}>
            <div style={{ height: 3, background: 'linear-gradient(90deg,#7c3aed,#06b6d4)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ width: 24, height: 24, borderRadius: 7, background: 'rgba(139,92,246,0.15)', border: '1px solid rgba(139,92,246,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: '#a78bfa', fontWeight: 700, flexShrink: 0 }}>i</div>
              <div style={{ fontSize: 10, fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#a78bfa', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{title || 'What is this?'}</div>
            </div>
            <div style={{ padding: '10px 14px 14px', fontSize: 12, lineHeight: 1.7, color: '#c8c8f0' }}>{text}</div>
          </div>
        </>,
        document.body
      )}
    </>
  )
}

const FEATURE_INFO = {
  vix_lag1: { label: "Previous Day's Volatility", description: "The VIX level from the previous trading day." },
  vix_lag5: { label: "Volatility 5 Days Ago", description: "The VIX level five trading days earlier." },
  vix_lag21: { label: "Volatility 21 Days Ago", description: "The VIX level 21 trading days earlier." },
  vix_roll_mean5: { label: "Average Volatility (5 Days)", description: "The average VIX level over the last five trading days." },
  vix_roll_std21: { label: "Volatility Variation (21 Days)", description: "How much VIX readings varied over the last 21 trading days." },
  fedfunds: { label: "Federal Reserve Interest Rate", description: "The federal funds rate, a key benchmark for short-term interest rates." },
  cpi: { label: "Inflation Rate (CPI)", description: "Consumer price changes measured by the Consumer Price Index." },
  unrate: { label: "Unemployment Rate", description: "The share of the labor force currently unemployed." },
  gs10: { label: "Long-Term Interest Rates", description: "The yield on 10-year U.S. Treasury securities." },
  indpro: { label: "Industrial Production", description: "An index tracking output from factories, mines, and utilities." },
  m2sl: { label: "Money Supply", description: "A broad measure of money available in the economy." },
  sentiment: { label: "Market News Sentiment", description: "The model's measure of positive or negative financial news." },
}

const RC = [
  { cls:'rc-low',    ico:'◈', name:'Stable Market',   big:'LOW',      bColor:'#22c55e', tColor:'#4ade80', lbl:'[STABLE]', regime:'LOW',      probs:[77,18,5]  },
  { cls:'rc-elev',   ico:'◉', name:'Elevated Stress', big:'ELEVATED', bColor:'#f59e0b', tColor:'#fbbf24', lbl:'[WATCH]',  regime:'ELEVATED', probs:[8,88,4]   },
  { cls:'rc-crisis', ico:'⚡', name:'Crisis Alert',    big:'CRISIS',   bColor:'#ef4444', tColor:'#f87171', lbl:'[ALERT]',  regime:'CRISIS',   probs:[2,15,83]  },
]
const PROB_LABELS = ['Stable','Elevated','Crisis']
const PROB_COLORS = ['#22c55e','#f59e0b','#ef4444']


export default function RegimePage({latest,analogues,shap}) {
 const ld=latest.data, regime=ld?.regime||'UNKNOWN', vix=ld?.vix
 const drivers=shap.data?.top_drivers||[]
 const maxVal=Math.max(.001,...drivers.map(d=>Math.abs(Number(d.shap_value??d.importance??0))))
 const current=regime==='LOW'?'Stable':regime==='ELEVATED'?'Elevated':regime==='CRISIS'?'Crisis':'Unavailable'
 const tone=regime==='LOW'?'stable':regime==='CRISIS'?'crisis':'elevated'
 const match=(analogues.data?.analogues||[]).slice(0,3)
 const probabilities=ld?.regime_probabilities||ld?.probabilities
 const probability=probabilities&&typeof probabilities==='object'?Number(probabilities[regime]??probabilities[regime.toLowerCase()]??probabilities[current.toLowerCase()]):null
 const percent=Number.isFinite(probability)?Math.round(probability<=1?probability*100:probability):null
 return <div className="vol-regime-page fade-up">
   <header className="vr-heading"><div><div className="vr-eyebrow">✧ Market Intelligence by Volarix</div><h1>Regime Analysis</h1><p>Decode market stress. Understand what moves volatility.</p></div><div className="vr-live"><span/> {latest.loading?'Updating':'Live market data'}</div></header>
   <section className="vr-hero vr-glass">
    <div className="vr-hero-top"><span>CURRENT MARKET REGIME <InfoTip title="What is Regime?" text="Volarix classifies markets as Stable (VIX below 20), Elevated (VIX 20 to 30), or Crisis (VIX above 30). This is a market-stress classification, not an investment recommendation."/></span><span>MODEL CLASSIFICATION</span></div>
    <div className="vr-hero-content"><div className="vr-hero-main"><div className={'vr-regime-name '+tone}>{latest.loading&&!ld?'LOADING':current.toUpperCase()}</div><p>{regime==='LOW'?'Markets are relatively calm':regime==='CRISIS'?'Market stress is unusually high':regime==='ELEVATED'?'Market stress is above normal':'Waiting for market classification'}</p><div className="vr-vix-chip">VIX <strong>{vix==null?'—':fmt(vix)}</strong>{ld?.date&&<span> · {ld.date}</span>}</div></div><div className="vr-orb-area"><div className={'vr-orb '+tone}><div className="vr-orb-inner"><span>{percent!=null?percent+'%':current}</span><small>{percent!=null?'Model probability':'Current regime'}</small></div></div>{percent==null&&<small className="vr-data-note">Probability not supplied by API</small>}</div></div>
    <div className="vr-insight"><span>✧</span><p><strong>Volarix Insight</strong> · {regime==='ELEVATED'?'Elevated volatility suggests increased uncertainty, below crisis levels.':regime==='CRISIS'?'High volatility signals heightened market uncertainty.':regime==='LOW'?'Lower volatility indicates relatively calmer market conditions.':'Market insight will appear when live data is available.'}</p></div>
   </section>
   <section className="vr-regime-grid" aria-label="Market regime definitions">
   {[{name:'STABLE',subtitle:'Low market stress',range:'VIX below 20',icon:'◇',tone:'stable',key:'LOW'},{name:'ELEVATED',subtitle:'Moderate market stress',range:'VIX 20–30',icon:'◈',tone:'elevated',key:'ELEVATED'},{name:'CRISIS',subtitle:'High market stress',range:'VIX above 30',icon:'⚡',tone:'crisis',key:'CRISIS'}].map(x=><div className={'vr-regime-card vr-glass '+x.tone+(regime===x.key?' selected':'')} key={x.name}>{regime===x.key&&<span className="vr-current"><span className="vr-current-dot"/> CURRENT</span>}<div className="vr-regime-icon">{x.icon}</div><div className="vr-regime-title">{x.name}</div><div className="vr-regime-sub">{x.subtitle}</div><div className="vr-regime-range">{x.range}</div>{regime===x.key&&<span className="vr-regime-accent" aria-hidden="true"/>}</div>)}
   </section>
   <div className="vr-bottom-grid">
    <section className="vr-panel vr-glass vr-drivers-panel">
      <div className="vr-panel-heading"><div><h2>What Drives Market Stress? <InfoTip title="How to read these factors" text="These bars show which inputs most influence the model's prediction. Longer bars mean a larger contribution. An up arrow increases the model output; a down arrow decreases it. This explains the model, not direct cause and effect in markets."/></h2><p>The key factors influencing today's market volatility</p></div><span className="vr-panel-icon">✧</span></div>
      <div className="vr-ai-explainer"><span className="vr-ai-explainer-icon">✧</span><div><strong>AI Market Analysis</strong><p>These factors have the strongest influence on the model's market stress assessment.</p></div></div>
      {shap.loading?<p className="vr-empty">Loading model explanations…</p>:shap.error?<p className="vr-empty">{shap.error}</p>:drivers.length===0?<p className="vr-empty">No model-driver data available.</p>:<div className="vr-driver-list">{drivers.slice(0,6).map((d,i)=>{const raw=Number(d.shap_value??d.importance??0),pos=raw>=0,info=FEATURE_INFO[d.feature];return <div className="vr-driver" key={d.feature||i}><div className="vr-driver-row"><div className="vr-driver-label"><span>{info?.label||d.feature||'Feature '+(i+1)}</span><small>{info?.description||'A model input influencing the prediction.'}</small></div><div className="vr-driver-value"><span className={pos?'up':'down'}>{pos?'↑':'↓'} {fmt(Math.abs(raw),4)}</span><small>Influence</small></div></div><div className="vr-track"><div className={pos?'up':'down'} style={{width:Math.min(100,Math.abs(raw)/maxVal*100)+'%'}}/></div></div>})}</div>}
      {shap.data?.stability_rho!=null&&<div className="vr-stability"><span>◇ &nbsp; Model Stability <InfoTip title="Model Stability" text="A model diagnostic indicating how consistently the model's feature rankings behave. It does not guarantee that forecasts are accurate."/></span><strong>{fmt(shap.data.stability_rho,3)}</strong></div>}
    </section>
    <section className="vr-panel vr-glass"><div className="vr-panel-heading"><div><h2>Historical Market Analogues <InfoTip title="What are Analogues?" text="Volarix identifies historical market periods resembling current conditions. Similarity scores describe resemblance, not a guarantee of future market behavior."/></h2><p>Explore similar conditions from the past</p></div><span className="vr-panel-icon">◷</span></div>
     {analogues.loading?<p className="vr-empty">Finding historical matches…</p>:analogues.error?<p className="vr-empty">{analogues.error}</p>:match.length===0?<p className="vr-empty">No historical matches available.</p>:<div className="vr-analogue-list">{match.map((a,i)=><div className="vr-analogue" key={a.date||i}><div className="vr-analogue-top"><strong>{a.date||'Historical period '+(i+1)}</strong><RegimeBadge regime={a.regime} regimeLabel={a.regime_label} size="sm"/></div><div className="vr-analogue-meta">{a.vix!=null&&<span>VIX then <strong>{fmt(a.vix)}</strong></span>}{a.similarity!=null&&<span>Similarity <strong>{(a.similarity*100).toFixed(1)}%</strong></span>}</div>{a.similarity!=null&&<div className="vr-track"><div className="match" style={{width:Math.max(0,Math.min(100,a.similarity*100))+'%'}}/></div>}</div>)}</div>}
    </section>
   </div>
   <p className="vr-disclaimer">Regime labels and explanations are informational. Forecasts and historical similarities are uncertain and are not investment advice.</p>
 </div>
}
