import { useEffect, useMemo, useState } from 'react'

const PAGES = [['overview','Overview'],['regime','Regime Analysis'],['bulletin','AI Intelligence'],['ask','Ask Volarix'],['backtest','Backtesting'],['about','About Volarix'],['market','Market Explorer'],['research','Research Lab'],['workspace','My Workspace'],['settings','Settings']]
const METRICS = [['total_return','Total return'],['ann_return','Annual return'],['sharpe_ratio','Sharpe ratio'],['max_drawdown','Maximum drawdown'],['ann_volatility','Annual volatility']]
const STRATEGIES = [['spy_stats','SPY Buy & Hold'],['naive_stats','Naive Switch'],['hyst_stats','Hysteresis']]
const EXPLAIN = {
  vix:'The VIX is an index of expected US equity market volatility. It is a risk indicator, not a prediction of market direction.',
  regime:'Volarix groups volatility into LOW, ELEVATED, and CRISIS conditions. These are model-defined stress categories, not guarantees.',
  forecast:'A forecast estimates future volatility using historical signals. It can be wrong; consider the prediction horizon and uncertainty.',
  shap:'SHAP helps explain how input features contribute to a model output. Feature attribution does not establish causation.',
  sharpe_ratio:'The Sharpe ratio compares excess return with return volatility. Historical values do not guarantee future performance.',
  max_drawdown:'Maximum drawdown measures the largest observed peak-to-trough portfolio decline over a tested period.',
}
const KEY='volarix-workspace-v1'
const safeRead=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch{return {}}}
const csvCell=v=>'"'+String(v??'').replace(/"/g,'""')+'"'
function download(name,content,type='text/plain'){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function Tile({children,style}){return <div style={{background:'var(--bg-card)',border:'1px solid var(--border)',borderRadius:14,padding:16,...style}}>{children}</div>}
const small={fontSize:12,color:'var(--text-2)',lineHeight:1.6}
const btn={border:'1px solid rgba(167,139,250,.35)',borderRadius:10,background:'var(--bg-card)',color:'var(--text-1)',padding:'9px 12px',cursor:'pointer',font:'inherit',fontSize:12}
const input={...btn,width:'100%',outlineColor:'#a78bfa'}
function NumberValue({value}){return <strong>{typeof value==='number'&&Number.isFinite(value)?value.toFixed(3):'Unavailable'}</strong>}
export default function IntelligenceWorkspace({page,onNav,latest,predict,analogues,backtest,shap}) {
  const [open,setOpen]=useState(false)
  const [tab,setTab]=useState('search')
  const [query,setQuery]=useState('')
  const [saved,setSaved]=useState(safeRead)
  const [explain,setExplain]=useState('vix')
  const [metric,setMetric]=useState('sharpe_ratio')
  const [first,setFirst]=useState('spy_stats')
  const [second,setSecond]=useState('hyst_stats')
  const [threshold,setThreshold]=useState('')
  const [notice,setNotice]=useState('')
  const [tour,setTour]=useState(-1)
  const [timeIndex,setTimeIndex]=useState(0)
  const [comparePeriod,setComparePeriod]=useState(false)
  const [availableAlerts,setAvailableAlerts]=useState([])
  const layout=saved.layout||['market','forecast','regime','backtest']
  const visible=saved.visible||['market','forecast','regime','backtest']
  const bookmarks=saved.bookmarks||[]
  const vix=Number(latest?.data?.vix)
  const regime=latest?.data?.regime
  const historical=useMemo(()=>{
    const source=analogues?.data
    const candidates=Array.isArray(source)?source:Array.isArray(source?.analogues)?source.analogues:Array.isArray(source?.history)?source.history:[]
    return candidates.filter(x=>x&&typeof x==='object').slice(0,60)
  },[analogues?.data])
  const setPref=(patch)=>setSaved(prev=>{const next={...prev,...patch};try{localStorage.setItem(KEY,JSON.stringify(next))}catch{}return next})
  useEffect(()=>{const fn=()=>{setTab('search');setOpen(true)};window.addEventListener('volarix-open-search',fn);return()=>window.removeEventListener('volarix-open-search',fn)},[])
  useEffect(()=>{const key=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();setTab('search');setOpen(true)}if(e.key==='Escape'){setOpen(false);setTour(-1)}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key)},[])
  useEffect(()=>{if(!Number.isFinite(vix)||!threshold)return;const limit=Number(threshold);if(!Number.isFinite(limit)||limit<=0)return;if(vix>=limit){setAvailableAlerts(prev=>prev.some(x=>x.limit===limit)?prev:[{limit,vix,at:new Date().toLocaleTimeString()},...prev].slice(0,5))}},[vix,threshold])
  const choose=(name)=>{setTab(name);setOpen(true)}
  const results=PAGES.filter(([id,name])=>(name+' '+id).toLowerCase().includes(query.toLowerCase()))
  const showTour=()=>{setTour(0);setOpen(true);setTab('tour')}
  const widgets={
    market:['Market snapshot', Number.isFinite(vix)?'VIX '+vix.toFixed(2):'Waiting for market data'],
    forecast:['Forecast horizon',predict?.data?'Forecast data available':'Forecast unavailable'],
    regime:['Risk regime',regime||'Unavailable'],
    backtest:['Strategy evaluation',backtest?.data?'Historical metrics available':'Backtest unavailable'],
  }
  const metricValue=(key)=>backtest?.data?.[key]?.[metric]
  const exportRows=[['Metric','Value','Source'],['VIX',Number.isFinite(vix)?vix:'','Latest market snapshot'],['Regime',regime||'','Latest market snapshot'],...STRATEGIES.flatMap(([key,label])=>METRICS.map(([m,ml])=>[label+' '+ml,backtest?.data?.[key]?.[m]??'','Historical backtest']))]
  const exportCSV=()=>download('volarix-insights.csv',exportRows.map(r=>r.map(csvCell).join(',')).join('\n'),'text/csv;charset=utf-8')
  const exportPNG=()=>{const canvas=document.createElement('canvas');canvas.width=1200;canvas.height=650;const ctx=canvas.getContext('2d');ctx.fillStyle='#151023';ctx.fillRect(0,0,1200,650);ctx.fillStyle='#d5b8ff';ctx.font='bold 54px sans-serif';ctx.fillText('Volarix | Market Snapshot',64,98);ctx.fillStyle='#ffffff';ctx.font='30px sans-serif';ctx.fillText('VIX: '+(Number.isFinite(vix)?vix.toFixed(2):'Unavailable'),64,185);ctx.fillText('Regime: '+(regime||'Unavailable'),64,240);ctx.font='24px sans-serif';STRATEGIES.forEach(([key,label],i)=>ctx.fillText(label+' Sharpe: '+(backtest?.data?.[key]?.sharpe_ratio??'Unavailable'),64,330+i*55));ctx.fillStyle='#baa5ce';ctx.font='20px sans-serif';ctx.fillText('Historical and model-derived values. Educational use only, not investment advice.',64,600);const a=document.createElement('a');a.download='volarix-snapshot.png';a.href=canvas.toDataURL('image/png');a.click()}
  const startAsk=()=>{onNav('ask');setOpen(false)}
  return <>
    {open&&<div role="presentation" onMouseDown={e=>{if(e.target===e.currentTarget)setOpen(false)}} style={{position:'fixed',inset:0,zIndex:100,background:'#090614bd',display:'flex',alignItems:'flex-start',justifyContent:'center',padding:'min(8vh,60px) 12px 18px'}}>
      <section role="dialog" aria-modal="true" aria-label="Volarix intelligence workspace" style={{background:'var(--bg-card)',color:'var(--text-1)',width:'min(940px,100%)',maxHeight:'85vh',overflow:'auto',border:'1px solid rgba(167,139,250,.45)',borderRadius:20,padding:20,boxShadow:'0 30px 90px #0008',fontFamily:'var(--font-sans)'}}>
        <header style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:12,marginBottom:16}}><div><strong style={{fontSize:20}}>Volarix Quick Tools</strong><div style={small}>Search, compare, explain, tour and export. Workspace has its own page.</div></div><button style={btn} onClick={()=>setOpen(false)} aria-label="Close workspace">Close ×</button></header>
        <nav aria-label="Workspace tools" style={{display:'flex',gap:7,overflowX:'auto',paddingBottom:12,marginBottom:14,borderBottom:'1px solid var(--border)'}}>
          {[['search','Search'],['compare','Compare'],['explain','Explain'],['tour','Tour'],['export','Export']].map(([id,name])=><button key={id} style={{...btn,whiteSpace:'nowrap',background:tab===id?'#6e43ad':'var(--bg-card)',color:tab===id?'#fff':'var(--text-1)'}} onClick={()=>setTab(id)}>{name}</button>)}
        </nav>
        {tab==='search'&&<div><input autoFocus aria-label="Search Volarix" placeholder="Search pages and tools..." value={query} onChange={e=>setQuery(e.target.value)} style={input}/><div style={{display:'grid',gap:7,marginTop:12}}>{results.map(([id,name])=><button key={id} style={{...btn,textAlign:'left'}} onClick={()=>{onNav(id);setOpen(false)}}>{name} <span style={{float:'right',color:'var(--text-2)'}}>Open page</span></button>)}{results.length===0&&<p style={small}>No matching pages. Try a different search.</p>}<button style={{...btn,textAlign:'left'}} onClick={startAsk}>Open Ask Volarix for a question</button></div></div>}
        {tab==='workspace'&&<div><h3>Personalize your dashboard</h3><p style={small}>Choose widgets and change their order. Your settings stay in this browser.</p><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(190px,1fr))',gap:12}}>{layout.map((key,i)=><Tile key={key}><label style={{display:'flex',gap:8,alignItems:'center'}}><input type="checkbox" checked={visible.includes(key)} onChange={e=>setPref({visible:e.target.checked?[...visible,key]:visible.filter(x=>x!==key)})}/><strong>{widgets[key][0]}</strong></label><p style={small}>{widgets[key][1]}</p><div style={{display:'flex',gap:6}}><button style={btn} disabled={i===0} onClick={()=>{const a=[...layout];[a[i-1],a[i]]=[a[i],a[i-1]];setPref({layout:a})}}>←</button><button style={btn} disabled={i===layout.length-1} onClick={()=>{const a=[...layout];[a[i+1],a[i]]=[a[i],a[i+1]];setPref({layout:a})}}>→</button></div></Tile>)}</div><h4>Your workspace</h4><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))',gap:10}}>{layout.filter(x=>visible.includes(x)).map(x=><Tile key={x}><strong>{widgets[x][0]}</strong><p style={small}>{widgets[x][1]}</p></Tile>)}</div><button style={{...btn,marginTop:14}} onClick={()=>setPref({layout:['market','forecast','regime','backtest'],visible:['market','forecast','regime','backtest']})}>Reset layout</button></div>}
        {tab==='timeline'&&<div><h3>Historical market timeline</h3><p style={small}>Only observations supplied by the existing historical analogues endpoint are shown. No market events or values are invented.</p>{historical.length?<><input type="range" min="0" max={historical.length-1} value={Math.min(timeIndex,historical.length-1)} onChange={e=>setTimeIndex(Number(e.target.value))} style={{width:'100%',accentColor:'#a78bfa'}}/><Tile><strong>Historical observation {timeIndex+1} of {historical.length}</strong><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(150px,1fr))',gap:10,marginTop:12}}>{[['Date',historical[Math.min(timeIndex,historical.length-1)]?.date],['VIX',historical[Math.min(timeIndex,historical.length-1)]?.vix],['Regime',historical[Math.min(timeIndex,historical.length-1)]?.regime_label||historical[Math.min(timeIndex,historical.length-1)]?.regime]].map(([label,value])=><Tile key={label}><div style={small}>{label}</div><strong>{value??'Unavailable'}</strong></Tile>)}</div></Tile></>:<Tile><p style={small}>Historical observations are not available in the current API response. Explore Regime Analysis for the existing analogue view.</p><button style={btn} onClick={()=>{onNav('regime');setOpen(false)}}>Open Regime Analysis</button></Tile>}</div>}
        {tab==='compare'&&<div><h3>Compare historical strategies</h3><p style={small}>Comparison uses actual available backtest metrics. Period comparison requires dated history and is not simulated.</p><label style={small}>Metric <select style={input} value={metric} onChange={e=>setMetric(e.target.value)}>{METRICS.map(([k,n])=><option key={k} value={k}>{n}</option>)}</select></label><div style={{display:'grid',gridTemplateColumns:'repeat(2,minmax(0,1fr))',gap:12,marginTop:12}}>{[[first,setFirst],[second,setSecond]].map(([selected,setSelected],i)=><Tile key={i}><select aria-label={'Strategy '+(i+1)} value={selected} onChange={e=>setSelected(e.target.value)} style={input}>{STRATEGIES.map(([key,label])=><option key={key} value={key}>{label}</option>)}</select><div style={{marginTop:18,fontSize:25}}><NumberValue value={metricValue(selected)}/></div></Tile>)}</div><p style={small}>Metric units follow the backend's reported values. These are historical comparisons, not forecasts.</p></div>}
        {tab==='alerts'&&<div><h3>Market watchlist & in-app alerts</h3><p style={small}>Threshold checks work only while this tab is open in your browser and new market data arrives. No push, email, or background monitoring is enabled.</p><Tile><strong>Current market</strong><p style={small}>VIX: {Number.isFinite(vix)?vix:'Unavailable'} · Regime: {regime||'Unavailable'}</p><button style={btn} onClick={()=>{const item={vix:Number.isFinite(vix)?vix:null,regime:regime||null,time:new Date().toISOString()};setPref({bookmarks:[item,...bookmarks].slice(0,20)})}}>Bookmark snapshot</button></Tile><div style={{marginTop:12}}><label style={small}>Notify in this workspace when VIX reaches or exceeds<input type="number" min="0.01" step="0.1" placeholder="e.g. 30" value={threshold} onChange={e=>setThreshold(e.target.value)} style={input}/></label></div>{availableAlerts.map((a,i)=><p key={i} role="status" style={{color:'#a78bfa'}}>VIX {a.vix} reached threshold {a.limit} at {a.at}</p>)}<h4>Saved snapshots ({bookmarks.length})</h4>{bookmarks.map((b,i)=><p key={i} style={small}>{new Date(b.time).toLocaleString()}: VIX {b.vix??'Unavailable'} · {b.regime??'Unavailable'} <button style={btn} onClick={()=>setPref({bookmarks:bookmarks.filter((_,j)=>i!==j)})}>Remove</button></p>)}</div>}
        {tab==='explain'&&<div><h3>Explain this metric</h3><p style={small}>Plain-language educational explanations. These are not generated live by AI.</p><select style={input} value={explain} onChange={e=>setExplain(e.target.value)}>{Object.keys(EXPLAIN).map(k=><option key={k} value={k}>{k.replaceAll('_',' ').toUpperCase()}</option>)}</select><Tile style={{marginTop:14}}><p style={{...small,fontSize:15}}>{EXPLAIN[explain]}</p></Tile><button style={{...btn,marginTop:12}} onClick={startAsk}>Ask Volarix for more context</button></div>}
        {tab==='tour'&&<div><h3>Explore Volarix</h3>{tour<0?<><p style={small}>A short guided introduction to the platform. You can skip or replay anytime.</p><button style={btn} onClick={()=>setTour(0)}>Start tour</button></>:<Tile><p style={small}>Step {tour+1} of 4</p><h3>{['Understand the market','Explore risk regimes','Ask questions','Evaluate strategies'][tour]}</h3><p style={small}>{['Overview displays current market indicators and model forecasts.','Regime Analysis explores market stress categories and available explanations.','Ask Volarix provides conversational market context subject to API limits.','Backtesting evaluates historical strategies using real reported metrics.'][tour]}</p><div style={{display:'flex',gap:8}}><button style={btn} onClick={()=>setTour(-1)}>Skip</button><button style={btn} onClick={()=>{if(tour===3){setTour(-1);setPref({tourSeen:true})}else setTour(tour+1)}}>{tour===3?'Finish':'Next'}</button><button style={btn} onClick={()=>{onNav(['overview','regime','ask','backtest'][tour]);setOpen(false)}}>Open page</button></div></Tile>}</div>}
        {tab==='export'&&<div><h3>Export and share insights</h3><p style={small}>Exports include only current available market and historical backtest values. Missing values are labeled unavailable. Financial analysis is not investment advice.</p><div style={{display:'flex',gap:9,flexWrap:'wrap'}}><button style={btn} onClick={exportCSV}>Download CSV</button><button style={btn} onClick={exportPNG}>Download PNG snapshot</button><button style={btn} onClick={()=>window.print()}>Print / Save as PDF</button><button style={btn} onClick={async()=>{const url=window.location.href;try{await navigator.clipboard.writeText(url);setNotice('Page link copied')}catch{setNotice('Copy the page URL from your browser')}}}>Copy page link</button></div>{notice&&<p role="status" style={small}>{notice}</p>}<p style={{...small,marginTop:18}}>For PDF, use your browser's “Save as PDF” destination. This exports the current page, not a separately generated financial report.</p></div>}
      </section>
    </div>}
  </>
}
