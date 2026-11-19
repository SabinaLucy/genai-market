import React, { useState } from 'react'
import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'
import { fmt, pct } from '../utils'

const STRATEGIES = [
  {key:'spy_stats',label:'SPY Buy and Hold',short:'SPY',color:'#a78bfa',emoji:'📈',description:'Stay invested in SPY throughout the test period.'},
  {key:'naive_stats',label:'Naive Regime Switch',short:'Naive',color:'#f5b84b',emoji:'🔄',description:'Switch to cash when a crisis is detected.'},
  {key:'hyst_stats',label:'Hysteresis Strategy',short:'Hysteresis',color:'#66cbb5',emoji:'🛡️',description:'Require repeated regime signals before switching.'},
]
const METRICS = [
  {key:'total_return',label:'Total return',format:pct,higher:true,description:'Cumulative growth over the historical test.'},
  {key:'ann_return',label:'Annual return',format:pct,higher:true,description:'Annualized growth during the test.'},
  {key:'sharpe_ratio',label:'Sharpe ratio',format:v=>fmt(v,3),higher:true,description:'Return relative to risk. Higher is generally better.'},
  {key:'max_drawdown',label:'Maximum drawdown',format:pct,higher:true,description:'Largest peak-to-trough loss. Closer to zero is better.'},
  {key:'ann_volatility',label:'Annual volatility',format:pct,higher:false,description:'How much returns fluctuated each year.'},
]
const valid = v => typeof v==='number' && Number.isFinite(v)
function MetricCard({title,value,sub,accent}) {
  return <div className="bt-metric"><div className="bt-metric-label">{title}</div><strong style={{color:accent||'var(--text-1)'}}>{value}</strong><span>{sub}</span></div>
}
function CompareBars({data,metric}) {
  const entries=STRATEGIES.map(s=>({...s,value:data?.[s.key]?.[metric.key]})).filter(s=>valid(s.value))
  if(!entries.length)return <p className="bt-muted">Comparison data is unavailable.</p>
  const lo=Math.min(0,...entries.map(s=>s.value)), hi=Math.max(0,...entries.map(s=>s.value))
  const range=hi-lo||1
  const zero=(0-lo)/range*100
  return <div className="bt-bars">{entries.map(s=>{
    const p=(s.value-lo)/range*100, left=Math.min(zero,p),width=Math.max(1,Math.abs(p-zero))
    return <div key={s.key} className="bt-bar-row"><div className="bt-bar-label"><span>{s.short}</span><strong>{metric.format(s.value)}</strong></div><div className="bt-bar-track"><div className="bt-zero" style={{left:zero+'%'}}/><div className="bt-bar-fill" style={{left:left+'%',width:width+'%',background:s.color}}/></div></div>
  })}</div>
}
export default function BacktestPage({backtest}) {
  const isMobile=useIsMobile()
  const [selected,setSelected]=useState('total_return')
  const d=backtest?.data
  const available=STRATEGIES.filter(s=>valid(d?.[s.key]?.sharpe_ratio))
  const best=available.length?[...available].sort((a,b)=>d[b.key].sharpe_ratio-d[a.key].sharpe_ratio)[0]:null
  const spy=d?.spy_stats,hyst=d?.hyst_stats
  const selectedMetric=METRICS.find(m=>m.key===selected)||METRICS[0]
  const spread=valid(spy?.total_return)&&valid(hyst?.total_return)?hyst.total_return-spy.total_return:null
  return <div className="fade-up bt-page">
    <style>{`
      .bt-page,.bt-page button,.bt-page p,.bt-page span,.bt-page td,.bt-page th,.bt-page select{font-family:var(--font-sans)!important}
      .bt-body{padding:24px 32px 36px;display:flex;flex-direction:column;gap:20px}
      .bt-intro{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
      .bt-intro h2{font-size:18px;font-weight:750;color:var(--text-1);margin:0 0 5px}
      .bt-muted{font-size:13px;color:var(--text-2);line-height:1.6;margin:0}
      .bt-period{font-size:12px;color:var(--text-2);background:var(--bg-card);border:1px solid var(--border);border-radius:999px;padding:8px 12px}
      .bt-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
      .bt-metric,.bt-panel,.bt-strategy{background:linear-gradient(145deg,var(--bg-card),rgba(139,92,246,.045));border:1px solid rgba(167,139,250,.23);border-radius:16px}
      .bt-metric{padding:19px;display:flex;flex-direction:column;gap:7px}
      .bt-metric-label{font-size:12px;color:var(--text-2);font-weight:600}
      .bt-metric strong{font-size:25px;letter-spacing:-.035em}
      .bt-metric>span{font-size:12px;color:var(--text-2);line-height:1.5}
      .bt-panel{padding:22px}
      .bt-panel h3{font-size:17px;color:var(--text-1);margin:0 0 6px;font-weight:750}
      .bt-main-grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:16px}
      .bt-strategies{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
      .bt-strategy{padding:19px;display:flex;flex-direction:column;gap:12px}
      .bt-strategy-top{display:flex;align-items:center;gap:9px;color:var(--text-1);font-size:13px;font-weight:700}
      .bt-strategy-top .bt-emoji{font-size:21px;background:rgba(167,139,250,.12);border-radius:10px;padding:8px}
      .bt-best{margin-left:auto;font-size:10px;color:#bca2f5;border:1px solid rgba(167,139,250,.4);padding:5px 7px;border-radius:999px;white-space:nowrap}
      .bt-strategy-value{font-size:26px;font-weight:750;letter-spacing:-.035em}
      .bt-strategy small{font-size:12px;color:var(--text-2);line-height:1.6}
      .bt-strategy-stats{display:flex;flex-direction:column;gap:9px;padding-top:10px;border-top:1px solid var(--border)}
      .bt-stat-row{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--text-2)}
      .bt-stat-row strong{color:var(--text-1);font-weight:650}
      .bt-select{background:var(--bg-card);color:var(--text-1);border:1px solid rgba(167,139,250,.4);border-radius:9px;padding:9px 12px;font-size:13px}
      .bt-bars{display:flex;flex-direction:column;gap:24px;margin-top:26px;margin-bottom:16px}
      .bt-bar-label{display:flex;justify-content:space-between;gap:12px;margin-bottom:8px;font-size:13px;color:var(--text-2)}
      .bt-bar-label strong{color:var(--text-1)}
      .bt-bar-track{position:relative;height:13px;background:rgba(167,139,250,.10);border-radius:999px;overflow:hidden}
      .bt-bar-fill{position:absolute;top:0;height:100%;border-radius:999px;min-width:3px;transition:width .25s}
      .bt-zero{position:absolute;top:0;bottom:0;width:1px;background:var(--text-2);opacity:.6;z-index:1}
      .bt-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
      .bt-legend span{font-size:12px;color:var(--text-2);display:inline-flex;align-items:center;gap:7px}
      .bt-legend i{display:inline-block;width:8px;height:8px;border-radius:50%}
      .bt-insight{background:rgba(167,139,250,.07);border:1px solid rgba(167,139,250,.18);border-radius:12px;padding:15px;margin-top:18px}
      .bt-insight strong{display:block;color:var(--text-1);font-size:13px;margin-bottom:7px}
      .bt-insight p{font-size:13px;line-height:1.7;color:var(--text-2);margin:0}
      .bt-table-wrap{overflow-x:auto}
      .bt-table{border-collapse:collapse;width:100%;min-width:640px}
      .bt-table th{text-align:right;font-size:12px;color:var(--text-2);font-weight:600;padding:12px 10px;border-bottom:1px solid var(--border)}
      .bt-table th:first-child,.bt-table td:first-child{text-align:left}
      .bt-table td{text-align:right;font-size:13px;color:var(--text-1);padding:15px 10px;border-bottom:1px solid var(--border)}
      .bt-table tr:last-child td{border-bottom:none}
      @media(max-width:1050px){.bt-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.bt-main-grid{grid-template-columns:1fr}}
      @media(max-width:700px){.bt-body{padding:14px;gap:14px}.bt-strategies{grid-template-columns:1fr}.bt-panel{padding:16px}.bt-metric{padding:14px}.bt-metric strong{font-size:22px}}
    `}</style>
    <PageHeader title="Backtesting" subtitle="See how Volarix strategies performed in historical markets" page="backtest"/>
    <div className="bt-body">
      {backtest?.loading?<div className="bt-panel"><p className="bt-muted">Loading historical backtesting results…</p></div>
      :backtest?.error?<div className="bt-panel"><p className="bt-muted">{backtest.error}</p></div>
      :!d?<div className="bt-panel"><p className="bt-muted">Historical backtesting results are not available yet.</p></div>
      :<>
        <div className="bt-intro"><div><h2>Strategy performance</h2><p className="bt-muted">Compare growth, risk, and consistency across three historical strategies.</p></div><span className="bt-period">{d.test_period_start||'Start unavailable'} – {d.test_period_end||'End unavailable'}</span></div>
        <div className="bt-kpis">
          <MetricCard title="Best Sharpe ratio" value={best?fmt(d[best.key].sharpe_ratio,3):'—'} sub={best?.label||'No data'} accent="#b99cff"/>
          <MetricCard title="SPY total return" value={valid(spy?.total_return)?pct(spy.total_return):'—'} sub="Buy and hold benchmark"/>
          <MetricCard title="Hysteresis total return" value={valid(hyst?.total_return)?pct(hyst.total_return):'—'} sub="Regime-aware strategy"/>
          <MetricCard title="Hysteresis vs SPY" value={spread===null?'—':(spread>0?'+':'')+(spread*100).toFixed(1)+' pp'} sub="Total-return difference" accent={spread===null?undefined:spread>=0?'#66cbb5':'#f5b84b'}/>
        </div>
        <div className="bt-main-grid">
          <section className="bt-panel"><div className="bt-intro"><div><h3>Strategy comparison</h3><p className="bt-muted">Actual aggregate backtest metrics, not simulated price curves.</p></div><select className="bt-select" value={selected} onChange={e=>setSelected(e.target.value)} aria-label="Comparison metric">{METRICS.map(m=><option key={m.key} value={m.key}>{m.label}</option>)}</select></div><CompareBars data={d} metric={selectedMetric}/><p className="bt-muted">{selectedMetric.description}</p><div className="bt-legend">{STRATEGIES.map(s=><span key={s.key}><i style={{background:s.color}}/>{s.label}</span>)}</div></section>
          <section className="bt-panel"><h3>Risk analysis</h3><p className="bt-muted">A closer look at downside exposure and fluctuations.</p><div className="bt-insight"><strong>Maximum drawdown</strong><p>{STRATEGIES.map(s=>s.short+': '+(valid(d[s.key]?.max_drawdown)?pct(d[s.key].max_drawdown):'—')).join(' · ')}</p></div><div className="bt-insight"><strong>Annualized volatility</strong><p>{STRATEGIES.map(s=>s.short+': '+(valid(d[s.key]?.ann_volatility)?pct(d[s.key].ann_volatility):'—')).join(' · ')}</p></div><div className="bt-insight"><strong>How to read this</strong><p>Higher returns are not always better if a strategy takes much more risk. Compare Sharpe ratio, volatility, and drawdown together.</p></div></section>
        </div>
        <section><div className="bt-intro" style={{marginBottom:12}}><div><h2>Explore each strategy</h2><p className="bt-muted">See the approach behind each result.</p></div></div><div className="bt-strategies">{STRATEGIES.map(s=>{const stats=d[s.key];return <div key={s.key} className="bt-strategy"><div className="bt-strategy-top"><span className="bt-emoji">{s.emoji}</span><span>{s.label}</span>{best?.key===s.key&&<span className="bt-best">★ Best Sharpe</span>}</div><small>{s.description}</small><div className="bt-strategy-value" style={{color:s.color}}>{valid(stats?.sharpe_ratio)?fmt(stats.sharpe_ratio,3):'—'}</div><small>Sharpe ratio</small><div className="bt-strategy-stats">{[['Total return','total_return'],['Annual return','ann_return'],['Max drawdown','max_drawdown']].map(([label,key])=><div key={key} className="bt-stat-row"><span>{label}</span><strong>{valid(stats?.[key])?pct(stats[key]):'—'}</strong></div>)}</div></div>})}</div></section>
        <section className="bt-panel"><h3>Full performance breakdown</h3><p className="bt-muted" style={{marginBottom:14}}>Historical results, including transaction costs and risk assumptions.</p><div className="bt-table-wrap"><table className="bt-table"><thead><tr><th>Strategy</th>{METRICS.map(m=><th key={m.key}>{m.label}</th>)}</tr></thead><tbody>{STRATEGIES.map(s=><tr key={s.key}><td><span style={{color:s.color,fontWeight:700}}>{s.label}</span></td>{METRICS.map(m=><td key={m.key}>{valid(d[s.key]?.[m.key])?m.format(d[s.key][m.key]):'—'}</td>)}</tr>)}</tbody></table></div><p className="bt-muted" style={{marginTop:14}}>Transaction cost: {valid(d.transaction_cost)?pct(d.transaction_cost):'—'} · Risk-free rate: {valid(d.risk_free_rate)?pct(d.risk_free_rate):'—'}. Historical performance does not guarantee future results.</p></section>
      </>}
    </div>
  </div>
}
