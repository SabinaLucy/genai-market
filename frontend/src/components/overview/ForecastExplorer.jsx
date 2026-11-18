import {useEffect,useState} from 'react'
import {fetchPredict} from '../../api'
import InfoTooltip from '../InfoTooltip'

const HORIZONS=[1,5,10]
const fmt=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toFixed(2)
const arrow=(data,current)=>data?.direction==='up'?'↑':data?.direction==='down'?'↓':data?.predicted_vix!=null&&current!=null?(data.predicted_vix>current?'↑':data.predicted_vix<current?'↓':'→'):''
export default function ForecastExplorer({latest,predict,horizon,setHorizon}){
 const [forecasts,setForecasts]=useState({})
 const [fetching,setFetching]=useState(false)
 const current=latest?.data?.vix
 // Keep the original dashboard's selected horizon and live prediction as the source of truth.
 useEffect(()=>{if(predict?.data)setForecasts(previous=>({...previous,[horizon]:predict.data}))},[predict?.data,horizon])
 // Fetch the other two actual model horizons instead of manufacturing values by scaling a single forecast.
 useEffect(()=>{
  let active=true
  const other=HORIZONS.filter(h=>h!==horizon)
  setFetching(true)
  Promise.allSettled(other.map(h=>fetchPredict(h))).then(results=>{
   if(!active)return
   setForecasts(previous=>{
    const next={...previous}
    results.forEach((result,index)=>{if(result.status==='fulfilled')next[other[index]]=result.value.data})
    return next
   })
  }).finally(()=>{if(active)setFetching(false)})
  return()=>{active=false}
 },[horizon])
 const selected=predict?.data||forecasts[horizon]
 const low=Number(selected?.interval_lo),high=Number(selected?.interval_hi),mid=Number(selected?.predicted_vix)
 const validInterval=selected?.interval_lo!=null&&selected?.interval_hi!=null&&high>low
 const marker=validInterval&&selected?.predicted_vix!=null?Math.max(2,Math.min(98,100*(mid-low)/(high-low))):50
 return <section className="vol-panel vol-forecast-explorer" aria-label="VIX Forecast Explorer">
  <div className="vol-forecast-header">
   <div className="vol-forecast-heading"><div className="vol-forecast-emblem">↗</div><div><h2>VIX Forecast Explorer <InfoTooltip title="VIX Forecast Explorer" text="Choose 1, 5, or 10 trading days to see predictions from the Volarix forecasting model. The forecast arrow shows the predicted direction relative to the current VIX. The 90% prediction interval describes forecast uncertainty."/></h2><p>Explore model predictions across three trading horizons</p></div></div>
   <div className="vol-forecast-horizons" aria-label="Forecast horizon">{HORIZONS.map(h=><button key={h} type="button" aria-pressed={horizon===h} className={horizon===h?'active':''} onClick={()=>setHorizon(h)}>{h} Day{h>1?'s':''}</button>)}</div>
  </div>
  <div className="vol-forecast-body">
   <div className="vol-forecast-primary"><div className="vol-forecast-overline">PREDICTED VIX · {horizon}-DAY HORIZON</div><div className="vol-forecast-big"><strong>{fmt(selected?.predicted_vix)}</strong><span className={arrow(selected,current)==='↑'?'rise':'fall'}>{arrow(selected,current)}</span></div><p>From current VIX <strong>{fmt(current)}</strong></p></div>
   <div className="vol-forecast-horizon-cards">{HORIZONS.map(h=>{const data=h===horizon?selected:forecasts[h];return <button key={h} type="button" className={'vol-forecast-mini '+(h===horizon?'active':'')} onClick={()=>setHorizon(h)} aria-pressed={h===horizon}><span>{h}-DAY {h===horizon?'✦':''}</span><strong>{fmt(data?.predicted_vix)} <i className={arrow(data,current)==='↑'?'rise':'fall'}>{arrow(data,current)}</i></strong></button>})}</div>
   <div className="vol-forecast-confidence"><div className="vol-forecast-ci-title"><span>90% Prediction Interval <InfoTooltip title="90% Prediction Interval" text="The model's lower and upper uncertainty bounds for the selected forecast horizon. A wide interval indicates greater uncertainty; it does not guarantee the future VIX will fall inside the range."/></span><span>{fmt(selected?.interval_lo)} – {fmt(selected?.interval_hi)}</span></div><div className="vol-forecast-track"><div className="vol-forecast-track-range"/>{validInterval&&<span className="vol-forecast-marker" style={{left:marker+'%'}}/>}</div><div className="vol-forecast-ci-values"><span><b>{fmt(selected?.interval_lo)}</b> lower</span><span>90% CI</span><span><b>{fmt(selected?.interval_hi)}</b> upper</span></div></div>
  </div>
  {fetching&&<p className="vol-forecast-loading" role="status">Updating additional model horizons…</p>}
  {predict?.error&&!selected&&<p className="vol-forecast-loading" role="status">Forecast temporarily unavailable. Please try refreshing.</p>}
 </section>
}
