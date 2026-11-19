import {useMemo,useEffect,useRef,useState} from 'react'
import {Area,AreaChart,CartesianGrid,ResponsiveContainer,Tooltip,XAxis,YAxis,ReferenceLine} from 'recharts'
import InfoTooltip from '../InfoTooltip'

const tickDate = d => {const t=new Date(d);return Number.isNaN(t.getTime())?d:t.toLocaleDateString('en-US',{month:'short',day:'numeric'})}
function OriginalTooltip({active,payload,label,illustrative}) {
 if(!active||!payload?.length)return null
 const v=Number(payload[0].value)
 return <div className="vol-vix-tooltip"><span>{tickDate(label)}</span><strong>VIX {v.toFixed(2)}</strong><small>{v>=30?'Crisis':v>=20?'Elevated':'Stable'}{illustrative?' · illustrative trend':''}</small></div>
}
export default function VixTrendChart({data=[],liveVix,loading}){
 // The original chart used a generated 30-day curve anchored to the live VIX.
 // Keep its appearance and behavior when historical API observations are unavailable,
 // but explicitly identify generated points as illustrative rather than real history.
 const illustrative=useMemo(()=>{
  const base=Number(liveVix)
  if(!Number.isFinite(base))return []
  let v=base*.87;const today=new Date();const points=[]
  for(let i=29;i>=0;i--){const d=new Date(today);d.setDate(d.getDate()-i);v=Math.max(10,Math.min(65,v+(Math.random()-.47)*1.2));points.push({date:d.toISOString().slice(0,10),vix:Number(v.toFixed(2))})}
  points[points.length-1].vix=base
  return points
 },[liveVix])
 const chartRef=useRef(null)
 const [dimensions,setDimensions]=useState({width:0,height:0})
 useEffect(()=>{const node=chartRef.current;if(!node)return;const check=()=>{const rect=node.getBoundingClientRect();const width=Math.floor(rect.width),height=Math.floor(rect.height);setDimensions(old=>old.width===width&&old.height===height?old:{width,height})};check();const observer=new ResizeObserver(check);observer.observe(node);return()=>observer.disconnect()},[loading,liveVix,data.length])
 const real=data.length>=2
 const values=real?data:illustrative
 return <section className="vol-panel vol-trend vol-original-trend">
  <div className="vol-panel-heading"><h2>VIX 30-Day Trend <InfoTooltip title="VIX 30-Day Trend" text="VIX tracks expected market volatility. The dashed lines indicate elevated volatility at 20 and crisis at 30. Hover over the chart to inspect values. If historical observations are unavailable, the chart displays an illustrative curve based on the original dashboard rather than actual historical prices."/></h2>
   <div className="vol-vix-legend"><span><i className="vol-vix-line crisis"/>Crisis ≥30</span><span><i className="vol-vix-line elevated"/>Elevated ≥20</span>{liveVix!=null&&<span>Live VIX {Number(liveVix).toFixed(2)}</span>}</div>
  </div>
  {loading&&!values.length?<div className="vol-chart-empty">Loading VIX data…</div>:values.length<2?<div className="vol-chart-empty">VIX market data is currently unavailable.</div>:<>
   <div className="vol-original-chart-wrap" ref={chartRef}>{dimensions.width>20&&dimensions.height>20&&<AreaChart width={dimensions.width} height={dimensions.height} data={values} margin={{top:10,right:10,bottom:0,left:-18}}>
    <defs><linearGradient id="vol-original-orange-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#f59e0b" stopOpacity=".25"/><stop offset="100%" stopColor="#f59e0b" stopOpacity=".02"/></linearGradient></defs>
    <CartesianGrid vertical={false} stroke="rgba(159,153,193,.14)"/>
    <XAxis dataKey="date" tickFormatter={tickDate} axisLine={false} tickLine={false} tick={{fill:'var(--vol-muted)',fontSize:10}} minTickGap={38}/>
    <YAxis domain={[d=>Math.min(19,Math.floor(d-1)),d=>Math.max(31,Math.ceil(d+1))]} axisLine={false} tickLine={false} tick={{fill:'var(--vol-muted)',fontSize:10}}/>
    <Tooltip content={<OriginalTooltip illustrative={!real}/>}/>
    <ReferenceLine y={20} stroke="#f59e0b" strokeDasharray="4 4" strokeOpacity=".55"/>
    <ReferenceLine y={30} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity=".55"/>
    <Area dataKey="vix" type="monotone" stroke="#f59e0b" strokeWidth={2.4} fill="url(#vol-original-orange-fill)" dot={false} activeDot={{r:5,stroke:'#fff',strokeWidth:2,fill:'#f59e0b'}} isAnimationActive={false}/>
   </AreaChart>}</div>
   {!real&&<div className="vol-vix-disclaimer">Illustrative 30-day trend · Current VIX is live</div>}
  </>}
 </section>
}