import {useEffect,useId,useRef,useState} from 'react'
import {createPortal} from 'react-dom'
export default function InfoTooltip({title='What is this?',text}){
 const [open,setOpen]=useState(false),[pos,setPos]=useState({top:0,left:0});const ref=useRef(null),id=useId()
 useEffect(()=>{if(!open)return;const close=e=>{if(e.key==='Escape')setOpen(false)};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close)},[open])
 const toggle=()=>{if(!open&&ref.current){const r=ref.current.getBoundingClientRect();setPos({top:Math.min(window.innerHeight-210,Math.max(12,r.bottom+12)),left:Math.min(window.innerWidth-290,Math.max(12,r.left-100))})}setOpen(x=>!x)}
 return <><button type="button" className="vol-info" ref={ref} onClick={toggle} aria-expanded={open} aria-controls={id} aria-label={title}>ⓘ</button>{open&&createPortal(<><div className="vol-info-backdrop" onClick={()=>setOpen(false)}/><div id={id} className="vol-info-pop" role="dialog" aria-label={title} style={pos}><div className="vol-info-title"><span>ⓘ</span>{title}<button onClick={()=>setOpen(false)} aria-label="Close information">×</button></div><p>{text}</p></div></>,document.body)}</>
}