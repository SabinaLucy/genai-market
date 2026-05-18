import axios from 'axios'

const BASE = 'https://sabinalucy-volarix.hf.space'

const api = axios.create({ baseURL: BASE, timeout: 20000 })

export const fetchLatest   = ()           => api.get('/latest')
export const fetchPredict  = (horizon)    => api.post('/predict', { horizon })
export const fetchAnalogues= ()           => api.get('/analogues')
export const fetchShap     = ()           => api.get('/shap')
export const fetchBacktest = ()           => api.get('/backtest')
export const fetchBulletin = ()           => api.get('/bulletin')
export const fetchAsk      = (question, bulletin_context) =>
  api.post('/ask', { question, bulletin_context })
