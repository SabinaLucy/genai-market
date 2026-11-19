import axios from 'axios'

const BASE = 'https://sabinalucy-volarix.hf.space'

const api = axios.create({ baseURL: BASE, timeout: 20000 })

export const fetchLatest   = ()           => api.get('/latest')
export const fetchPredict  = (horizon)    => api.post('/predict', { horizon })
export const fetchAnalogues= ()           => api.get('/analogues')
export const fetchShap     = ()           => api.get('/shap')
export const fetchBacktest = ()           => api.get('/backtest')
export const fetchBulletin = ()           => api.get('/bulletin')
export const fetchHeadlines= ()           => api.get('/headlines')
export const fetchAsk = (question, bulletin_context, history = [], response_style = 'auto') =>
  api.post('/ask', { question, bulletin_context, history, response_style })
export const fetchAskUpload = (file, question, bulletin_context, history = [], response_style = 'auto') => {
  const form = new FormData()
  form.append('file', file)
  form.append('question', question)
  form.append('bulletin_context', bulletin_context || '')
  form.append('history', JSON.stringify(history))
  form.append('response_style', response_style)
  return api.post('/ask/upload', form, { timeout: 90000 })
}

export const fetchVixHistory = (range = '30D') => api.get('/vix/history', { params: { range } })
