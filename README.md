<div align="center">

# Volarix
### GenAI Financial Market Stress Intelligence System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-volarix--rouge.vercel.app-6d28d9?style=for-the-badge&logo=vercel)](https://volarix-rouge.vercel.app)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-fbbf24?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/SabinaLucy/volarix)
[![API Docs](https://img.shields.io/badge/FastAPI-Docs-06b6d4?style=for-the-badge&logo=fastapi)](https://sabinalucy-volarix.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-genai--market-white?style=for-the-badge&logo=github)](https://github.com/SabinaLucy/genai-market)

![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=black)
![Anthropic](https://img.shields.io/badge/Claude-Anthropic-6d28d9?logo=anthropic&logoColor=white)

*A production-grade ML system that forecasts market volatility, classifies stress regimes, and generates AI-powered financial intelligence - updated automatically every trading day.*

</div>

---

## What is Volarix?

Volarix is an end-to-end machine learning system built to forecast the **CBOE Volatility Index (VIX)** using deep learning, macroeconomic data, and natural language processing. It combines quantitative finance with generative AI to produce actionable market intelligence - live, every day.

Every number on the dashboard comes from a real market feed, a trained model, or a live API call. Nothing is simulated.

**[→ Try it live](https://volarix-rouge.vercel.app)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  GitHub Actions (6:30 AM ET)            │
│  yfinance → VIX  │  FRED API → Macro  │  NewsAPI → NLP  │
│                  ↓                                      │
│            FinBERT Sentiment Scoring                    │
│                  ↓                                      │
│         master_df.csv → HuggingFace Space               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              HuggingFace Space (FastAPI)                │
│                                                         │
│  LSTM Network    →  VIX Forecast (1d / 5d / 10d)       │
│  XGBoost + SHAP  →  Regime Classification + Drivers    │
│  Conformal Pred  →  90% Confidence Intervals           │
│  HMM             →  Regime Probabilities               │
│  Anthropic Claude→  Bulletin + Ask Volarix             │
│  NewsAPI         →  Live Headlines                     │
│  yfinance        →  Live VIX (every request)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Vercel (React 18 + Vite)                   │
│                                                         │
│  Overview    →  Live VIX chart + regime dashboard      │
│  Regime      →  SHAP feature importance                │
│  Bulletin    →  AI-generated market stress memo        │
│  Ask Volarix →  Claude-powered financial Q&A           │
│  Backtest    →  Strategy comparison (SPY data)         │
│  About       →  Architecture + model details           │
└─────────────────────────────────────────────────────────┘
```

---

## Models

### LSTM Neural Network - VIX Forecasting
- Input: 60 trading days of VIX history + 6 FRED macro series + daily sentiment
- Output: 1-day, 5-day, and 10-day VIX forecasts
- Confidence intervals via **conformal prediction** (guaranteed 90% coverage)
- Attention weights for interpretability

### XGBoost Classifier - Regime Detection
- Three regimes: **LOW** (VIX < 20), **ELEVATED** (20–30), **CRISIS** (VIX > 30)
- SHAP values for human-readable feature importance
- Real-time classification on every API request

### Strategy Backtest - SPY 2022–2026
Three strategies tested against real price data:
| Strategy | Description |
|---|---|
| **Buy and Hold** | Baseline - hold SPY throughout |
| **Naive Regime Switch** | Move to cash on any CRISIS signal |
| **Hysteresis** | Requires 2 consecutive CRISIS signals to exit, 3 clear signals to re-enter |

The hysteresis strategy delivered the best risk-adjusted returns by reducing unnecessary trades.

---

## Data Pipeline

Runs automatically every weekday at **6:30 AM ET** via GitHub Actions:

| Source | Data | Frequency |
|---|---|---|
| **yfinance** | Live VIX close | Daily |
| **FRED API** | Fed funds rate, CPI, unemployment, 10Y yield, industrial production, M2 | Daily (monthly releases forward-filled) |
| **NewsAPI** | Top financial headlines | Daily |
| **ProsusAI/FinBERT** | Sentiment scoring on headlines | Daily |

After update → pushes `master_df.csv` to HuggingFace Space → Space restarts → loads fresh data.

---

## Tech Stack

### Backend
- **Python 3.11** - FastAPI, PyTorch, XGBoost, SHAP, scikit-learn
- **Anthropic Claude** - bulletin generation and financial Q&A
- **FinBERT** (`ProsusAI/finbert`) - daily sentiment scoring
- **SlowAPI** - rate limiting (5 requests/hour on AI endpoints)
- **HuggingFace Spaces** - model hosting and API serving

### Frontend
- **React 18** + Vite + Recharts
- **Tailwind CSS** - styling
- **Vercel** - deployment and CDN

### MLOps
- **GitHub Actions** - daily automated data pipeline
- **HuggingFace Hub** - model artifact storage
- **Conformal Prediction (MAPIE)** - distribution-free uncertainty quantification

---

## API Endpoints

Base URL: `https://sabinalucy-volarix.hf.space`

| Endpoint | Method | Description |
|---|---|---|
| `/latest` | GET | Live VIX, regime, sentiment |
| `/predict` | POST | LSTM forecast with confidence intervals |
| `/bulletin` | GET | AI-generated market stress memo |
| `/ask` | POST | Claude-powered financial Q&A |
| `/shap` | GET | Feature importance drivers |
| `/analogues` | GET | Historical market analogues |
| `/backtest` | GET | Strategy comparison results |
| `/headlines` | GET | Live financial news (cached 1hr) |
| `/health` | GET | API health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
genai-market/
├── src/
│   ├── main.py                 # FastAPI app + all endpoints
│   ├── bulletin_generator.py   # Claude prompt engineering
│   ├── modeling.py             # LSTM architecture + inference
│   ├── analogues.py            # Historical analogue search
│   ├── explainability.py       # SHAP feature importance
│   └── backtesting.py          # Strategy backtesting engine
├── scripts/
│   └── update_master_df.py     # Daily data pipeline script
├── .github/workflows/
│   └── update_vix.yml          # GitHub Actions daily automation
├── notebooks/                  # Phase 1-9 development notebooks
├── data/processed/
│   └── master_df.csv           # Feature-engineered dataset
└── frontend/
    ├── src/
    │   ├── pages/              # Overview, Regime, Bulletin, Ask, Backtest, About
    │   ├── components/         # BulletinCard, PageHeader, MobileNav, etc.
    │   ├── api.js              # All HF Space API calls
    │   └── utils.js            # Shared utilities
    └── vite.config.js
```

---

## Local Development

```bash
# Clone
git clone https://github.com/SabinaLucy/genai-market.git
cd genai-market

# Backend (requires conda)
conda create -n genai_market python=3.11
conda activate genai_market
pip install -r requirements-deploy.txt

# Environment variables
cp .env.example .env
# Add: ANTHROPIC_API_KEY, FRED_API_KEY, NEWSAPI_KEY, HF_TOKEN

# Run backend locally
uvicorn src.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## About

Built by **Sabina Bimbi** - MS Data Science student at Michigan Technological University, specialising in AI/ML.

This project was developed across 9 phases covering data ingestion, feature engineering, model training, conformal prediction, backtesting, GenAI integration, and full-stack deployment.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0a66c2?logo=linkedin)](https://linkedin.com/in/sabinalucy)

---

<div align="center">
<sub>Volarix is for research and informational purposes only. Not financial advice.</sub>
</div>