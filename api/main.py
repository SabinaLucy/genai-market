"""
api/main.py
===========
Volarix FastAPI backend — Phase 9
/bulletin and /ask: max 5 requests per IP per hour
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

#  Import src modules 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling      import LSTMModel, VIXDataset, predict
from src.analogues     import get_analogues, format_analogues_for_bulletin
from src.explainability import load_explainability_cache, get_top_drivers
from src.backtesting   import run_full_backtest
from src.bulletin_generator import (
    generate_bulletin,
    answer_financial_question,
    build_prediction_dict,
)

#  Constants

HF_REPO_ID   = "SabinaLucy/volarix-models"
MODEL_TAG    = os.getenv("VOLARIX_MODEL_TAG", "").strip()   # e.g. "v2" or ""
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_END      = "2021-12-31"

FEATURE_COLS = [
    "vix_lag1", "vix_lag5", "vix_lag21",
    "vix_roll_mean5", "vix_roll_std21",
    "fedfunds", "cpi", "unrate", "gs10", "indpro", "m2sl",
    "sentiment",
]

HORIZONS = [1, 5, 10]

REGIME_META = {
    "LOW"      : {"label": "[STABLE]", "hex": "#16A34A", "tailwind": "bg-green-600"},
    "ELEVATED" : {"label": "[WATCH]",  "hex": "#D97706", "tailwind": "bg-amber-600"},
    "CRISIS"   : {"label": "[ALERT]",  "hex": "#DC2626", "tailwind": "bg-red-600"},
}

#  JSON serialiser — converts numpy / nan to Python / null 

class SafeJSONResponse(JSONResponse):
    """Replaces NaN/Inf with null so the browser never sees invalid JSON."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            default=self._default,
            allow_nan=False,        # raises ValueError on NaN — forces our handler
        ).encode("utf-8")

    @staticmethod
    def _default(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def safe_json(content: dict, status_code: int = 200) -> SafeJSONResponse:
    return SafeJSONResponse(content=content, status_code=status_code)


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


#  Regime helpers 
def regime_payload(regime: str) -> dict:
    """Return regime string + color metadata as a dict ready for JSON."""
    meta = REGIME_META.get(regime, REGIME_META["ELEVATED"])
    return {
        "regime"        : regime,
        "regime_label"  : meta["label"],
        "regime_color"  : meta["hex"],
        "regime_tailwind": meta["tailwind"],
    }


#  HF Hub download helper 
def hub_download(filename: str) -> str:
    """Download a file from HF Hub and return local path. Cached by huggingface_hub."""
    subfolder = MODEL_TAG if MODEL_TAG else None
    return hf_hub_download(
        repo_id   = HF_REPO_ID,
        filename  = filename,
        subfolder = subfolder,
        repo_type = "model",
        token     = os.getenv("HF_TOKEN"),
    )


#  App state (loaded once at startup) 

class AppState:
    lstm_model      : LSTMModel             = None
    lstm_config     : dict                  = None
    regime_clf      : Any                   = None
    regime_le       : Any                   = None
    clf_feature_cols: list[str]             = None
    conformal_data  : dict                  = None
    expl_cache      : dict                  = None
    scaler          : Any                   = None
    scaler_features : list[str]             = None
    hmm_data        : dict                  = None
    master_df       : pd.DataFrame          = None
    test_df         : pd.DataFrame          = None
    backtest_results: dict                  = None
    startup_time    : float                 = None
    startup_ts      : datetime              = None
    latest_vix_date : str                   = None

state = AppState()

#  Rate limiter 

limiter = Limiter(key_func=get_remote_address)

#  Lifespan (startup / shutdown) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    t_total = time.time()
    print("=" * 60)
    print("Volarix API — starting up")
    print(f"Device : {DEVICE}")
    print(f"Tag    : {MODEL_TAG or '(none)'}")
    print("=" * 60)

    #  1. master_df 
    t = time.time()
    state.master_df = pd.read_csv("data/processed/master_df.csv", index_col="date", parse_dates=True)
    train_mask = state.master_df.index <= "2018-12-31"
    val_mask   = (state.master_df.index > "2018-12-31") & (state.master_df.index <= VAL_END)
    test_mask  = state.master_df.index > VAL_END
    state.test_df = state.master_df[test_mask].copy()
    state.latest_vix_date = str(state.master_df.index[-1].date())
    print(f"  master_df        {time.time()-t:.1f}s  ({len(state.master_df)} rows)")

    #  2. scaler 
    t = time.time()
    scaler_path = hub_download("scaler.pkl")
    state.scaler = joblib.load(scaler_path)
    sf_path = hub_download("scaler_features.json")
    with open(sf_path) as f:
        state.scaler_features = json.load(f)
    print(f"  scaler           {time.time()-t:.1f}s")

    #  3. LSTM 
    t = time.time()
    cfg_path = hub_download("lstm_config.json")
    with open(cfg_path) as f:
        state.lstm_config = json.load(f)
    state.lstm_model = LSTMModel(
        input_size  = state.lstm_config["input_size"],
        hidden_size = state.lstm_config["hidden_size"],
        num_layers  = state.lstm_config["num_layers"],
        dropout     = state.lstm_config["dropout"],
        n_horizons  = state.lstm_config["n_horizons"],
    ).to(DEVICE)
    wt_path = hub_download("lstm_model.pt")
    state.lstm_model.load_state_dict(
        torch.load(wt_path, map_location=DEVICE)
    )
    state.lstm_model.eval()
    print(f"  LSTM             {time.time()-t:.1f}s  (best epoch {state.lstm_config.get('best_epoch')})")

    #  4. Regime classifier 
    t = time.time()
    clf_path = hub_download("regime_classifier.pkl")
    clf_bundle = joblib.load(clf_path)
    state.regime_clf        = clf_bundle["clf"]
    state.regime_le         = clf_bundle["le"]
    state.clf_feature_cols  = clf_bundle["clf_feature_cols"]
    print(f"  regime clf       {time.time()-t:.1f}s")

    #  5. Conformal intervals 
    t = time.time()
    conf_path = hub_download("conformal_intervals.pkl")
    state.conformal_data = joblib.load(conf_path)
    print(f"  conformal        {time.time()-t:.1f}s  (alpha={state.conformal_data.get('alpha', 0.1)})")

    #  6. Explainability cache 
    t = time.time()
    expl_path = hub_download("explainability_cache.pkl")
    state.expl_cache = joblib.load(expl_path)
    print(f"  explainability   {time.time()-t:.1f}s")

    # 7. HMM 
    t = time.time()
    hmm_path = hub_download("hmm_regime.pkl")
    state.hmm_data = joblib.load(hmm_path)
    print(f"  HMM              {time.time()-t:.1f}s")

    #  8. Pre-compute backtest 
    t = time.time()
    try:
        regime_signal = pd.Series(
            state.regime_clf.predict(
                state.test_df[state.clf_feature_cols].values
            ),
            index=state.test_df.index,
        )
        # decode integer labels → strings
        regime_signal = regime_signal.map(
            dict(enumerate(state.regime_le.classes_))
        ).fillna("LOW")

        state.backtest_results = run_full_backtest(
            test_df       = state.test_df,
            regime_signal = regime_signal,
            val_end       = VAL_END,
        )
        # Drop DataFrame objects — only keep stats dicts and trade counts
        state.backtest_results = {
            k: v for k, v in state.backtest_results.items()
            if not isinstance(v, pd.DataFrame)
        }
        print(f"  backtest         {time.time()-t:.1f}s  ✓")
    except Exception as exc:
        print(f"  backtest         FAILED ({exc}) — /backtest endpoint will return 503")
        state.backtest_results = None

    #  Done 
    state.startup_time = time.time() - t_total
    state.startup_ts   = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"Startup complete in {state.startup_time:.1f}s")
    print("=" * 60)

    yield   # ← API is live from here until shutdown

    print("Volarix API — shutting down")


#  App 

app = FastAPI(
    title       = "Volarix — Financial Market Stress Intelligence",
    description = "LSTM + GenAI risk bulletin for VIX forecasting",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to Vercel URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


#  Pydantic schemas 

class PredictRequest(BaseModel):
    horizon: int = 5   # 1, 5, or 10

class AskRequest(BaseModel):
    question        : str
    bulletin_context: Optional[str] = None


#  Inference helpers 

def _get_latest_feature_vector() -> tuple[np.ndarray, pd.Timestamp]:
    """Return the feature vector and date for the most recent row in master_df."""
    latest_row  = state.master_df.iloc[-1]
    latest_date = state.master_df.index[-1]
    return latest_row[FEATURE_COLS].values.astype(np.float32), latest_date


def _lstm_predict_latest(horizon: int = 5) -> dict:
    """
    Run a single LSTM forward pass using the last 60 rows of master_df.
    Returns predicted VIX, interval bounds, and attention weights.
    """
    h_idx = HORIZONS.index(horizon)
    seq   = state.master_df[FEATURE_COLS].values[-60:].astype(np.float32)
    x     = torch.tensor(seq).unsqueeze(0).to(DEVICE)   # (1, 60, 12)

    with torch.no_grad():
        out, attn_w = state.lstm_model(x)

    preds_log = out.cpu().numpy()[0]          # (3,)
    attn      = attn_w.cpu().numpy()[0]       # (60,)

    pred_vix = float(np.exp(preds_log[h_idx]))

    # Conformal interval
    conf_model  = state.conformal_data["conformal_models"][horizon]
    half_width  = float(conf_model.quantile_)
    current_log = float(np.log(state.master_df["vix"].iloc[-1]))
    pred_log    = float(preds_log[h_idx])
    lo          = float(np.exp(pred_log - half_width))
    hi          = float(np.exp(pred_log + half_width))

    return {
        "predicted_vix"   : round(pred_vix, 2),
        "interval_lo"     : round(lo, 2),
        "interval_hi"     : round(hi, 2),
        "interval_alpha"  : state.conformal_data.get("alpha", 0.1),
        "confidence_pct"  : round((1 - state.conformal_data.get("alpha", 0.1)) * 100),
        "attn_weights"    : attn.tolist(),
        "preds_log_all"   : preds_log.tolist(),
    }


def _regime_predict_latest() -> dict:
    """Run XGBoost regime classifier on the latest row."""
    feat_vec = state.master_df[state.clf_feature_cols].iloc[-1].values.reshape(1, -1)
    proba    = state.regime_clf.predict_proba(feat_vec)[0]
    classes  = list(state.regime_le.classes_)
    pred_cls = classes[int(np.argmax(proba))]
    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    return {"regime": pred_cls, "regime_probabilities": prob_map}


#  Routes 

# 1. Health 

@app.get("/health", tags=["meta"])
def health():
    """Container health check — used by HF Spaces."""
    uptime = (
        round((datetime.now(timezone.utc) - state.startup_ts).total_seconds())
        if state.startup_ts else None
    )
    return safe_json({
        "status"         : "ok",
        "models_loaded"  : state.lstm_model is not None,
        "latest_vix_date": state.latest_vix_date,
        "uptime_seconds" : uptime,
        "startup_seconds": round(state.startup_time, 1) if state.startup_time else None,
        "device"         : str(DEVICE),
        "model_tag"      : MODEL_TAG or "default",
    })


# 2. Models metadata 
@app.get("/models", tags=["meta"])
def models_meta():
    """Loaded model configuration """
    if state.lstm_config is None:
        raise HTTPException(503, "Models not yet loaded")
    return safe_json({
        "lstm"      : {k: v for k, v in state.lstm_config.items()
                       if k not in ("horizon_weights",)},
        "features"  : FEATURE_COLS,
        "horizons"  : HORIZONS,
        "repo_id"   : HF_REPO_ID,
        "model_tag" : MODEL_TAG or "default",
    })


# 3. Latest 

@app.get("/latest", tags=["data"])
def latest(response: Response):
    """Current VIX value and regime. Cached by browser for 60 seconds."""
    if state.master_df is None:
        raise HTTPException(503, "Data not loaded")

    row         = state.master_df.iloc[-1]
    date_str    = str(state.master_df.index[-1].date())
    current_vix = round(float(row["vix"]), 2)
    regime_str  = str(row.get("regime_label", "ELEVATED"))

    # HTTP caching headers — prevents React from hammering the API
    response.headers["Cache-Control"] = "public, max-age=60"
    response.headers["Last-Modified"] = datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )

    return safe_json({
        "date"       : date_str,
        "vix"        : current_vix,
        "vix_log"    : round(float(np.log(current_vix)), 4),
        **regime_payload(regime_str),
        "sentiment"  : round(float(row.get("sentiment", 0.0)), 4),
    })


# 4. Predict 

@app.post("/predict", tags=["forecast"])
def predict_endpoint(req: PredictRequest):
    """
    Run full LSTM + regime pipeline for a given horizon.
    Returns predicted VIX, conformal interval, regime, and probabilities.
    """
    if req.horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {HORIZONS}")
    if state.lstm_model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        lstm_out   = _lstm_predict_latest(req.horizon)
        regime_out = _regime_predict_latest()

        feat_vec, latest_date = _get_latest_feature_vector()
        current_vix = float(state.master_df["vix"].iloc[-1])

        direction = (
            "up"   if lstm_out["predicted_vix"] > current_vix else
            "down" if lstm_out["predicted_vix"] < current_vix else
            "flat"
        )

        return safe_json(_clean({
            "date"              : str(latest_date.date()),
            "horizon_days"      : req.horizon,
            "current_vix"       : round(current_vix, 2),
            "predicted_vix"     : lstm_out["predicted_vix"],
            "direction"         : direction,
            "interval_lo"       : lstm_out["interval_lo"],
            "interval_hi"       : lstm_out["interval_hi"],
            "confidence_pct"    : lstm_out["confidence_pct"],
            **regime_payload(regime_out["regime"]),
            "regime_probabilities": regime_out["regime_probabilities"],
        }))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {exc}")


# 5. Bulletin 

@app.get("/bulletin", tags=["genai"])
@limiter.limit("5/hour")
def bulletin(request: Request):
    """
    Generate the 6-section GenAI risk bulletin for the latest date.
    Rate limited: 5 requests per IP per hour.
    """
    if state.lstm_model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        horizon   = 5
        lstm_out  = _lstm_predict_latest(horizon)
        reg_out   = _regime_predict_latest()
        feat_vec, latest_date = _get_latest_feature_vector()
        current_vix = float(state.master_df["vix"].iloc[-1])

        attn_w = np.array(lstm_out["attn_weights"])

        # Analogues
        analogues = get_analogues(
            current_vector    = feat_vec,
            historical_df     = state.master_df,
            feature_cols      = FEATURE_COLS,
            top_n             = 3,
            current_date      = latest_date,
            attention_weights = attn_w,
        )
        analogue_str = format_analogues_for_bulletin(analogues, top_n=2)

        # Drivers
        drivers_out = get_top_drivers(
            feature_vector = feat_vec,
            current_regime = reg_out["regime"],
            horizon        = horizon,
            top_n          = 3,
            cache          = state.expl_cache,
            le             = state.regime_le,
        )

        prediction_dict = {
            "date"              : str(latest_date.date()),
            "horizon_days"      : horizon,
            "current_vix"       : round(current_vix, 2),
            "predicted_vix"     : lstm_out["predicted_vix"],
            "interval_lo"       : lstm_out["interval_lo"],
            "interval_hi"       : lstm_out["interval_hi"],
            "confidence_pct"    : lstm_out["confidence_pct"],
            "regime"            : reg_out["regime"],
            "regime_probabilities": reg_out["regime_probabilities"],
            "top_drivers"       : drivers_out["top_drivers"],
            "analogue_str"      : analogue_str,
            "analogues"         : analogues,
            "sentiment"         : round(float(state.master_df["sentiment"].iloc[-1]), 4),
        }

        bulletin_text = generate_bulletin(prediction_dict)

        return safe_json(_clean({
            **prediction_dict,
            **regime_payload(reg_out["regime"]),
            "bulletin"   : bulletin_text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Bulletin generation failed: {exc}")


# 6. Analogues 

@app.get("/analogues", tags=["forecast"])
def analogues():
    """Top 3 historical analogues for current market conditions."""
    if state.master_df is None:
        raise HTTPException(503, "Data not loaded")

    try:
        feat_vec, latest_date = _get_latest_feature_vector()
        lstm_out = _lstm_predict_latest(horizon=5)
        attn_w   = np.array(lstm_out["attn_weights"])

        results = get_analogues(
            current_vector    = feat_vec,
            historical_df     = state.master_df,
            feature_cols      = FEATURE_COLS,
            top_n             = 3,
            current_date      = latest_date,
            attention_weights = attn_w,
        )

        # Attach regime color to each analogue
        enriched = []
        for a in results:
            r = a.get("regime", "ELEVATED")
            enriched.append({**a, **regime_payload(r)})

        return safe_json(_clean({
            "query_date" : str(latest_date.date()),
            "analogues"  : enriched,
        }))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Analogue search failed: {exc}")


# 7. SHAP 

@app.get("/shap", tags=["explainability"])
def shap():
    """Top 5 SHAP feature importances for the latest prediction."""
    if state.expl_cache is None:
        raise HTTPException(503, "Explainability cache not loaded")

    try:
        feat_vec, _     = _get_latest_feature_vector()
        reg_out         = _regime_predict_latest()

        drivers = get_top_drivers(
            feature_vector = feat_vec,
            current_regime = reg_out["regime"],
            horizon        = 5,
            top_n          = 5,
            cache          = state.expl_cache,
            le             = state.regime_le,
        )

        return safe_json(_clean({
            "date"          : state.latest_vix_date,
            **regime_payload(reg_out["regime"]),
            "top_drivers"   : drivers["top_drivers"],
            "stability_rho" : drivers.get("stability_rho"),
            "granger_sig"   : drivers.get("granger_sig", []),
        }))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"SHAP retrieval failed: {exc}")


# 8. Ask 

@app.post("/ask", tags=["genai"])
@limiter.limit("5/hour")
def ask(request: Request, body: AskRequest):
    """
    Financial Q&A with optional bulletin context.
    Rate limited: 5 requests per IP per hour.
    """
    try:
        answer = answer_financial_question(
            question        = body.question,
            bulletin_context= body.bulletin_context,
        )
        return safe_json({"question": body.question, "answer": answer})
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Q&A failed: {exc}")


# 9. Backtest 
@app.get("/backtest", tags=["backtest"])
def backtest():
    """
    Pre-computed backtest results (naive vs hysteresis vs buy-and-hold SPY).
    Computed once at startup from the test period regime signal.
    """
    if state.backtest_results is None:
        raise HTTPException(503, "Backtest results unavailable — check startup logs")

    return safe_json(_clean({
        "test_period_start" : str(state.test_df.index[0].date()),
        "test_period_end"   : str(state.test_df.index[-1].date()),
        "transaction_cost"  : 0.0005,
        "risk_free_rate"    : 0.04,
        **state.backtest_results,
    }))
