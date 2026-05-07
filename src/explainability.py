from __future__ import annotations

import numpy  as np
import pandas as pd
import joblib


REGIME_ORDER = ["LOW", "ELEVATED", "CRISIS"]
GRANGER_LAGS = [1, 5, 10, 21]


def infer_direction(feature_name: str, feature_value: float, feature_mean: float) -> str:
    """Return human-readable direction of a feature relative to its historical mean."""
    delta = feature_value - feature_mean
    if abs(delta) < 0.05 * abs(feature_mean + 1e-8):
        return "neutral"
    if "sentiment" in feature_name:
        return "negative" if feature_value < 0 else "positive"
    return "elevated" if delta > 0 else "depressed"


def load_explainability_cache(cache_path: str) -> dict:
    """Load pre-computed SHAP and permutation importance arrays from disk."""
    return joblib.load(cache_path)


def get_top_drivers(
    feature_vector : np.ndarray,
    current_regime : str,
    horizon        : int  = 5,
    top_n          : int  = 3,
    cache          : dict = None,
    le             = None,
) -> dict:
    """
    Return the top feature drivers for a given prediction context.

    Combines XGBoost TreeExplainer SHAP (regime-specific) and
    LSTM permutation importance (horizon-specific).

    Parameters
    ----------
    feature_vector : (n_features,) array for the current date
    current_regime : 'LOW' | 'ELEVATED' | 'CRISIS'
    horizon        : forecast horizon in trading days — 1, 5, or 10
    top_n          : number of drivers to return
    cache          : output of load_explainability_cache()
    le             : fitted LabelEncoder from regime_classifier.pkl

    Returns
    -------
    dict with keys: top_drivers, horizon, current_regime, stability_rho, granger_sig
    """
    if cache is None:
        raise ValueError("cache must be provided — load with load_explainability_cache()")

    feature_cols     = cache["feature_cols"]
    clf_feature_cols = cache["clf_feature_cols"]
    xgb_shap         = cache["xgb_shap_values"]
    lstm_imp         = cache["lstm_shap_per_horizon"]
    feat_means       = cache["feat_means"]
    granger_df       = cache.get("granger_df")
    stability_rho    = cache.get("stability_rho")

    drivers = []

    #  XGBoost SHAP — regime-specific
    cls_list     = list(le.classes_) if le is not None else REGIME_ORDER
    cls_i        = cls_list.index(current_regime) if current_regime in cls_list else 0
    mean_abs_xgb = np.abs(xgb_shap[cls_i]).mean(axis=0)
    top_xgb      = np.argsort(mean_abs_xgb)[::-1][:top_n]

    for fi in top_xgb:
        fname = clf_feature_cols[fi]
        fval  = feature_vector[feature_cols.index(fname)] if fname in feature_cols else 0.0
        fmean = feat_means.get(fname, 0.0)
        drivers.append({
            "feature"   : fname,
            "direction" : infer_direction(fname, fval, fmean),
            "shap_value": round(float(mean_abs_xgb[fi]), 4),
            "source"    : "xgb",
        })

    #  LSTM permutation importance — horizon-specific
    if horizon in lstm_imp:
        mean_abs_lstm = np.abs(lstm_imp[horizon]).mean(axis=0)
        existing      = {d["feature"] for d in drivers}
        top_lstm      = [i for i in np.argsort(mean_abs_lstm)[::-1]
                         if feature_cols[i] not in existing][:max(0, top_n - len(drivers))]
        for fi in top_lstm:
            fname = feature_cols[fi]
            fval  = feature_vector[fi]
            fmean = feat_means.get(fname, 0.0)
            drivers.append({
                "feature"   : fname,
                "direction" : infer_direction(fname, fval, fmean),
                "shap_value": round(float(mean_abs_lstm[fi]), 6),
                "source"    : "lstm_perm",
            })

    #  Granger-significant variables
    granger_sig = []
    if granger_df is not None:
        for _, grow in granger_df.iterrows():
            for lag in GRANGER_LAGS:
                p = grow.get(f"lag_{lag}", float("nan"))
                if not np.isnan(p) and p < 0.05:
                    granger_sig.append(grow["variable"])
                    break

    return {
        "top_drivers"    : drivers[:top_n],
        "horizon"        : horizon,
        "current_regime" : current_regime,
        "stability_rho"  : round(float(stability_rho), 4) if stability_rho else None,
        "granger_sig"    : list(set(granger_sig)),
    }