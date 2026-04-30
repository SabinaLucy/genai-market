from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path


ALPHA_DEFAULT   = 0.10
HORIZONS        = [1, 5, 10]
HORIZON_NAMES   = ["t+1 (next day)", "t+5 (next week)", "t+10 (next fortnight)"]


def calibrate_conformal(
    val_preds_vix: np.ndarray,
    val_truths_vix: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
) -> dict[int, dict]:
    """
    Fit split conformal quantiles for each forecast horizon.

    Calibration uses the finite-sample corrected quantile level
    ceil((1 - alpha) * (n + 1)) / n to maintain the marginal coverage guarantee.

    Parameters
    ----------
    val_preds_vix   : (N, 3) validation predictions in VIX space
    val_truths_vix  : (N, 3) validation ground truth in VIX space
    alpha           : miscoverage level (default 0.10 -> 90% coverage)

    Returns
    -------
    dict mapping horizon -> {'hat_q': float, 'alpha': float}
    """
    models = {}
    n = val_preds_vix.shape[0]

    for h_idx, horizon in enumerate(HORIZONS):
        residuals = np.abs(val_truths_vix[:, h_idx] - val_preds_vix[:, h_idx])
        q_level   = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        hat_q     = float(np.quantile(residuals, q_level))
        models[horizon] = {"hat_q": hat_q, "alpha": alpha}

    return models


def predict_intervals(
    test_preds_vix: np.ndarray,
    conformal_models: dict[int, dict],
) -> dict[int, dict]:
    """
    Generate conformal prediction intervals for all horizons.

    Parameters
    ----------
    test_preds_vix  : (N, 3) test predictions in VIX space
    conformal_models: output of calibrate_conformal()

    Returns
    -------
    dict mapping horizon -> {lower, upper, preds, coverage, mean_width, awp}
    """
    results = {}

    for h_idx, horizon in enumerate(HORIZONS):
        hat_q  = conformal_models[horizon]["hat_q"]
        preds  = test_preds_vix[:, h_idx]
        lower  = preds - hat_q
        upper  = preds + hat_q
        results[horizon] = {
            "lower"     : lower,
            "upper"     : upper,
            "preds"     : preds,
            "hat_q"     : hat_q,
            "mean_width": float((upper - lower).mean()),
        }

    return results


def evaluate_coverage(
    intervals: dict[int, dict],
    test_truths_vix: np.ndarray,
) -> dict[int, dict]:
    """
    Compute coverage, AWP, and per-point coverage mask for each horizon.

    Parameters
    ----------
    intervals       : output of predict_intervals()
    test_truths_vix : (N, 3) test ground truth in VIX space

    Returns
    -------
    intervals dict updated in-place with coverage, covered, awp, truths
    """
    for h_idx, horizon in enumerate(HORIZONS):
        res     = intervals[horizon]
        truths  = test_truths_vix[:, h_idx]
        covered = (truths >= res["lower"]) & (truths <= res["upper"])

        coverage = float(covered.mean())
        awp      = res["mean_width"] / max(coverage, 1e-6)

        res.update({
            "truths"  : truths,
            "covered" : covered,
            "coverage": coverage,
            "awp"     : awp,
        })

    return intervals


def coverage_by_regime(
    intervals: dict[int, dict],
    regime_labels: np.ndarray,
    horizon: int = 5,
    regime_order: list[str] = None,
) -> dict[str, dict]:
    """
    Compute per-regime interval width and coverage for a single horizon.

    Parameters
    ----------
    intervals     : output of evaluate_coverage()
    regime_labels : (N,) array of regime strings aligned to the test window
    horizon       : which horizon to analyse (default 5)
    regime_order  : list of regime labels to iterate over

    Returns
    -------
    dict mapping regime -> {n, mean_width, median_width, coverage}
    """
    if regime_order is None:
        regime_order = ["LOW", "ELEVATED", "CRISIS"]

    res    = intervals[horizon]
    widths = res["upper"] - res["lower"]
    stats  = {}

    for regime in regime_order:
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        stats[regime] = {
            "n"           : int(mask.sum()),
            "mean_width"  : float(widths[mask].mean()),
            "median_width": float(np.median(widths[mask])),
            "coverage"    : float(res["covered"][mask].mean()),
        }

    return stats


def rolling_coverage(
    intervals: dict[int, dict],
    horizon: int = 5,
    window: int = 60,
    alarm_threshold: float = 0.80,
) -> dict:
    """
    Compute rolling empirical coverage and interval width over the test period.

    Parameters
    ----------
    intervals        : output of evaluate_coverage()
    horizon          : which horizon to analyse (default 5)
    window           : rolling window in trading days (default 60)
    alarm_threshold  : coverage level below which an alarm is triggered

    Returns
    -------
    dict with rolling_coverage, rolling_width, n_alarms, pct_alarms
    """
    import pandas as pd

    res         = intervals[horizon]
    cov_series  = pd.Series(res["covered"].astype(float))
    wid_series  = pd.Series(res["upper"] - res["lower"])
    roll_cov    = cov_series.rolling(window).mean().values * 100
    roll_wid    = wid_series.rolling(window).mean().values
    valid       = roll_cov[window:]
    n_alarms    = int((valid < alarm_threshold * 100).sum())
    pct_alarms  = n_alarms / max(len(valid), 1) * 100

    return {
        "rolling_coverage" : roll_cov,
        "rolling_width"    : roll_wid,
        "n_alarms"         : n_alarms,
        "pct_alarms"       : pct_alarms,
        "alarm_threshold"  : alarm_threshold,
        "window"           : window,
    }


def save_conformal(conformal_models: dict, path: str, alpha: float = ALPHA_DEFAULT) -> None:
    joblib.dump({"conformal_models": conformal_models, "alpha": alpha}, path)


def load_conformal(path: str) -> tuple[dict, float]:
    bundle = joblib.load(path)
    return bundle["conformal_models"], bundle["alpha"]


def get_interval_for_date(
    date,
    test_dates,
    intervals: dict[int, dict],
    horizon: int = 5,
) -> dict:
    """
    Retrieve the conformal interval for a specific date.

    Returns
    -------
    dict with lower, upper, pred, covered, hat_q
    """
    import pandas as pd

    ts  = pd.Timestamp(date)
    idx = int(np.searchsorted(test_dates, ts))
    res = intervals[horizon]

    return {
        "date"   : ts.date(),
        "lower"  : float(res["lower"][idx]),
        "upper"  : float(res["upper"][idx]),
        "pred"   : float(res["preds"][idx]),
        "covered": bool(res["covered"][idx]),
        "hat_q"  : float(res["hat_q"]),
    }
