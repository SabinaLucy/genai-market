"""Leakage-safe VIX-only forecast challenger; does not touch production.

The legacy macro/sentiment history is unreliable. This challenger deliberately
uses only observed VIX closes, trains on earlier origins, selects regularization
on a later validation period, and evaluates on a final chronological holdout.
A forecast is not eligible for deployment unless it beats persistence on test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HORIZONS = (1, 5, 10)
LAGS = (0, 1, 2, 4, 9, 19, 39, 59)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

def make_examples(frame):
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates")
    vix = pd.to_numeric(frame["vix"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vix).all() or (vix <= 0).any():
        raise ValueError("Invalid VIX")
    origins = np.arange(max(LAGS), len(vix) - max(HORIZONS))
    # Use only values known at each origin; predict changes from that day's VIX.
    X = np.array([[np.log(vix[i - lag] / vix[i]) for lag in LAGS[1:]]
                  + [np.log(vix[i]), np.log(np.mean(vix[i-4:i+1]) / vix[i]),
                     np.log(np.mean(vix[i-19:i+1]) / vix[i]),
                     np.std(np.log(vix[i-19:i+1]), ddof=1)]
                  for i in origins], dtype=float)
    Y = np.stack([np.log(vix[origins + h] / vix[origins]) for h in HORIZONS], axis=1)
    return X, Y, origins, vix, frame.index

def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))

def run(df, train_end="2022-12-31", val_end="2024-12-31"):
    X, Y, origins, vix, dates = make_examples(df)
    train = dates[origins] <= pd.Timestamp(train_end)
    val = (dates[origins] > pd.Timestamp(train_end)) & (dates[origins] <= pd.Timestamp(val_end))
    test = dates[origins] > pd.Timestamp(val_end)
    if min(train.sum(), val.sum(), test.sum()) < 100:
        raise ValueError("Each chronological split needs at least 100 examples")
    report = {"splits": {"train": int(train.sum()), "validation": int(val.sum()),
                         "test": int(test.sum())}, "horizons": {}}
    for j, h in enumerate(HORIZONS):
        best_alpha, best_val_mae = None, float("inf")
        for alpha in ALPHAS:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(X[train], Y[train, j])
            pred = vix[origins[val]] * np.exp(candidate.predict(X[val]))
            truth = vix[origins[val] + h]
            score = mae(pred, truth)
            if score < best_val_mae:
                best_alpha, best_val_mae = alpha, score
        # Fit on train+validation only after hyperparameter selection.
        final = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        final.fit(X[train | val], Y[train | val, j])
        pred = vix[origins[test]] * np.exp(final.predict(X[test]))
        truth = vix[origins[test] + h]
        baseline = vix[origins[test]]
        model_mae, baseline_mae = mae(pred, truth), mae(baseline, truth)
        # Diagnose whether gains survive distinct calendar years, rather than
        # hiding weak regimes behind a single pooled error statistic.
        yearly = {}
        test_years = dates[origins[test]].year
        for year in sorted(set(test_years)):
            mask = test_years == year
            if mask.sum() < 20:
                continue
            ym = mae(pred[mask], truth[mask])
            yb = mae(baseline[mask], truth[mask])
            yearly[str(year)] = {
                "samples": int(mask.sum()),
                "challenger_mae": ym,
                "persistence_mae": yb,
                "improvement_pct": 100 * (yb - ym) / yb if yb else None,
            }
        report["horizons"][str(h)] = {
            "yearly_breakdown": yearly,
            "alpha": best_alpha, "validation_mae": best_val_mae,
            "test_samples": int(test.sum()), "challenger_mae": model_mae,
            "persistence_mae": baseline_mae,
            "improvement_pct": 100 * (baseline_mae - model_mae) / baseline_mae,
            "eligible_to_replace_baseline": bool(model_mae < baseline_mae),
        }
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/master_df.csv")
    parser.add_argument("--output", default="vix_challenger_report.json")
    args = parser.parse_args()
    report = run(pd.read_csv(args.csv, index_col=0, parse_dates=True))
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
