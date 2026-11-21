"""Chronological VIX-only forecasting benchmark. Research only; never deploys.

All candidates share identical forecast origins, horizon-specific labels and
train/validation/test boundaries. Test targets are observed, never imputed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_vix_challenger import HORIZONS, make_examples


def candidate_factories():
    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "har_ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.05,
            min_samples_leaf=25, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = lambda: XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            n_jobs=2, random_state=42)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=200, num_leaves=15, learning_rate=0.04,
            min_child_samples=25, verbosity=-1, n_jobs=2, random_state=42)
    except ImportError:
        pass
    return models


def metrics(actual, predicted):
    err = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


def run(frame, train_end="2022-12-31", val_end="2024-12-31"):
    X, Y, origins, vix, dates = make_examples(frame)
    train = dates[origins + max(HORIZONS)] <= pd.Timestamp(train_end)
    validation = ((dates[origins] > pd.Timestamp(train_end)) &
                  (dates[origins + max(HORIZONS)] <= pd.Timestamp(val_end)))
    test = dates[origins] > pd.Timestamp(val_end)
    if min(train.sum(), validation.sum(), test.sum()) < 100:
        raise ValueError("Insufficient samples in chronological split")
    # HAR-style features: level and short/medium rolling relative VIX means.
    # This is a HAR-inspired benchmark, not a conventional HAR-RV model.
    har_columns = [7, 8, 9, 10]
    factories = candidate_factories()
    result = {"splits": {"train": int(train.sum()),
                         "validation": int(validation.sum()),
                         "test": int(test.sum())},
              "models": {}, "notes": [
                  "Research only. No deployment.",
                  "Model hyperparameters are provisional, not tuned on test.",
                  "Regimes use origin VIX: LOW <20, ELEVATED 20-30, CRISIS >=30.",
                  "2025-2026 is a shared final holdout; no walk-forward refits yet.",
                  "Corrected LSTM requires a separate identical-split run."]}
    for horizon_index, h in enumerate(HORIZONS):
        # Targets for every evaluated origin must be observed by the evaluation cutoff.
        origin_test = origins[test]
        observed = vix[origin_test + h]
        current = vix[origin_test]
        regimes = np.where(current < 20, "LOW",
                           np.where(current < 30, "ELEVATED", "CRISIS"))
        horizon_result = {"persistence": {"overall": metrics(observed, current)}}
        for name, factory in factories.items():
            features = X[:, har_columns] if name == "har_ridge" else X
            # Fit only on train and validation; no test observations used.
            model = factory()
            model.fit(features[train | validation], Y[train | validation, horizon_index])
            forecast = current * np.exp(model.predict(features[test]))
            if not np.isfinite(forecast).all():
                raise ValueError(f"Non-finite predictions from {name}")
            report = {"overall": metrics(observed, forecast),
                      "regimes": {}, "years": {}}
            for regime in ("LOW", "ELEVATED", "CRISIS"):
                mask = regimes == regime
                report["regimes"][regime] = {
                    "samples": int(mask.sum()),
                    **(metrics(observed[mask], forecast[mask]) if mask.any() else {})}
            years = dates[origin_test].year
            for year in sorted(set(years)):
                mask = years == year
                report["years"][str(year)] = {
                    "samples": int(mask.sum()),
                    **metrics(observed[mask], forecast[mask])}
            horizon_result[name] = report
        # Also report baseline by regime and calendar year for fair comparison.
        baseline = horizon_result["persistence"]
        baseline["regimes"] = {
            regime: {"samples": int((regimes == regime).sum()),
                     **(metrics(observed[regimes == regime], current[regimes == regime])
                        if (regimes == regime).any() else {})}
            for regime in ("LOW", "ELEVATED", "CRISIS")}
        years = dates[origin_test].year
        baseline["years"] = {
            str(year): {"samples": int((years == year).sum()),
                        **metrics(observed[years == year], current[years == year])}
            for year in sorted(set(years))}
        result["models"][str(h)] = horizon_result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/master_df.csv")
    parser.add_argument("--output", default="forecast_common_benchmark.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()
    report = run(frame)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "research_only", "splits": report["splits"],
                      "models": list(report["models"]["1"].keys()),
                      "output": args.output}, indent=2))
