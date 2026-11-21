"""Single-run, research-only VIX hybrid selection audit.

All features use VIX observed at or before the origin. No deployment or model
artifact is produced. 2025+ is held out until candidates and blend weights
are selected on 2023-24 validation. Walk-forward years are diagnostics, not
an excuse to tune against the final holdout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 5, 10)
REGIMES = ("LOW", "ELEVATED", "CRISIS")
LAGS = (1, 2, 4, 9, 19, 39, 59)
SEED = 42


def features(frame):
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates")
    v = pd.to_numeric(frame["vix"], errors="coerce").to_numpy(float)
    if not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("Invalid VIX observations")
    logv = np.log(v)
    origins = np.arange(59, len(v) - max(HORIZONS))
    x, extended = [], []
    for i in origins:
        cur = v[i]
        recent = v[i-59:i+1]
        log_recent = logv[i-59:i+1]
        basic = [np.log(v[i-lag]/cur) for lag in LAGS] + [
            np.log(cur), np.log(np.mean(v[i-4:i+1])/cur),
            np.log(np.mean(v[i-19:i+1])/cur),
            np.std(logv[i-19:i+1], ddof=1)]
        # No current-week look-ahead: all windows end at origin i.
        stats = [np.log(np.mean(recent[-n:])/cur) for n in (3, 10, 40, 60)]
        stats += [np.std(np.diff(log_recent[-n:]), ddof=1) for n in (6, 11, 21, 41)]
        stats += [logv[i] - logv[i-n] for n in (1, 3, 5, 10, 20)]
        stats += [float((cur - np.min(recent[-20:])) /
                        max(np.max(recent[-20:]) - np.min(recent[-20:]), 1e-6))]
        x.append(basic)
        extended.append(basic + stats)
    y = np.stack([logv[origins+h] - logv[origins] for h in HORIZONS], axis=1)
    return np.array(x), np.array(extended), y, origins, v, frame.index


def error(y, p):
    return {"mae": float(np.mean(np.abs(y-p))),
            "rmse": float(np.sqrt(np.mean((y-p)**2)))}


def fit_ridge(x, y, mask, alpha, sample_weight=None):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x[mask], y[mask], ridge__sample_weight=(sample_weight[mask] if sample_weight is not None else np.ones(int(mask.sum()))))
    return model


def mean_reversion(v, origins, horizon, speed, target_window):
    # Calibrated parameters are chosen exclusively on validation.
    target = np.array([np.median(v[max(0, i-target_window+1):i+1])
                       for i in origins])
    return target + (v[origins]-target)*np.exp(-speed*horizon)


def regime(v):
    return np.where(v < 20, "LOW", np.where(v < 30, "ELEVATED", "CRISIS"))


def evaluate(y, pred, origin_vix):
    regimes = regime(origin_vix)
    out = {"overall": error(y, pred), "regimes": {}}
    for label in REGIMES:
        mask = regimes == label
        out["regimes"][label] = {"samples": int(mask.sum()),
            **(error(y[mask], pred[mask]) if mask.any() else {})}
    return out


def candidates(x, ext, y, origins, v, train, predict_mask, hidx, alphas,
               rev_speeds, rev_windows):
    h = HORIZONS[hidx]
    origin_vix = v[origins[predict_mask]]
    truth = v[origins[predict_mask]+h]
    forecasts = {"persistence": origin_vix.copy()}
    # Ridge candidates are trained on origin-aligned log changes.
    for alpha in alphas:
        for name, xx in (("ridge", x), ("enhanced_ridge", ext),
                         ("har_style", ext[:, [7,8,9,10,11,12,13,14]])):
            model = fit_ridge(xx, y[:,hidx], train, alpha)
            forecasts[f"{name}_a{alpha}"] = origin_vix * np.exp(
                np.clip(model.predict(xx[predict_mask]), -2, 2))
    # Simple recent-history regression: only the current VIX and recent lags,
    # trained on all historical regimes. Crisis weighting is a separate challenger.
    short = ext[:, [7, 0, 1, 2, 3, 4, 8, 9, 10]]
    origin_regimes = regime(v[origins])
    for alpha in alphas:
        for crisis_weight in (1.0, 2.0, 4.0):
            weights = np.where(origin_regimes == "CRISIS", crisis_weight, 1.0)
            model = fit_ridge(short, y[:,hidx], train, alpha, weights)
            key = f"short_regression_a{alpha}_cw{crisis_weight}"
            forecasts[key] = origin_vix * np.exp(np.clip(
                model.predict(short[predict_mask]), -2, 2))
    for speed in rev_speeds:
        for window in rev_windows:
            forecasts[f"mean_reversion_s{speed}_w{window}"] = mean_reversion(
                v, origins[predict_mask], h, speed, window)
    return truth, origin_vix, forecasts


def run(frame):
    x, ext, y, origins, v, dates = features(frame)
    train = np.asarray(dates[origins+10] <= pd.Timestamp("2022-12-31"))
    val = np.asarray((dates[origins] > pd.Timestamp("2022-12-31")) &
                     (dates[origins+10] <= pd.Timestamp("2024-12-31")))
    test = np.asarray(dates[origins] > pd.Timestamp("2024-12-31"))
    if min(train.sum(), val.sum(), test.sum()) < 100:
        raise ValueError("Insufficient train/validation/test samples")
    report = {"status": "research_only_not_deployable",
        "splits": {k: int(m.sum()) for k,m in
                   (("train",train),("validation",val),("test",test))},
        "selection": {}, "walk_forward": {}, "limitations": [
          "Crisis weighting is a validation-selected challenger, not a guarantee of crisis accuracy.",
          "2025+ is a final untouched holdout for model selection.",
          "Crisis holdout is small; report sample counts and avoid claims of reliability.",
          "HAR-style uses VIX levels and statistics, not realized variance.",
          "Mean reversion target is rolling historical median, never future data.",
          "No prediction intervals are calibrated or production model deployed.",
          "Walk-forward folds are diagnostics; they are not independent of the earlier training years.",
          "Model candidates use only verified VIX, not stale sentiment or mixed-unit macro."]}
    alphas = (0.1, 10.0, 100.0)
    speeds = (0.02, 0.08, 0.2)
    windows = (20, 60, 252)
    for j,h in enumerate(HORIZONS):
        val_truth, val_vix, vp = candidates(
            x,ext,y,origins,v,train,val,j,alphas,speeds,windows)
        # Choose a base candidate by validation MAE; ties favor simpler Ridge.
        ranking = sorted(vp, key=lambda k: (
            error(val_truth,vp[k])["mae"],
            0 if k.startswith("ridge_") else 1, k))
        base = ranking[0]
        best_short = next(k for k in ranking if k.startswith("short_regression_"))
        # A predeclared blend family: validation-only weights, no regime tuning
        # when there are too few validation samples for that regime.
        ridge_keys = [k for k in ranking if k.startswith("ridge_")]
        stat_keys = [k for k in ranking if k.startswith("mean_reversion_") or
                     k.startswith("har_style_")]
        best_ridge = ridge_keys[0]
        best_stat = stat_keys[0]
        blends = {}
        for weight in (0.25, 0.5, 0.75):
            key = f"blend_ridge_{weight}"
            blends[key] = weight*vp[best_ridge] + (1-weight)*vp[best_stat]
        choices = {**vp, **blends}
        selected = min(choices, key=lambda k: error(val_truth,choices[k])["mae"])
        # Refit learned models on train+validation, preserving selected parameters.
        combined = train | val
        test_truth, test_vix, tp = candidates(
            x,ext,y,origins,v,combined,test,j,alphas,speeds,windows)
        for weight in (0.25,0.5,0.75):
            tp[f"blend_ridge_{weight}"] = (
                weight*tp[best_ridge] + (1-weight)*tp[best_stat])
        report["selection"][str(h)] = {
            "selected_on_validation": selected,
            "validation_mae": error(val_truth, choices[selected])["mae"],
            "validation_regime_counts": {
                r: int((regime(val_vix)==r).sum()) for r in REGIMES},
            "best_ridge": best_ridge, "best_statistical": best_stat,
            "best_short_regression": best_short,
            "test": {name: evaluate(test_truth, tp[name], test_vix)
                     for name in ("persistence", best_ridge, best_stat, best_short, selected)
                     if name in tp},
            "test_origin_regime_counts": {
                r: int((regime(test_vix)==r).sum()) for r in REGIMES}}
        # Earlier independent-origin chronological folds. Never choose the
        # final candidate by their scores; this is a robustness diagnostic.
        folds = {}
        for year in (2016, 2018, 2020, 2022, 2023, 2024):
            cutoff = pd.Timestamp(f"{year-1}-12-31")
            fold_train = np.asarray(dates[origins+10] <= cutoff)
            fold_test = np.asarray((dates[origins] >= pd.Timestamp(f"{year}-01-01")) &
                                  (dates[origins+10] <= pd.Timestamp(f"{year}-12-31")))
            if fold_train.sum() < 250 or fold_test.sum() < 50:
                continue
            truth, current, pred = candidates(
                x,ext,y,origins,v,fold_train,fold_test,j,
                alphas,speeds,windows)
            if selected.startswith("blend_"):
                w = float(selected.split("_")[-1])
                pred[selected] = w*pred[best_ridge]+(1-w)*pred[best_stat]
            folds[str(year)] = {
                "samples": int(fold_test.sum()),
                "selected": evaluate(truth,pred[selected],current),
                "short_regression": evaluate(truth,pred[best_short],current),
                "ridge": evaluate(truth,pred[best_ridge],current),
                "persistence": evaluate(truth,pred["persistence"],current)}
        report["walk_forward"][str(h)] = folds
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/master_df.csv")
    p.add_argument("--output", default="forecast_common_benchmark.json")
    args = p.parse_args()
    df = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()
    result = run(df)
    Path(args.output).write_text(json.dumps(result,indent=2)+"
")
    print(json.dumps({"status": result["status"], "splits": result["splits"],
        "selection": {h: {"model": v["selected_on_validation"],
                         "test_mae": v["test"][v["selected_on_validation"]]["overall"]["mae"],
                         "crisis_samples": v["test_origin_regime_counts"]["CRISIS"],
                         "short_regression": v["best_short_regression"],
                         "short_regression_test_mae": v["test"][v["best_short_regression"]]["overall"]["mae"]}
                      for h,v in result["selection"].items()}},indent=2))
