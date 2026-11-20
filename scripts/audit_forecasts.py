"""Audit all three Volarix forecast horizons without modifying production.

Usage:
    python scripts/audit_forecasts.py --csv data/processed/master_df.csv
    python scripts/audit_forecasts.py --csv data/processed/master_df.csv --weights models/lstm_model.pt

The optional weights file is the original LSTM checkpoint. This script does not
download private artifacts or retrain/deploy models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10)
SEQUENCE_LENGTH = 60
FEATURE_COLS = (
    "vix_lag1", "vix_lag5", "vix_lag21", "vix_roll_mean5",
    "vix_roll_std21", "fedfunds", "cpi", "unrate",
    "gs10", "indpro", "m2sl", "sentiment",
)


def audit_data(df: pd.DataFrame) -> dict:
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if len(df) < SEQUENCE_LENGTH + max(HORIZONS) + 2:
        raise ValueError("Not enough history for a 60-day window and 10-day targets")
    missing = (set(FEATURE_COLS) | {"vix", "vix_log"}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    numeric = df[list(FEATURE_COLS) + ["vix", "vix_log"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite feature or target values")
    if (numeric["vix"] <= 0).any():
        raise ValueError("VIX must be positive")
    recent = df.tail(SEQUENCE_LENGTH)
    expected = {
        "vix_log": np.log(numeric["vix"]),
        "vix_lag1": numeric["vix"].shift(1),
        "vix_lag5": numeric["vix"].shift(5),
        "vix_lag21": numeric["vix"].shift(21),
        "vix_roll_mean5": numeric["vix"].rolling(5).mean(),
        "vix_roll_std21": numeric["vix"].rolling(21).std(),
    }
    discrepancies = {}
    for col, exp in expected.items():
        observed = numeric.loc[recent.index, col].to_numpy()
        target = exp.loc[recent.index].to_numpy()
        bad = ~np.isclose(observed, target, atol=1e-5, rtol=1e-5)
        discrepancies[col] = int(bad.sum())
    sentiment_source = recent.get("sentiment_source")
    carried = int((sentiment_source == "forward_fill").sum()) if sentiment_source is not None else None
    return {
        "rows": len(df),
        "latest_date": str(df.index[-1].date()),
        "latest_vix": float(numeric["vix"].iloc[-1]),
        "recent_derived_mismatches": discrepancies,
        "recent_forward_filled_sentiment_rows": carried,
        "recent_macro_unique_values": {
            col: int(recent[col].nunique()) for col in
            ("fedfunds", "cpi", "unrate", "gs10", "indpro", "m2sl")
        },
    }


def evaluate(df: pd.DataFrame, model, device: str, test_start: str) -> dict:
    import torch

    # At origin i, use exactly the last 60 observations ending at i.
    # Target for horizon h is row i+h. No future features are included.
    features = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    vix = df["vix"].to_numpy(dtype=np.float64)
    origins = np.arange(SEQUENCE_LENGTH - 1, len(df) - max(HORIZONS))
    origins = origins[df.index[origins] >= pd.Timestamp(test_start)]
    if len(origins) == 0:
        raise ValueError("No held-out samples for requested test period")
    windows = np.stack([features[i - SEQUENCE_LENGTH + 1:i + 1] for i in origins])
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(windows, max(1, int(np.ceil(len(windows) / 128)))):
            out, _ = model(torch.from_numpy(batch).to(device))
            predictions.append(np.exp(out.cpu().numpy()))
    pred = np.concatenate(predictions)
    with torch.no_grad():
        latest_out, _ = model(torch.from_numpy(features[-SEQUENCE_LENGTH:][None]).to(device))
        latest_forecasts = np.exp(latest_out.cpu().numpy()[0])
    result = {}
    for j, h in enumerate(HORIZONS):
        truth = vix[origins + h]
        baseline = vix[origins]
        forecast = pred[:, j]
        result[str(h)] = {
            "samples": int(len(truth)),
            "lstm_mae": float(np.mean(np.abs(forecast - truth))),
            "persistence_mae": float(np.mean(np.abs(baseline - truth))),
            "lstm_rmse": float(np.sqrt(np.mean((forecast - truth) ** 2))),
            "persistence_rmse": float(np.sqrt(np.mean((baseline - truth) ** 2))),
            "lstm_directional_accuracy_pct": float(
                100 * np.mean(np.sign(forecast - baseline) == np.sign(truth - baseline))
            ),
            "latest_forecast": float(latest_forecasts[j]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/master_df.csv")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default="models/lstm_config.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--test-start", default="2023-01-01")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()
    report = {"data_audit": audit_data(df)}
    if args.weights:
        import torch
        from src.modeling import LSTMModel
        config = json.loads(Path(args.config).read_text())
        if tuple(config["feature_cols"]) != FEATURE_COLS:
            raise ValueError("Checkpoint feature order differs from expected inputs")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LSTMModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            n_horizons=config["n_horizons"],
        ).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        report["horizon_evaluation"] = evaluate(df, model, device, args.test_start)
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
