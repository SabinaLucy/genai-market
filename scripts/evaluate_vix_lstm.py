"""Train an experimental VIX-only LSTM with correct horizons and no deployment.

Uses identical origin splits to the Ridge challenger. Validation determines
early stopping; the 2025+ holdout is evaluated only after training.
"""
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.evaluate_vix_challenger import HORIZONS, make_examples, run as ridge_run

torch.set_num_threads(2)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def run(df, epochs=35, patience=7):
    df = df.sort_index()
    X, Y, origins, vix, dates = make_examples(df)
    # The challenger has 60 trading-day context; here each input is the
    # last 60 log VIX observations relative to its current close.
    lookback = 60
    logv = np.log(vix)
    windows = np.stack([logv[i-lookback+1:i+1] for i in origins])
    # Detrend by subtracting current log VIX; retain level as one channel.
    windows = np.stack([windows - logv[origins, None],
                        np.broadcast_to(logv[origins, None], windows.shape)], axis=-1)
    train = dates[origins] <= pd.Timestamp("2022-12-31")
    val = (dates[origins] > pd.Timestamp("2022-12-31")) & (dates[origins] <= pd.Timestamp("2024-12-31"))
    test = dates[origins] > pd.Timestamp("2024-12-31")
    if min(train.sum(), val.sum(), test.sum()) < 100:
        raise ValueError("Insufficient chronological split")
    # Scale from training origins only.
    mu = windows[train].reshape(-1, 2).mean(axis=0)
    sigma = windows[train].reshape(-1, 2).std(axis=0).clip(min=1e-6)
    windows = ((windows - mu) / sigma).astype("float32")
    y = Y.astype("float32")
    model = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
    head = nn.Linear(32, 3)
    params = list(model.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()
    xtrain, ytrain = torch.from_numpy(windows[train]), torch.from_numpy(y[train])
    xval, yval = torch.from_numpy(windows[val]), torch.from_numpy(y[val])
    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        model.train(); head.train()
        order = torch.randperm(len(xtrain))
        for batch in order.split(128):
            optim.zero_grad()
            output, _ = model(xtrain[batch])
            pred = head(output[:, -1])
            loss = loss_fn(pred, ytrain[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
        model.eval(); head.eval()
        with torch.no_grad():
            output, _ = model(xval)
            val_loss = float(loss_fn(head(output[:, -1]), yval))
        if val_loss < best_loss - 1e-6:
            best_loss, no_improve = val_loss, 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                          {k: v.detach().clone() for k, v in head.state_dict().items()})
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    model.eval(); head.eval()
    with torch.no_grad():
        output, _ = model(torch.from_numpy(windows[test]))
        pred_return = head(output[:, -1]).numpy()
    baseline = vix[origins[test]]
    result = {"training_epochs": epoch + 1, "validation_loss": best_loss,
              "test_samples": int(test.sum()), "horizons": {}}
    for j, h in enumerate(HORIZONS):
        truth = vix[origins[test] + h]
        prediction = baseline * np.exp(pred_return[:, j])
        model_mae = float(np.mean(np.abs(prediction - truth)))
        base_mae = float(np.mean(np.abs(baseline - truth)))
        result["horizons"][str(h)] = {
            "lstm_mae": model_mae, "persistence_mae": base_mae,
            "improvement_pct": 100 * (base_mae - model_mae) / base_mae,
        }
    result["ridge_reference"] = ridge_run(df)["horizons"]
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/master_df.csv")
    p.add_argument("--output", default="vix_lstm_experiment.json")
    a = p.parse_args()
    result = run(pd.read_csv(a.csv, index_col=0, parse_dates=True))
    print(json.dumps(result, indent=2))
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
