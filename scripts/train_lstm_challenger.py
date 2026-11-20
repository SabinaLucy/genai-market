"""Train a corrected VIX-only LSTM challenger; never write production weights.

Uses identical chronological origin splits to the Ridge benchmark, but purges
origins whose target extends beyond a split boundary. Fits scaling on training
windows only. Outputs test MAE and a temporary checkpoint artifact.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.modeling import LSTMModel

HORIZONS = (1, 5, 10)
SEQ = 60

def run(csv, output, checkpoint, epochs=35, predictions="tmp/lstm_holdout_predictions.npz"):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    frame = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates")
    vix = pd.to_numeric(frame["vix"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(vix).all() or np.any(vix <= 0):
        raise ValueError("Invalid VIX")
    logv = np.log(vix)
    # Origin i is the last observed day, target i+h.
    origins = np.arange(SEQ - 1, len(vix) - max(HORIZONS))
    train_end = pd.Timestamp("2022-12-31")
    val_end = pd.Timestamp("2024-12-31")
    dates = frame.index
    train = np.array([dates[i+10] <= train_end for i in origins])
    val = np.array([dates[i] > train_end and dates[i+10] <= val_end for i in origins])
    test = np.array([dates[i] > val_end for i in origins])
    if min(train.sum(), val.sum(), test.sum()) < 100:
        raise ValueError("Insufficient samples")
    # Normalize each sequence relative to its last observed VIX.
    X = np.stack([logv[i-SEQ+1:i+1] - logv[i] for i in origins]).astype("float32")[...,None]
    Y = np.stack([logv[origins+h] - logv[origins] for h in HORIZONS], axis=1).astype("float32")
    scale = float(np.std(X[train]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training scale")
    X /= scale
    device = torch.device("cpu")
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, dropout=0, n_horizons=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    xtr, ytr = torch.from_numpy(X[train]), torch.from_numpy(Y[train])
    xv, yv = torch.from_numpy(X[val]), torch.from_numpy(Y[val])
    best, best_loss, wait = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for batch in perm.split(128):
            optimizer.zero_grad()
            pred, _ = model(xtr[batch])
            loss = loss_fn(pred, ytr[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, _ = model(xv)
            val_loss = float(loss_fn(vp, yv))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best = {k: value.detach().clone() for k,value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 7:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        validation_prediction, _ = model(torch.from_numpy(X[val]))
        prediction, _ = model(torch.from_numpy(X[test]))
    pred = vix[origins[test],None] * np.exp(prediction.numpy())
    truth = np.stack([vix[origins[test]+h] for h in HORIZONS],axis=1)
    baseline = vix[origins[test]]
    report = {"train_origins":int(train.sum()),"validation_origins":int(val.sum()),
              "test_origins":int(test.sum()),"best_val_mse":best_loss,
              "epochs_completed":epoch+1,"horizons":{}}
    for j,h in enumerate(HORIZONS):
        m = float(np.mean(np.abs(pred[:,j]-truth[:,j])))
        b = float(np.mean(np.abs(baseline-truth[:,j])))
        report["horizons"][str(h)] = {"lstm_mae":m,"persistence_mae":b,
            "improvement_pct":100*(b-m)/b}
    np.savez(predictions, val_origin_dates=dates[origins[val]].astype(str).to_numpy(),
        test_origin_dates=dates[origins[test]].astype(str).to_numpy(),
        val_predictions=vix[origins[val],None]*np.exp(validation_prediction.numpy()),
        test_predictions=pred, val_truth=np.stack([vix[origins[val]+h] for h in HORIZONS],axis=1),
        test_truth=truth, val_vix=vix[origins[val]], test_vix=baseline)
    Path(output).write_text(json.dumps(report,indent=2)+"\n")
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best,"scale":scale,"features":"relative_log_vix",
                "horizons":HORIZONS,"sequence_length":SEQ},checkpoint)
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/processed/master_df.csv")
    p.add_argument("--output",default="lstm_challenger_report.json")
    p.add_argument("--checkpoint",default="tmp/lstm_challenger.pt")
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--predictions",default="tmp/lstm_holdout_predictions.npz")
    a=p.parse_args()
    run(a.csv,a.output,a.checkpoint,a.epochs,a.predictions)
