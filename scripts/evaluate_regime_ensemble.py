"""Evaluate Ridge/LSTM regime-aware ensemble without touching production.

Weights are selected only on 2023-2024 validation origins. 2025+ is untouched
until final scoring. Regime is defined from the VIX known at forecast origin.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from evaluate_vix_challenger import make_examples,HORIZONS,ALPHAS

WEIGHTS=np.linspace(0,1,11) # weight on Ridge
REGIME_THRESHOLD=25.0
MIN_REGIME_SAMPLES=30

def run(csv,lstm_predictions,output):
    df=pd.read_csv(csv,index_col=0,parse_dates=True).sort_index()
    X,Y,origins,vix,dates=make_examples(df)
    train=dates[origins+10]<=pd.Timestamp("2022-12-31")
    val=(dates[origins]>pd.Timestamp("2022-12-31"))&(dates[origins+10]<=pd.Timestamp("2024-12-31"))
    test=dates[origins]>pd.Timestamp("2024-12-31")
    pack=np.load(lstm_predictions)
    for split,mask in (("val",val),("test",test)):
        expected=dates[origins[mask]].astype(str).to_numpy()
        if not np.array_equal(expected,pack[split+"_origin_dates"]):
            raise ValueError("LSTM/Ridge forecast origin dates do not align")
        if not np.allclose(vix[origins[mask]],pack[split+"_vix"]):
            raise ValueError("LSTM/Ridge origin VIX mismatch")
    report={"status":"research_only_not_deployed","regime_threshold_vix":REGIME_THRESHOLD,
            "weights":"fraction assigned to Ridge","horizons":{}}
    for j,h in enumerate(HORIZONS):
        # Choose Ridge regularization on validation only.
        choices=[]
        for alpha in ALPHAS:
            m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X[train],Y[train,j])
            vp=vix[origins[val]]*np.exp(m.predict(X[val]))
            err=float(np.mean(np.abs(vp-pack["val_truth"][:,j])))
            choices.append((err,alpha,vp))
        _,alpha,ridge_val=min(choices,key=lambda z:z[0])
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X[train|val],Y[train|val,j])
        ridge_test=vix[origins[test]]*np.exp(ridge.predict(X[test]))
        lstm_val=pack["val_predictions"][:,j]
        lstm_test=pack["test_predictions"][:,j]
        truth_val=pack["val_truth"][:,j]
        truth_test=pack["test_truth"][:,j]
        regimes_val=pack["val_vix"]>=REGIME_THRESHOLD
        regimes_test=pack["test_vix"]>=REGIME_THRESHOLD
        def score(a,b):return float(np.mean(np.abs(a-b)))
        def best_weight(mask):
            return float(min(WEIGHTS,key=lambda w:score(w*ridge_val[mask]+(1-w)*lstm_val[mask],truth_val[mask])))
        global_w=best_weight(np.ones(len(ridge_val),dtype=bool))
        weights={}
        for key,flag in (("normal",False),("elevated",True)):
            mask=regimes_val==flag
            weights[key]=best_weight(mask) if mask.sum()>=MIN_REGIME_SAMPLES else global_w
        w_test=np.where(regimes_test,weights["elevated"],weights["normal"])
        blended=w_test*ridge_test+(1-w_test)*lstm_test
        ridge_mae=score(ridge_test,truth_test)
        ensemble_mae=score(blended,truth_test)
        persistence_mae=score(pack["test_vix"],truth_test)
        by_regime={}
        for key,flag in (("normal",False),("elevated",True)):
            mask=regimes_test==flag
            by_regime[key]={"samples":int(mask.sum()),"ridge_mae":score(ridge_test[mask],truth_test[mask]) if mask.any() else None,
                "ensemble_mae":score(blended[mask],truth_test[mask]) if mask.any() else None}
        report["horizons"][str(h)]={"ridge_alpha":alpha,"ridge_weights":weights,
            "validation_regime_counts":{"normal":int((~regimes_val).sum()),"elevated":int(regimes_val.sum())},
            "test_samples":int(test.sum()),"ridge_mae":ridge_mae,
            "lstm_mae":score(lstm_test,truth_test),"ensemble_mae":ensemble_mae,
            "persistence_mae":persistence_mae,"by_regime":by_regime,
            "ensemble_beats_ridge":bool(ensemble_mae<ridge_mae)}
    Path(output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/processed/master_df.csv")
    p.add_argument("--lstm-predictions",default="tmp/lstm_holdout_predictions.npz")
    p.add_argument("--output",default="regime_ensemble_report.json")
    a=p.parse_args()
    run(a.csv,a.lstm_predictions,a.output)
