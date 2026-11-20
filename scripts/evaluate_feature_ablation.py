"""Leakage-safe feature ablation on historical Volarix data (research only).

Compares VIX-only and VIX + verified FinBERT sentiment. Macro variants are
blocked unless full historical vintage provenance is supplied, rather than
silently training on unverified historical FRED values.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate_vix_challenger import make_examples, HORIZONS, ALPHAS

MACRO=("fedfunds","cpi","unrate","gs10","indpro","m2sl")

def run(df, macro_verified=False):
    X,Y,origins,vix,dates=make_examples(df)
    train=dates[origins+10]<=pd.Timestamp("2022-12-31")
    val=(dates[origins]>pd.Timestamp("2022-12-31"))&(dates[origins+10]<=pd.Timestamp("2024-12-31"))
    test=dates[origins]>pd.Timestamp("2024-12-31")
    source=df["sentiment_source"].fillna("").astype(str).str.lower()
    scores=pd.to_numeric(df["sentiment"],errors="coerce")
    observed=(source.eq("finbert")&scores.notna()).to_numpy()
    # Explicit missingness indicator; zero is a computational placeholder,
    # never interpreted as an observed neutral news day.
    score_feature=np.where(observed,scores.fillna(0).to_numpy(),0.)
    extra=np.stack([score_feature[origins],observed[origins].astype(float)],axis=1)
    result={"splits":{"train":int(train.sum()),"validation":int(val.sum()),"test":int(test.sum())},
            "sentiment":{"observed_total":int(observed.sum()),
                         "observed_train_origins":int(observed[origins[train]].sum()),
                         "observed_validation_origins":int(observed[origins[val]].sum()),
                         "observed_test_origins":int(observed[origins[test]].sum())},
            "variants":{},"macro_note":"Macro ablations require complete point-in-time FRED history; historical legacy values are not certified."}
    variants={"vix_only":X,"vix_plus_verified_sentiment":np.column_stack([X,extra])}
    if macro_verified:
        for c in MACRO:
            if c not in df:
                raise ValueError(f"Missing macro: {c}")
        macro=df[list(MACRO)].apply(pd.to_numeric,errors="coerce").to_numpy()
        # Only enable after externally verifying vintage-aware complete history.
        if not np.isfinite(macro).all():
            raise ValueError("Macro history incomplete")
        variants["vix_plus_macro"]=np.column_stack([X,macro[origins]])
        variants["vix_plus_macro_sentiment"]=np.column_stack([X,macro[origins],extra])
    else:
        result["variants"]["vix_plus_macro"]={"status":"blocked_unverified_historical_FRED"}
        result["variants"]["vix_plus_macro_sentiment"]={"status":"blocked_unverified_historical_FRED"}
    for name,features in variants.items():
        metrics={}
        for j,h in enumerate(HORIZONS):
            best=None
            for alpha in ALPHAS:
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                model.fit(features[train],Y[train,j])
                p=vix[origins[val]]*np.exp(model.predict(features[val]))
                error=float(np.mean(np.abs(p-vix[origins[val]+h])))
                if best is None or error<best[0]:
                    best=(error,alpha)
            model=make_pipeline(StandardScaler(),Ridge(alpha=best[1]))
            model.fit(features[train|val],Y[train|val,j])
            p=vix[origins[test]]*np.exp(model.predict(features[test]))
            truth=vix[origins[test]+h]
            metrics[str(h)]={"validation_mae":best[0],"test_mae":float(np.mean(np.abs(p-truth))),
                "persistence_mae":float(np.mean(np.abs(vix[origins[test]]-truth))),
                "alpha":best[1]}
        result["variants"][name]={"status":"evaluated","horizons":metrics}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/processed/master_df.csv")
    p.add_argument("--output",default="feature_ablation_report.json")
    a=p.parse_args()
    df=pd.read_csv(a.csv,index_col=0,parse_dates=True).sort_index()
    report=run(df)
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
