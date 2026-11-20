"""Build reproducible VIX-only Ridge inference bundle for a staged release.

Uses validated 2025-2026 holdout only for selection. Refit on all available
completed targets for production inference, with no future labels.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from evaluate_vix_challenger import make_examples, HORIZONS, run

def build(csv, output):
    df=pd.read_csv(csv,index_col=0,parse_dates=True).sort_index()
    report=run(df)
    X,Y,origins,vix,dates=make_examples(df)
    models={}
    for j,h in enumerate(HORIZONS):
        alpha=report["horizons"][str(h)]["alpha"]
        # Each horizon trains on every origin with a completed target.
        eligible=origins+h<len(df)
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(X[eligible],Y[eligible,j])
        models[h]=model
    bundle={"models":models,"horizons":list(HORIZONS),
            "last_training_date":str(df.index[-1].date()),
            "feature_spec":"evaluate_vix_challenger.make_examples:v1",
            "validation_report":report}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output)
    restored=joblib.load(output)
    from ridge_inference import latest_ridge_features
    live_features=latest_ridge_features(df)
    for h in HORIZONS:
        pred=float(vix[-1]*np.exp(restored["models"][h].predict(live_features)[0]))
        if not np.isfinite(pred) or pred<=0:
            raise RuntimeError(f"Invalid staged prediction for {h}")
    print(json.dumps({"status":"staged_not_deployed","last_training_date":bundle["last_training_date"],
        "horizons":bundle["horizons"],"holdout_mae":{str(h):report["horizons"][str(h)]["challenger_mae"] for h in HORIZONS}},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/processed/master_df.csv")
    p.add_argument("--output",default="tmp/vix_ridge_staging.joblib")
    a=p.parse_args()
    build(a.csv,a.output)
