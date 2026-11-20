"""Create a non-production training copy with stale sentiment explicitly missing.

Never silently carry stale news sentiment forward as if it were observed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def clean(df):
    df=df.copy()
    if "sentiment_source" not in df:
        raise ValueError("sentiment_source missing: cannot establish provenance")
    source=df["sentiment_source"].fillna("").astype(str).str.lower()
    stale=source.isin(["forward_fill","ffill","carried","stale"])
    df["sentiment_observed"]=~stale & pd.to_numeric(df["sentiment"],errors="coerce").notna()
    df.loc[~df["sentiment_observed"],"sentiment"]=np.nan
    return df

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="macro_repair_preview.csv")
    p.add_argument("--output",default="forecast_data_repair_preview.csv")
    a=p.parse_args()
    if Path(a.csv).resolve()==Path(a.output).resolve():
        raise SystemExit("Refusing to overwrite input")
    df=pd.read_csv(a.csv,index_col=0,parse_dates=True)
    out=clean(df)
    out.to_csv(a.output)
    print(f"Saved {len(out)} rows; {int((~out['sentiment_observed']).sum())} sentiment rows marked missing; source unchanged")
