"""End-to-end audit of repaired forecast data; refuses unverifiable values."""
import argparse
import json
import numpy as np
import pandas as pd

MACRO=("fedfunds","cpi","unrate","gs10","indpro","m2sl")

def audit(source, repaired):
    original=pd.read_csv(source,index_col=0,parse_dates=True).sort_index()
    df=pd.read_csv(repaired,index_col=0,parse_dates=True).sort_index()
    if not original.index.equals(df.index) or df.index.has_duplicates:
        raise ValueError("Repair changed dates or duplicated rows")
    if not np.allclose(original["vix"],df["vix"],equal_nan=True):
        raise ValueError("Repair unexpectedly modified VIX")
    recent=df.tail(60)
    if recent[list(MACRO)].isna().any().any():
        raise ValueError("Recent FRED values are missing")
    if "sentiment_observed" not in df:
        raise ValueError("Missing sentiment provenance")
    observed=df["sentiment_observed"].astype(str).str.lower().eq("true")
    if df.loc[~observed,"sentiment"].notna().any():
        raise ValueError("Unverified sentiment still represented as observed")
    if df.loc[observed,"sentiment"].isna().any():
        raise ValueError("Observed sentiment lacks numeric value")
    report={"rows":len(df),"recent_macro_differing_rows":{
        c:int((~np.isclose(original.loc[recent.index,c],recent[c],equal_nan=True)).sum()) for c in MACRO},
        "sentiment_observed_rows":int(observed.sum()),
        "sentiment_missing_rows":int((~observed).sum()),
        "recent_60_sentiment_observed":int(observed.tail(60).sum()),
        "vix_preserved":True}
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="data/processed/master_df.csv")
    p.add_argument("--repaired",default="forecast_data_repair_preview.csv")
    p.add_argument("--output",default="forecast_data_integrity.json")
    a=p.parse_args()
    with open(a.output,"w") as f:
        json.dump(audit(a.source,a.repaired),f,indent=2)
