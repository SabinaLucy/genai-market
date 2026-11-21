"""Shared latest-date Ridge feature construction, matching research features."""
import numpy as np
import pandas as pd

LAGS=(0,1,2,4,9,19,39,59)

def latest_ridge_features(frame):
    v=pd.to_numeric(frame["vix"],errors="coerce").to_numpy(dtype=float)
    if len(v)<60 or not np.isfinite(v[-60:]).all() or np.any(v[-60:]<=0):
        raise ValueError("Need 60 valid observed VIX closes")
    i=len(v)-1
    return np.array([[np.log(v[i-lag]/v[i]) for lag in LAGS[1:]]
        +[np.log(v[i]),np.log(np.mean(v[i-4:i+1])/v[i]),
          np.log(np.mean(v[i-19:i+1])/v[i]),
          np.std(np.log(v[i-19:i+1]),ddof=1)]],dtype=float)
