"""Repair recent FRED features using vintage-aware release dates.

Runs in preview mode by default. Does not alter the source CSV unless --write
is specified. FRED's all-releases endpoint provides realtime_start (the date
a value became available) and observation date, avoiding look-ahead leakage.

Usage:
  FRED_API_KEY=... python scripts/repair_macro_history.py
  FRED_API_KEY=... python scripts/repair_macro_history.py --write --output repaired_master_df.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

SERIES = {
    "fedfunds": "FEDFUNDS", "cpi": "CPIAUCSL", "unrate": "UNRATE",
    "gs10": "GS10", "indpro": "INDPRO", "m2sl": "M2SL",
}

def vintage_asof(releases: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Most recently published observation, as known on each trading date."""
    needed = {"date", "realtime_start", "value"}
    if not needed.issubset(releases.columns):
        raise ValueError(f"FRED releases missing {needed - set(releases.columns)}")
    r = releases.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r["realtime_start"] = pd.to_datetime(r["realtime_start"]).dt.normalize()
    r["value"] = pd.to_numeric(r["value"], errors="coerce")
    r = r.dropna(subset=["date", "realtime_start", "value"])
    r = r.sort_values(["realtime_start", "date"])
    # Each release event updates an observation. At every point in time,
    # the newest observation date known so far is the available macro value.
    latest_by_observation = {}
    event_dates, event_values = [], []
    for publication_day, group in r.groupby("realtime_start", sort=True):
        for row in group.itertuples(index=False):
            latest_by_observation[row.date] = float(row.value)
        latest_obs = max(latest_by_observation)
        event_dates.append(publication_day)
        event_values.append(latest_by_observation[latest_obs])
    if not event_dates:
        raise ValueError("FRED returned no valid vintage events")
    events = pd.DataFrame({"published": event_dates, "value": event_values})
    query = pd.DataFrame({"published": pd.DatetimeIndex(dates)}).sort_values("published")
    matched = pd.merge_asof(query, events, on="published", direction="backward")
    return pd.Series(matched["value"].to_numpy(), index=query["published"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/master_df.csv")
    parser.add_argument("--output", default="macro_repair_preview.csv")
    parser.add_argument("--days", type=int, default=370)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY is required; refusing to guess macro values")
    from fredapi import Fred
    fred = Fred(api_key=key)
    df = pd.read_csv(args.csv, index_col="date", parse_dates=True).sort_index()
    cutoff = df.index.max() - pd.Timedelta(days=args.days)
    dates = df.index[df.index >= cutoff]
    print(f"Auditing {len(dates)} trading dates from {dates.min().date()} to {dates.max().date()}")
    repaired = df.copy()
    for column, series_id in SERIES.items():
        releases = fred.get_series_all_releases(series_id)
        if releases is None or releases.empty:
            raise RuntimeError(f"Missing FRED release history: {series_id}")
        available = vintage_asof(releases, dates).reindex(dates)
        if available.isna().any():
            raise RuntimeError(f"No point-in-time macro value for {series_id} on {available.isna().sum()} dates")
        original = pd.to_numeric(df.loc[dates, column], errors="coerce")
        changed = ~np.isclose(original.to_numpy(), available.to_numpy(), rtol=1e-7, atol=1e-7, equal_nan=True)
        print(f"{column} ({series_id}): {int(changed.sum())}/{len(dates)} differing rows")
        repaired.loc[dates, column] = available.to_numpy()
    if args.write:
        out = Path(args.output)
        if out.resolve() == Path(args.csv).resolve():
            raise ValueError("Refusing to overwrite source CSV; choose a separate --output")
        repaired.to_csv(out)
        print(f"Wrote repaired copy to {out}; original unchanged")
    else:
        print("Preview only: no file written. Add --write to save a separate repaired copy.")

if __name__ == "__main__":
    main()
