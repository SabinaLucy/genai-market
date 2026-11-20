"""Audit sentiment freshness without inventing missing historical headlines.

Creates a dated CSV report; source dataset is never modified.
Historical NewsAPI access depends on the account plan, so this script only
flags stale observations rather than silently manufacturing sentiment.
"""
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/master_df.csv")
    p.add_argument("--output", default="sentiment_quality_report.csv")
    p.add_argument("--max-stale-days", type=int, default=5)
    a = p.parse_args()
    df = pd.read_csv(a.csv, parse_dates=["date"]).sort_values("date")
    if "sentiment_source" not in df or "sentiment" not in df:
        raise ValueError("Missing sentiment or sentiment_source column")
    observed = df["sentiment_source"].eq("finbert") & pd.to_numeric(df["sentiment"], errors="coerce").notna()
    source_date = df["date"].where(observed).ffill()
    age = (df["date"] - source_date).dt.days
    report = pd.DataFrame({
        "date": df["date"], "sentiment": df["sentiment"],
        "source": df["sentiment_source"], "last_observed_date": source_date,
        "age_calendar_days": age,
        "usable": observed | (age.le(a.max_stale_days) & age.notna()),
        "independently_observed": observed,
    })
    report.to_csv(a.output, index=False)
    print(f"Recent 60 rows: {int(observed.tail(60).sum())} independently observed; "
          f"{int((~report['usable'].tail(60)).sum())} stale/missing")
    print(f"Report saved: {a.output}. No production data changed.")

if __name__ == "__main__":
    main()
