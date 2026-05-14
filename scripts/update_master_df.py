"""
scripts/update_master_df.py

Runs daily via GitHub Actions at 6:30 AM ET on weekdays.

Updates master_df.csv with three sources — exactly matching Phase 2 ingestion:
  1. VIX         — yfinance (^VIX latest close)
  2. FRED macro  — fedfunds, cpi, unrate, gs10, indpro, m2sl (latest available)
  3. Sentiment   — NewsAPI headlines → FinBERT daily compound score

Usage:
    python scripts/update_master_df.py

Secrets needed in GitHub repo (Settings → Secrets → Actions):
    FRED_API_KEY    — from fred.stlouisfed.org/docs/api/api_key.html
    NEWSAPI_KEY     — from newsapi.org
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / 'data' / 'processed' / 'master_df.csv'

FRED_SERIES = {
    'fedfunds': 'FEDFUNDS',
    'cpi':      'CPIAUCSL',
    'unrate':   'UNRATE',
    'gs10':     'GS10',
    'indpro':   'INDPRO',
    'm2sl':     'M2SL',
}

NEWS_QUERY = (
    'stock market OR financial markets OR S&P 500 OR '
    'Federal Reserve OR inflation OR VIX OR recession OR '
    'interest rates OR Wall Street OR economy'
)
NEWS_SOURCES = (
    'reuters,bloomberg,the-wall-street-journal,financial-times,'
    'cnbc,marketwatch,business-insider'
)


# ── 1. VIX ─────────────────────────────────────────────────────────────────

def fetch_latest_vix() -> tuple[date, float]:
    import yfinance as yf
    hist = yf.Ticker('^VIX').history(period='5d')
    if hist.empty:
        raise RuntimeError('yfinance returned empty VIX history')
    return hist.index[-1].date(), float(hist['Close'].iloc[-1])


# ── 2. FRED ────────────────────────────────────────────────────────────────

def fetch_fred_values() -> dict[str, float]:
    api_key = os.getenv('FRED_API_KEY', '')
    if not api_key:
        print('  WARNING: FRED_API_KEY not set — forward-filling macro values')
        return {}
    try:
        from fredapi import Fred
        fred, out = Fred(api_key=api_key), {}
        for col, sid in FRED_SERIES.items():
            try:
                s = fred.get_series(sid).dropna()
                if not s.empty:
                    out[col] = float(s.iloc[-1])
                    print(f'  FRED {sid}: {out[col]:.4f}')
            except Exception as e:
                print(f'  FRED {sid} failed: {e}')
        return out
    except ImportError:
        print('  fredapi not installed — skipping FRED')
        return {}


# ── 3. NewsAPI + FinBERT ───────────────────────────────────────────────────

def fetch_headlines(target_date: date) -> list[str]:
    api_key = os.getenv('NEWSAPI_KEY', '')
    if not api_key:
        print('  WARNING: NEWSAPI_KEY not set — skipping news fetch')
        return []
    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        from_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time())
        to_dt   = datetime.combine(target_date, datetime.max.time())
        resp = client.get_everything(
            q          = NEWS_QUERY,
            sources    = NEWS_SOURCES,
            language   = 'en',
            from_param = from_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            to         = to_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            sort_by    = 'publishedAt',
            page_size  = 100,
        )
        headlines = [
            a['title'] for a in resp.get('articles', [])
            if a.get('title') and a['title'] != '[Removed]'
        ]
        print(f'  NewsAPI: {len(headlines)} headlines for {target_date}')
        return headlines
    except Exception as e:
        print(f'  NewsAPI failed: {e}')
        return []


def compute_finbert_sentiment(headlines: list[str]) -> float | None:
    """
    Run ProsusAI/finbert over headlines.
    Returns compound score = mean(positive_score - negative_score),
    matching the Phase 2 FinBERT pipeline exactly.
    """
    if not headlines:
        return None
    try:
        from transformers import pipeline
        print(f'  Running FinBERT on {len(headlines)} headlines…')
        finbert = pipeline(
            'text-classification',
            model      = 'ProsusAI/finbert',
            tokenizer  = 'ProsusAI/finbert',
            device     = -1,        # CPU only on GitHub Actions
            truncation = True,
            max_length = 512,
        )
        scores = []
        for i in range(0, len(headlines), 32):
            batch = [h[:512] for h in headlines[i:i+32]]
            try:
                for pred in finbert(batch):
                    lbl   = pred['label'].lower()
                    score = float(pred['score'])
                    scores.append(score if lbl == 'positive' else
                                  -score if lbl == 'negative' else 0.0)
            except Exception as e:
                print(f'    batch error: {e}')
        if not scores:
            return None
        compound = float(np.mean(scores))
        print(f'  Sentiment: {compound:.6f}  (n={len(scores)})')
        return compound
    except ImportError:
        print('  transformers not installed — skipping sentiment')
        return None
    except Exception as e:
        print(f'  FinBERT error: {e}')
        return None


# ── 4. Rolling VIX features ────────────────────────────────────────────────

def recompute_rolling(df: pd.DataFrame, new_vix: float) -> dict[str, float]:
    vix = list(df['vix'].values)
    out = {}
    if 'vix_lag1'       in df.columns: out['vix_lag1']       = vix[-1]
    if 'vix_lag5'       in df.columns: out['vix_lag5']       = vix[-5]  if len(vix) >= 5  else vix[0]
    if 'vix_lag21'      in df.columns: out['vix_lag21']      = vix[-21] if len(vix) >= 21 else vix[0]
    if 'vix_roll_mean5' in df.columns:
        tail = vix[-4:] + [new_vix]
        out['vix_roll_mean5'] = float(np.mean(tail))
    if 'vix_roll_std21' in df.columns:
        tail = vix[-20:] + [new_vix]
        out['vix_roll_std21'] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f'\n=== Daily master_df update — {date.today()} ===\n')

    if not CSV.exists():
        print(f'ERROR: {CSV} not found'); sys.exit(1)

    df        = pd.read_csv(CSV, index_col=0, parse_dates=True)
    df.index  = pd.to_datetime(df.index)
    last_date = df.index[-1].date()
    print(f'Last row: {last_date}  ({len(df)} rows total)')

    # 1. VIX
    print('\n[1] VIX')
    vix_date, vix_close = fetch_latest_vix()
    print(f'    {vix_date}: VIX = {vix_close:.2f}')

    if vix_date <= last_date:
        print(f'    Already up to date ({last_date}). Exiting.')
        sys.exit(0)

    # 2. FRED
    print('\n[2] FRED')
    fred_vals = fetch_fred_values()

    # 3. News sentiment
    print(f'\n[3] News sentiment ({vix_date})')
    headlines = fetch_headlines(vix_date)
    sentiment = compute_finbert_sentiment(headlines)
    if sentiment is None:
        sentiment = float(df['sentiment'].iloc[-1]) if 'sentiment' in df.columns else 0.0
        print(f'    Forward-filling sentiment: {sentiment:.6f}')

    # 4. Build new row
    print('\n[4] Building row')
    new_row = df.iloc[-1].copy()
    new_row['vix'] = vix_close

    for col in FRED_SERIES:
        if col in df.columns:
            new_row[col] = fred_vals.get(col, float(df[col].iloc[-1]))

    if 'sentiment'        in df.columns: new_row['sentiment']        = sentiment
    if 'sentiment_source' in df.columns: new_row['sentiment_source'] = 'finbert'

    for col, val in recompute_rolling(df, vix_close).items():
        new_row[col] = val

    # 5. Append, sort, save
    df.loc[pd.Timestamp(vix_date)] = new_row
    df = df.sort_index()
    df.to_csv(CSV)

    print(f'\n✓ Saved. master_df now has {len(df)} rows.')
    print(f'  VIX: {vix_close:.2f}  |  Sentiment: {sentiment:.6f}')
    for c, v in fred_vals.items():
        print(f'  {c}: {v:.4f}')


if __name__ == '__main__':
    main()
