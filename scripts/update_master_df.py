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

def fetch_vix_history(start: date) -> pd.Series:
    """Fetch a rolling year of closes so missed workflow runs are recovered."""
    import yfinance as yf
    # A one-year reconciliation window includes the May-November 2026 gap.
    begin = min(start, date.today() - timedelta(days=370))
    hist = yf.Ticker('^VIX').history(
        start=begin.isoformat(),
        end=(date.today() + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if hist.empty or 'Close' not in hist:
        raise RuntimeError('Yahoo Finance returned no VIX closing prices; refusing to overwrite CSV')
    closes = pd.to_numeric(hist['Close'], errors='coerce').dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes[(closes > 0) & np.isfinite(closes)]
    closes = closes[~closes.index.duplicated(keep='last')].sort_index()
    if closes.empty:
        raise RuntimeError('No valid VIX closes returned')
    return closes


def update_derived_features(df: pd.DataFrame, from_date: pd.Timestamp) -> pd.DataFrame:
    """Recompute dependent VIX fields from the first inserted day onward."""
    v = pd.to_numeric(df['vix'], errors='coerce')
    if v.isna().any() or (v <= 0).any():
        raise RuntimeError('Invalid VIX value in master_df; refusing to publish')
    mask = df.index >= from_date
    derived = {
        'vix_log': np.log(v),
        'vix_return': v.pct_change(),
        'vix_lag1': v.shift(1),
        'vix_lag5': v.shift(5),
        'vix_lag21': v.shift(21),
        'vix_roll_mean5': v.rolling(5).mean(),
        'vix_roll_std21': v.rolling(21).std(),
        'regime_label': pd.Series(
            np.select([v < 20, v < 30], ['LOW', 'ELEVATED'], default='CRISIS'),
            index=df.index,
        ),
    }
    for col, series in derived.items():
        if col in df.columns:
            df.loc[mask, col] = series.loc[mask]
    return df


def main():
    print(f'=== Daily master_df reconciliation — {date.today()} ===')
    if not CSV.exists():
        raise FileNotFoundError(f'Missing dataset: {CSV}')

    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.has_duplicates:
        raise RuntimeError('Duplicate dates in master_df; resolve before publishing')
    df = df.sort_index()
    original_count = len(df)
    last_date = df.index.max().date()
    print(f'Existing: {original_count} rows; latest {last_date}')

    # Reconcile historical trading days, not just the most recent close.
    closes = fetch_vix_history(last_date - timedelta(days=370))
    missing = closes.index.difference(df.index)
    if len(missing):
        print(f'Backfilling {len(missing)} missing trading days, {missing.min().date()} to {missing.max().date()}')
    else:
        print('No missing VIX trading dates in reconciliation window')

    # Preserve existing macro and sentiment observations. Historical backfill
    # cannot reconstruct historical news sentiment from today's NewsAPI query.
    # Carry forward known values, and label carried sentiment explicitly.
    new_dates = []
    for dt in missing:
        prior = df.loc[df.index < dt]
        if prior.empty:
            print(f'Skipping {dt.date()}: no earlier feature row to carry forward')
            continue
        row = prior.iloc[-1].copy()
        row['vix'] = float(closes.loc[dt])
        if 'sentiment_source' in df.columns:
            row['sentiment_source'] = 'forward_fill'
        df.loc[dt] = row
        new_dates.append(dt)
    df = df.sort_index()

    # Refresh the latest observation's non-VIX features only if a new
    # trading day was added; older backfilled rows keep labeled carried data.
    latest_market_day = closes.index.max()
    if latest_market_day in new_dates:
        print('Fetching current macro and sentiment for latest added day')
        fred_vals = fetch_fred_values()
        for col, value in fred_vals.items():
            if col in df.columns:
                df.loc[latest_market_day, col] = value
        headlines = fetch_headlines(latest_market_day.date())
        sentiment = compute_finbert_sentiment(headlines)
        if sentiment is not None and 'sentiment' in df.columns:
            df.loc[latest_market_day, 'sentiment'] = sentiment
            if 'sentiment_source' in df.columns:
                df.loc[latest_market_day, 'sentiment_source'] = 'finbert'

    # Recalculate from the earliest inserted date. Also repair the existing
    # last row, which previously copied stale log/return/regime values.
    first_changed = min(new_dates) if new_dates else df.index.max()
    df = update_derived_features(df, first_changed)

    # Validate before replacing the on-disk dataset.
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise RuntimeError('Dataset dates are not unique and sorted')
    if len(df) < original_count:
        raise RuntimeError('Reconciliation unexpectedly removed observations')
    remaining = closes.index.difference(df.index)
    if len(remaining):
        raise RuntimeError(f'{len(remaining)} VIX dates still missing after reconciliation')
    for col in ('vix', 'vix_log', 'vix_return', 'regime_label'):
        if col in df.columns and df.loc[first_changed:, col].isna().any():
            raise RuntimeError(f'Invalid calculated field: {col}')

    tmp = CSV.with_suffix('.csv.tmp')
    df.to_csv(tmp)
    tmp.replace(CSV)
    print(f'Validated and saved: {len(df)} rows (+{len(new_dates)}), latest {df.index.max().date()}')


if __name__ == '__main__':
    main()
