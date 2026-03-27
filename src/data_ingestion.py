import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Logger, writes INFO messages to console and logs/ingestion_log.txt
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ingestion_log.txt', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

# FRED series to collect
FRED_SERIES = {
    'fedfunds': 'FEDFUNDS',   # Federal Funds Rate
    'cpi':      'CPIAUCSL',   # Consumer Price Index
    'unrate':   'UNRATE',     # Unemployment Rate
    'gs10':     'GS10',       # 10-Year Treasury Yield
    'indpro':   'INDPRO',     # Industrial Production Index
    'm2sl':     'M2SL',       # M2 Money Supply
}

# Output paths
RAW_DIR        = 'data/raw'
VIX_PATH       = os.path.join(RAW_DIR, 'vix_raw.csv')
MACRO_PATH     = os.path.join(RAW_DIR, 'macro_raw.csv')
NEWS_PATH      = os.path.join(RAW_DIR, 'news_raw.csv')
LOG_PATH       = 'logs/ingestion_log.txt'

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    """Create required output directories if they don't exist."""
    for d in ['data/raw', 'data/processed', 'figures', 'models', 'logs']:
        os.makedirs(d, exist_ok=True)


def _fetch_fred_with_retry(fred_client, series_id: str,
                            start_date: str = '2000-01-01',
                            max_retries: int = 3) -> pd.Series | None:
    
    for attempt in range(1, max_retries + 1):
        try:
            return fred_client.get_series(series_id, observation_start=start_date)
        except Exception as exc:
            logger.warning(f'FRED {series_id} attempt {attempt}/{max_retries}: {exc}')
            if attempt < max_retries:
                time.sleep(2 ** attempt)   # 2s → 4s → 8s
    logger.error(f'Failed to fetch {series_id} after {max_retries} attempts.')
    return None


def _deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate headlines.
    Two rows are duplicates if they share the same date AND
    the first 80 characters of the headline (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'date' and 'headline' columns.

    Returns
    -------
    pd.DataFrame with duplicates removed.
    """
    df = df.copy()
    df['_dedup'] = (
        df['date'].astype(str) + '|' +
        df['headline'].str[:80].str.lower().str.strip()
    )
    deduped = df.drop_duplicates(subset='_dedup').drop(columns='_dedup')
    removed = len(df) - len(deduped)
    if removed > 0:
        logger.info(f'Deduplication removed {removed:,} duplicate headlines.')
    return deduped


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_vix(start_date: str = '2000-01-01',
            save: bool = True,
            update_existing: bool = True) -> pd.DataFrame:
    """
    Download VIX daily close prices from Yahoo Finance.

    If vix_raw.csv already exists (from Phase 1), loads it and appends
    any new trading days rather than re-downloading the full history.

    Parameters
    ----------
    start_date : str
        Earliest date to download if starting fresh.
    save : bool
        Whether to save/update data/raw/vix_raw.csv.
    update_existing : bool
        If True, appends new rows to existing file instead of re-downloading.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and a single column 'VIX'.
    """
    _ensure_dirs()

    if update_existing and os.path.exists(VIX_PATH):
        vix_df = pd.read_csv(VIX_PATH, index_col=0, parse_dates=True)
        last_date = vix_df.index[-1]
        logger.info(f'Loaded existing VIX: {len(vix_df)} rows, last date {last_date.date()}')

        # Check whether an update is needed
        days_behind = (pd.Timestamp.today() - last_date).days
        if days_behind > 1:
            logger.info(f'VIX is {days_behind} days behind — updating...')
            new_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            new_raw = yf.download('^VIX', start=new_start, progress=False)
            if len(new_raw) > 0:
                new_raw.columns = ['_'.join(col).strip() for col in new_raw.columns.values]
                new_raw = new_raw[['Close_^VIX']].rename(columns={'Close_^VIX': 'VIX'}).dropna()
                vix_df = pd.concat([vix_df, new_raw])
                vix_df = vix_df[~vix_df.index.duplicated(keep='last')]
                logger.info(f'Appended {len(new_raw)} new VIX rows. Total: {len(vix_df)}')
        else:
            logger.info('VIX data is current — no update needed.')
    else:
        logger.info(f'Downloading full VIX history from {start_date}...')
        raw = yf.download('^VIX', start=start_date, progress=False)
        raw.columns = ['_'.join(col).strip() for col in raw.columns.values]
        vix_df = raw[['Close_^VIX']].rename(columns={'Close_^VIX': 'VIX'}).dropna()
        vix_df.index.name = 'Date'
        logger.info(f'Downloaded {len(vix_df)} VIX rows.')

    if save:
        vix_df.to_csv(VIX_PATH)
        logger.info(f'Saved: {VIX_PATH}')

    return vix_df


def get_macro(start_date: str = '2000-01-01',
              save: bool = True) -> pd.DataFrame:
    
    _ensure_dirs()

    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        raise EnvironmentError(
            'FRED_API_KEY not found. '
            'Add it to your .env file: FRED_API_KEY=your_key_here\n'
            'Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html'
        )

    from fredapi import Fred
    fred = Fred(api_key=fred_key)

    frames = {}
    logger.info('Fetching FRED macro series...')
    for col_name, series_id in FRED_SERIES.items():
        s = _fetch_fred_with_retry(fred, series_id, start_date)
        if s is not None:
            frames[col_name] = s
            logger.info(f'  {series_id:10s} -> {len(s)} obs, latest: {s.dropna().index[-1].date()}')

    if len(frames) < len(FRED_SERIES):
        logger.warning(
            f'Only {len(frames)}/{len(FRED_SERIES)} FRED series fetched. '
            'Check your API key and network connection.'
        )

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.index.name = 'date'

    if save:
        macro_df.to_csv(MACRO_PATH)
        logger.info(f'Saved: {MACRO_PATH} — shape: {macro_df.shape}')

    return macro_df


def get_news(kaggle_path: str | None = None,
             use_newsapi: bool = True,
             save: bool = True) -> pd.DataFrame:
    
    _ensure_dirs()
    frames = []

    # --- Source 1: Kaggle ---
    if kaggle_path and os.path.exists(kaggle_path):
        logger.info(f'Loading Kaggle news dataset: {kaggle_path}')
        try:
            raw = pd.read_csv(kaggle_path)
            logger.info(f'Kaggle raw shape: {raw.shape}, columns: {list(raw.columns)}')

            # Auto-detect headline and date columns
            headline_col = next(
                (c for c in ['title', 'headline', 'text', 'Title', 'Headline'] if c in raw.columns),
                None
            )
            date_col = next(
                (c for c in ['date', 'Date', 'published', 'publishedAt', 'pub_date'] if c in raw.columns),
                None
            )

            if headline_col and date_col:
                kaggle_df = raw[[headline_col, date_col]].rename(
                    columns={headline_col: 'headline', date_col: 'date'}
                )
                kaggle_df['source'] = 'kaggle'
                kaggle_df['date']   = pd.to_datetime(kaggle_df['date'], errors='coerce')
                kaggle_df = kaggle_df.dropna(subset=['date', 'headline'])
                kaggle_df['date'] = kaggle_df['date'].dt.normalize()
                kaggle_df = kaggle_df[kaggle_df['date'] >= '2000-01-01']
                frames.append(kaggle_df[['date', 'headline', 'source']])
                logger.info(f'Kaggle headlines loaded: {len(kaggle_df):,}')
            else:
                logger.warning(
                    f'Could not auto-detect headline/date columns in Kaggle file. '
                    f'Columns found: {list(raw.columns)}'
                )
        except Exception as exc:
            logger.error(f'Failed to load Kaggle file: {exc}')
    elif kaggle_path:
        logger.warning(f'Kaggle path specified but file not found: {kaggle_path}')
    else:
        logger.info('No Kaggle path provided — skipping historical headlines.')

    # --- Source 2: NewsAPI ---
    newsapi_key = os.getenv('NEWS_API_KEY')
    if use_newsapi and newsapi_key:
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=newsapi_key)

            QUERIES = [
                'VIX volatility index',
                'Federal Reserve interest rates',
                'stock market crash recession',
                'inflation CPI unemployment',
                'S&P 500 market selloff',
            ]

            rows = []
            logger.info('Fetching NewsAPI recent headlines...')
            for q in QUERIES:
                for attempt in range(1, 4):
                    try:
                        resp = newsapi.get_everything(
                            q=q,
                            language='en',
                            sort_by='publishedAt',
                            page_size=100,
                            from_param=(datetime.now() - timedelta(days=27)).strftime('%Y-%m-%d'),
                        )
                        for art in resp.get('articles', []):
                            rows.append({
                                'date':     art['publishedAt'][:10],
                                'headline': art.get('title') or '',
                                'source':   'newsapi',
                            })
                        time.sleep(0.5)
                        break
                    except Exception as exc:
                        logger.warning(f'NewsAPI attempt {attempt}/3 for "{q}": {exc}')
                        if attempt < 3:
                            time.sleep(2 ** attempt)

            if rows:
                na_df = pd.DataFrame(rows)
                na_df['date'] = pd.to_datetime(na_df['date'])
                na_df = na_df[na_df['headline'].str.strip().str.len() > 10]
                frames.append(na_df[['date', 'headline', 'source']])
                logger.info(f'NewsAPI headlines loaded: {len(na_df):,}')
        except Exception as exc:
            logger.error(f'NewsAPI ingestion failed: {exc}')
    elif use_newsapi and not newsapi_key:
        logger.info('NEWSAPI_KEY not set — skipping NewsAPI source.')

    # --- Merge, deduplicate, sort ---
    if not frames:
        logger.warning(
            'No news data loaded from any source. '
            'Creating empty placeholder. Sentiment will default to 0.0 in Phase 3.'
        )
        news_df = pd.DataFrame(columns=['date', 'headline', 'source'])
    else:
        news_df = pd.concat(frames, ignore_index=True)
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df[news_df['headline'].notna()]
        news_df = news_df[news_df['headline'].str.strip().str.len() > 10]
        news_df = _deduplicate_news(news_df)
        news_df = news_df.sort_values('date').reset_index(drop=True)
        logger.info(
            f'Final news dataset: {len(news_df):,} headlines, '
            f'{news_df["date"].min().date()} to {news_df["date"].max().date()}'
        )

    if save:
        news_df.to_csv(NEWS_PATH, index=False)
        logger.info(f'Saved: {NEWS_PATH}')

    return news_df


if __name__ == '__main__':
    """
    Run this file directly to ingest all three data sources:
        python src/data_ingestion.py
    or with a Kaggle path:
        python src/data_ingestion.py --kaggle data/raw/analyst_ratings_processed.csv
    """
    import argparse
    parser = argparse.ArgumentParser(description='Ingest all raw data sources.')
    parser.add_argument('--kaggle',  type=str, default=None, help='Path to Kaggle news CSV file')
    parser.add_argument('--no-news', action='store_true',     help='Skip news ingestion entirely')
    args = parser.parse_args()

    logger.info('=== Starting full data ingestion ===')

    vix_df   = get_vix()
    macro_df = get_macro()

    if not args.no_news:
        news_df = get_news(kaggle_path=args.kaggle)
    else:
        logger.info('News ingestion skipped (--no-news flag set).')

    logger.info('=== Ingestion complete ===')
    logger.info(f'VIX:   {len(vix_df)} rows')
    logger.info(f'Macro: {macro_df.shape}')
    if not args.no_news:
        logger.info(f'News:  {len(news_df):,} headlines')
