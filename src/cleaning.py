import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, 'logs', 'cleaning_log.txt'), mode='a'),
    ]
)
logger = logging.getLogger(__name__)

TRAIN_END = '2018-12-31'
VAL_END   = '2021-12-31'


# Paths
RAW_DIR            = 'data/raw'
PROCESSED_DIR      = 'data/processed'
MODELS_DIR         = 'models'

VIX_PATH           = os.path.join(RAW_DIR, 'vix_raw.csv')
MACRO_PATH         = os.path.join(RAW_DIR, 'macro_raw.csv')
NEWS_PATH          = os.path.join(RAW_DIR, 'news_raw.csv')
SENTIMENT_PATH     = os.path.join(PROCESSED_DIR, 'sentiment_scores.csv')
MASTER_PATH        = os.path.join(PROCESSED_DIR, 'master_df.csv')
SCALER_PATH        = os.path.join(MODELS_DIR, 'scaler.pkl')
SCALER_COLS_PATH   = os.path.join(MODELS_DIR, 'scaler_features.json')

MACRO_COLS = ['fedfunds', 'cpi', 'unrate', 'gs10', 'indpro', 'm2sl']
SCALE_COLS = MACRO_COLS + ['sentiment']

def _ensure_dirs():
    for d in [PROCESSED_DIR, MODELS_DIR, os.path.join(BASE_DIR, 'logs')]:
        os.makedirs(d, exist_ok=True)



# VIX cleaning
def clean_vix(vix_path: str = VIX_PATH) -> pd.DataFrame:
    df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    df.index.name = 'date'
    df.columns = ['vix']

    df = df[df.index.dayofweek < 5]
    df = df.dropna()
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    df['vix_log']    = np.log(df['vix'])
    df['vix_return'] = df['vix'].pct_change()

    df['vix_lag1']  = df['vix'].shift(1)
    df['vix_lag5']  = df['vix'].shift(5)
    df['vix_lag21'] = df['vix'].shift(21)

    df['vix_roll_mean5']  = df['vix'].rolling(5).mean()
    df['vix_roll_std21']  = df['vix'].rolling(21).std()

    df['regime_label'] = pd.cut(
        df['vix'],
        bins=[0, 20, 30, np.inf],
        labels=['LOW', 'ELEVATED', 'CRISIS']
    ).astype(str)

    logger.info(f'VIX cleaned: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}')
    return df



# Macro cleaning
def clean_macro(macro_path=MACRO_PATH):
    df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df.index.name = 'date'
    df = df.resample('B').ffill().ffill()
    logger.info(f'Macro forward-filled to {len(df)} business-day rows')
    return df



# FinBERT sentiment
def _score_batch(texts, tokenizer, model):
    import torch
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).numpy()
    return (probs[:, 0] - probs[:, 1]).tolist()


def build_sentiment(news_path=NEWS_PATH, save_path=SENTIMENT_PATH, batch_size=32, checkpoint_every=5000):
    if os.path.exists(save_path):
        logger.info(f'Loading sentiment from disk: {save_path}')
        return pd.read_csv(save_path, parse_dates=['date'])

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm
    _ensure_dirs()

    news_df = pd.read_csv(news_path, parse_dates=['date'])
    news_df = news_df.dropna(subset=['headline'])
    news_df = news_df[news_df['headline'].str.strip().str.len() > 10].reset_index(drop=True)

    checkpoint_path = save_path.replace('.csv', '_checkpoint.csv')
    scored_rows, scored_indices = [], set()

    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        scored_rows    = ckpt.to_dict('records')
        scored_indices = set(ckpt['_idx'].tolist())
        logger.info(f'Resuming from checkpoint: {len(scored_indices):,} already scored')

    logger.info('Loading FinBERT...')
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model     = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    model.eval()

    remaining     = news_df[~news_df.index.isin(scored_indices)]
    batch_texts   = []
    batch_indices = []

    for idx, row in tqdm(remaining.iterrows(), total=len(remaining), desc='FinBERT'):
        batch_texts.append(row['headline'])
        batch_indices.append(idx)
        if len(batch_texts) == batch_size:
            scores = _score_batch(batch_texts, tokenizer, model)
            for i, score in zip(batch_indices, scores):
                scored_rows.append({'_idx': i, 'date': news_df.loc[i, 'date'], 'compound': score})
            batch_texts.clear()
            batch_indices.clear()
            if len(scored_rows) % checkpoint_every < batch_size:
                pd.DataFrame(scored_rows).to_csv(checkpoint_path, index=False)

    if batch_texts:
        scores = _score_batch(batch_texts, tokenizer, model)
        for i, score in zip(batch_indices, scores):
            scored_rows.append({'_idx': i, 'date': news_df.loc[i, 'date'], 'compound': score})

    scored_df = pd.DataFrame(scored_rows)
    scored_df['date'] = pd.to_datetime(scored_df['date'])
    daily = (
        scored_df.groupby(scored_df['date'].dt.normalize())['compound']
        .mean().reset_index().rename(columns={'compound': 'sentiment'})
    )
    daily['sentiment_source'] = 'finbert'
    daily.to_csv(save_path, index=False)
    logger.info(f'Saved: {save_path}')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return daily



# Master dataframe assembly

def build_master(vix_df, macro_df, sentiment_df):
    _ensure_dirs()

    df = vix_df.copy()
    df = df.join(macro_df[MACRO_COLS], how='left').ffill()

    sent = sentiment_df.copy()
    sent['date'] = pd.to_datetime(sent['date']).dt.normalize()
    sent = sent.set_index('date')

    df = df.join(sent[['sentiment', 'sentiment_source']], how='left')
    df['sentiment_source'] = df['sentiment_source'].fillna('default_neutral')
    df['sentiment']        = df['sentiment'].fillna(0.0)
    df = df.dropna(subset=['vix_lag21', 'vix_roll_std21'])
    df = df.dropna(subset=MACRO_COLS)

    logger.info(f'Master df assembled: {df.shape}, {df.index.min().date()} to {df.index.max().date()}')
    return df



# Scaling
def fit_and_apply_scaler(master_df):
    df         = master_df.copy()
    train_mask = df.index <= TRAIN_END
    scaler     = StandardScaler()

    df.loc[train_mask,  SCALE_COLS] = scaler.fit_transform(df.loc[train_mask,  SCALE_COLS])
    df.loc[~train_mask, SCALE_COLS] = scaler.transform(df.loc[~train_mask, SCALE_COLS])
    joblib.dump(scaler, SCALER_PATH)
    with open(SCALER_COLS_PATH, 'w') as f:
        json.dump(SCALE_COLS, f)

    logger.info(f'Scaler fitted on train rows ({train_mask.sum()}), saved to {SCALER_PATH}')
    return df



# Quality checks
def run_quality_checks(df):

    print('\n' + '=' * 55)
    print('  MASTER DATAFRAME - QUALITY REPORT')
    print('=' * 55)

    print(f'\n  Shape          : {df.shape}')
    print(f'  Date range     : {df.index.min().date()} to {df.index.max().date()}')

    train = df[df.index <= TRAIN_END]
    val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test  = df[df.index > VAL_END]

    print(f'\n  Train rows     : {len(train):,}  (through {TRAIN_END})')
    print(f'  Val rows       : {len(val):,}  ({TRAIN_END} to {VAL_END})')
    print(f'  Test rows      : {len(test):,}  ({VAL_END} to present)')

    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]

    print(f'\n  NaN check      : {"No NaN values" if len(nan_cols) == 0 else nan_cols}')
    print(f'\n  Regime distribution:')

    for label, pct in (df['regime_label'].value_counts(normalize=True) * 100).items():
        print(f'    {label:<10}: {pct:.1f}%')

    fb = (df['sentiment_source'] == 'finbert').sum()

    print(f'\n  Sentiment coverage: {fb:,} / {len(df):,} days ({fb/len(df)*100:.1f}% FinBERT scored)')

    date_gaps = pd.date_range(df.index.min(), df.index.max(), freq='B').difference(df.index)

    print(f'  Date continuity : {"No missing business days" if len(date_gaps) == 0 else f"{len(date_gaps)} missing business days (market holidays)"}')
    print('\n' + '=' * 55 + '\n')


# Main pipeline
def run_cleaning_pipeline(save=True):
    _ensure_dirs()

    logger.info('=== Starting cleaning pipeline ===')
    vix_df       = clean_vix()
    macro_df     = clean_macro()
    sentiment_df = build_sentiment()
    master_df    = build_master(vix_df, macro_df, sentiment_df)
    master_df    = fit_and_apply_scaler(master_df)

    run_quality_checks(master_df)
    if save:
        master_df.to_csv(MASTER_PATH)
        logger.info(f'Saved: {MASTER_PATH}')

    logger.info('=== Cleaning pipeline complete ===')
    return master_df


if __name__ == '__main__':
    run_cleaning_pipeline()
