from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


EXCLUSION_WINDOW = 90
SEQUENCE_LENGTH  = 60


def get_analogues(
    current_vector   : np.ndarray,
    historical_df    : pd.DataFrame,
    feature_cols     : list[str],
    top_n            : int = 3,
    current_date     = None,
    attention_weights: np.ndarray = None,
    sequence_length  : int = SEQUENCE_LENGTH,
    exclusion_window : int = EXCLUSION_WINDOW,
) -> list[dict]:
    """
    Retrieve the top_n most similar historical dates using cosine similarity.

    Two modes are supported:

    Standard mode (attention_weights=None):
        Cosine similarity between current_vector and every row in historical_df.
        All features contribute equally.

    Attention-weighted mode (attention_weights provided):
        The query vector is built by collapsing the 60-day input window using
        LSTM attention weights as a weighted mean across timesteps. Features
        the model focused on most receive proportionally higher influence in
        the similarity search. This links the Phase 5 attention mechanism
        directly to the analogue retrieval.

    Parameters
    ----------
    current_vector    : (n_features,) feature vector for the query date
    historical_df     : DataFrame with DatetimeIndex — the search library
    feature_cols      : ordered list of feature column names
    top_n             : number of analogues to return
    current_date      : query date used to exclude nearby dates from results
    attention_weights : (sequence_length,) LSTM attention weights for query date
    sequence_length   : LSTM context window length (default 60)
    exclusion_window  : trading days around current_date to mask from results

    Returns
    -------
    list of dicts, each containing:
        date, similarity, vix, regime, vix_return
    """
    hist_matrix = historical_df[feature_cols].values

    if attention_weights is not None and current_date is not None:
        try:
            loc_i = historical_df.index.get_loc(pd.Timestamp(current_date))
        except KeyError:
            loc_i = int(historical_df.index.searchsorted(pd.Timestamp(current_date)))

        start  = max(0, loc_i - sequence_length)
        window = historical_df[feature_cols].values[start:loc_i]

        if len(window) > 0:
            at    = attention_weights[-len(window):]
            at    = at / (at.sum() + 1e-8)
            query = (window * at[:, None]).sum(axis=0, keepdims=True)
        else:
            query = current_vector.reshape(1, -1)
    else:
        query = current_vector.reshape(1, -1)

    sims = cosine_similarity(query, hist_matrix)[0]

    if current_date is not None:
        q_ts = pd.Timestamp(current_date)
        for i, d in enumerate(historical_df.index):
            if abs((d - q_ts).days) < exclusion_window:
                sims[i] = -1.0

    top_idx = np.argsort(sims)[::-1][:top_n]

    return [
        {
            "date"      : historical_df.index[i].date(),
            "similarity": round(float(sims[i]), 4),
            "vix"       : round(float(historical_df.iloc[i]["vix"]), 2),
            "regime"    : historical_df.iloc[i]["regime_label"],
            "vix_return": round(float(historical_df.iloc[i].get("vix_return", np.nan)), 4),
        }
        for i in top_idx
    ]


def get_analogues_both_modes(
    current_vector   : np.ndarray,
    historical_df    : pd.DataFrame,
    feature_cols     : list[str],
    current_date,
    attention_weights: np.ndarray,
    top_n            : int = 3,
    sequence_length  : int = SEQUENCE_LENGTH,
    exclusion_window : int = EXCLUSION_WINDOW,
) -> tuple[list[dict], list[dict]]:
    """
    Run both standard and attention-weighted analogue search in one call.

    Returns
    -------
    (standard_results, attention_results) — each is a list of top_n dicts
    """
    standard = get_analogues(
        current_vector   = current_vector,
        historical_df    = historical_df,
        feature_cols     = feature_cols,
        top_n            = top_n,
        current_date     = current_date,
        exclusion_window = exclusion_window,
    )
    attention = get_analogues(
        current_vector    = current_vector,
        historical_df     = historical_df,
        feature_cols      = feature_cols,
        top_n             = top_n,
        current_date      = current_date,
        attention_weights = attention_weights,
        sequence_length   = sequence_length,
        exclusion_window  = exclusion_window,
    )
    return standard, attention


def format_analogues_for_bulletin(analogues: list[dict], top_n: int = 2) -> str:
    """
    Format top analogues into a plain-English string for the risk bulletin.

    Example output:
        "October 2008 (VIX 52.1, CRISIS, similarity 0.981) and
         March 2020 (VIX 82.7, CRISIS, similarity 0.973)"
    """
    lines = []
    for a in analogues[:top_n]:
        date_str = pd.Timestamp(str(a["date"])).strftime("%B %Y")
        lines.append(
            f"{date_str} (VIX {a['vix']:.1f}, {a['regime']}, similarity {a['similarity']:.3f})"
        )
    return " and ".join(lines)


def build_analogue_matrix(
    query_vector  : np.ndarray,
    analogues     : list[dict],
    historical_df : pd.DataFrame,
    feature_cols  : list[str],
    query_label   : str = "QUERY",
) -> tuple[np.ndarray, list[str]]:
    """
    Build a normalised feature matrix for heatmap visualisation.

    Stacks the query vector and each analogue's feature vector,
    then z-scores column-wise so features are comparable on one scale.

    Returns
    -------
    (normalised_matrix, row_labels)
    """
    rows   = [query_vector]
    labels = [query_label]

    for a in analogues:
        date_match = historical_df.index[historical_df.index.date == a["date"]]
        if len(date_match) > 0:
            rows.append(historical_df.loc[date_match[0], feature_cols].values)
            labels.append(f"{a['date']} | VIX={a['vix']:.0f} | {a['regime']}")

    matrix  = np.vstack(rows)
    normed  = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-8)
    return normed, labels


def search_analogues_for_dates(
    query_dates    : list,
    test_df        : pd.DataFrame,
    historical_df  : pd.DataFrame,
    feature_cols   : list[str],
    te_dates,
    te_attn        : np.ndarray,
    top_n          : int = 3,
    sequence_length: int = SEQUENCE_LENGTH,
    labels         : list[str] = None,
) -> list[dict]:
    """
    Run analogue search across a list of named query dates.

    Parameters
    ----------
    query_dates     : list of date strings (e.g. ['2022-03-07', '2023-03-14'])
    test_df         : test split dataframe
    historical_df   : training period dataframe used as search library
    feature_cols    : feature column names
    te_dates        : array of dates aligned to test LSTM inference output
    te_attn         : (N, seq_len) attention weights from LSTM test inference
    top_n           : analogues per query date
    sequence_length : LSTM context window
    labels          : optional human-readable labels for each query date

    Returns
    -------
    list of result dicts, one per query date
    """
    if labels is None:
        labels = query_dates

    results = []

    for query_date, label in zip(query_dates, labels):
        try:
            q_ts  = pd.Timestamp(query_date)
            q_row = test_df.index[test_df.index.searchsorted(q_ts)]
            q_vec = test_df.loc[q_row, feature_cols].values
            q_vix = float(test_df.loc[q_row, "vix"])

            attn_match    = np.where(te_dates == q_row)[0]
            q_attn        = te_attn[attn_match[0]] if len(attn_match) > 0 else None

            std_res, att_res = get_analogues_both_modes(
                current_vector    = q_vec,
                historical_df     = historical_df,
                feature_cols      = feature_cols,
                current_date      = q_row,
                attention_weights = q_attn,
                top_n             = top_n,
                sequence_length   = sequence_length,
            )
            results.append({
                "query_date"         : q_row.date(),
                "label"              : label,
                "query_vix"          : q_vix,
                "standard_analogues" : std_res,
                "attention_analogues": att_res,
            })
        except Exception as exc:
            results.append({
                "query_date": query_date,
                "label"     : label,
                "error"     : str(exc),
            })

    return results
