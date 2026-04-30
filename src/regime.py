from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


REGIME_ORDER = ["LOW", "ELEVATED", "CRISIS"]
REGIME_SHIFT = 5


def build_regime_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
    val_end: str,
    shift: int = REGIME_SHIFT,
) -> tuple:
    """
    Build forward-shifted regime classification splits from master_df.

    Labels are shifted `shift` trading days forward so the classifier
    predicts next-week regime rather than current regime.

    Returns
    -------
    X_tr, y_tr, X_va, y_va, X_te, y_te, le, feat_df
    """
    le = LabelEncoder()
    le.fit(REGIME_ORDER)

    feat_df       = df[feature_cols].copy()
    regime_target = df["regime_label"].shift(-shift)
    valid_mask    = feat_df.notna().all(axis=1) & regime_target.notna()
    feat_df       = feat_df[valid_mask]
    regime_target = regime_target[valid_mask]
    y_enc         = le.transform(regime_target.values)

    tr_m = feat_df.index <= train_end
    va_m = (feat_df.index > train_end) & (feat_df.index <= val_end)
    te_m = feat_df.index > val_end

    return (
        feat_df[tr_m].values, y_enc[tr_m],
        feat_df[va_m].values, y_enc[va_m],
        feat_df[te_m].values, y_enc[te_m],
        le,
        feat_df,
    )


def train_regime_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int = 42,
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 40,
) -> xgb.XGBClassifier:
    """
    Train XGBoost regime classifier with balanced sample weights.

    Balanced weights counteract class imbalance: CRISIS (9.4% of days)
    receives ~6.5x the training weight of LOW days.
    """
    sample_wts = compute_sample_weight("balanced", y_tr)

    clf = xgb.XGBClassifier(
        n_estimators          = n_estimators,
        max_depth             = max_depth,
        learning_rate         = learning_rate,
        subsample             = subsample,
        colsample_bytree      = colsample_bytree,
        reg_alpha             = reg_alpha,
        reg_lambda            = reg_lambda,
        eval_metric           = "mlogloss",
        early_stopping_rounds = early_stopping_rounds,
        random_state          = seed,
        n_jobs                = -1,
    )
    clf.fit(
        X_tr, y_tr,
        sample_weight = sample_wts,
        eval_set      = [(X_va, y_va)],
        verbose       = 100,
    )
    return clf


def calibrate_probabilities(
    clf: xgb.XGBClassifier,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling to correct raw XGBoost probability estimates.

    Raw XGBoost probabilities cluster near 0 and 1 (overconfident).
    Platt scaling fits a logistic layer on validation set outputs so that
    a predicted probability of 0.80 corresponds to ~80% empirical frequency.
    Required for Phase 8 bulletin probability statements.
    """
    cal_clf = CalibratedClassifierCV(estimator=clf, cv="prefit", method="sigmoid")
    cal_clf.fit(X_va, y_va)
    return cal_clf


def evaluate_regime_classifier(
    clf: xgb.XGBClassifier,
    X_te: np.ndarray,
    y_te: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """
    Evaluate regime classifier on the test set.

    Returns
    -------
    dict with y_pred, y_proba, macro_f1, per_class_f1, report, confusion_matrix
    """
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)

    return {
        "y_pred"         : y_pred,
        "y_proba"        : y_proba,
        "y_pred_labels"  : le.inverse_transform(y_pred),
        "macro_f1"       : float(f1_score(y_te, y_pred, average="macro")),
        "per_class_f1"   : f1_score(y_te, y_pred, average=None),
        "report"         : classification_report(y_te, y_pred, target_names=le.classes_),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0, 1, 2]),
    }


def predict_regime(
    feature_vector: np.ndarray,
    clf: xgb.XGBClassifier,
    cal_clf: CalibratedClassifierCV,
    le: LabelEncoder,
) -> dict:
    """
    Predict next-week regime for a single feature vector.

    Parameters
    ----------
    feature_vector : (n_features,) array for the current date

    Returns
    -------
    dict with regime, probabilities, calibrated_probabilities
    """
    x        = feature_vector.reshape(1, -1)
    raw_pred = clf.predict(x)[0]
    raw_prob = clf.predict_proba(x)[0]
    cal_prob = cal_clf.predict_proba(x)[0]

    return {
        "regime"               : le.inverse_transform([raw_pred])[0],
        "raw_probabilities"    : dict(zip(REGIME_ORDER, raw_prob.tolist())),
        "calibrated_probabilities": dict(zip(REGIME_ORDER, cal_prob.tolist())),
    }


def get_top_regime_probability(
    cal_clf: CalibratedClassifierCV,
    feature_vector: np.ndarray,
    le: LabelEncoder,
) -> tuple[str, float]:
    """
    Return the predicted regime and its calibrated probability for bulletin use.
    """
    proba   = cal_clf.predict_proba(feature_vector.reshape(1, -1))[0]
    top_idx = int(np.argmax(proba))
    return le.inverse_transform([top_idx])[0], float(proba[top_idx])


def save_regime_classifier(
    clf: xgb.XGBClassifier,
    cal_clf: CalibratedClassifierCV,
    le: LabelEncoder,
    feature_cols: list[str],
    clf_path: str,
    cal_path: str,
) -> None:
    joblib.dump({"clf": clf, "le": le, "feature_cols": feature_cols}, clf_path)
    joblib.dump(cal_clf, cal_path)


def load_regime_classifier(clf_path: str, cal_path: str) -> tuple:
    bundle  = joblib.load(clf_path)
    cal_clf = joblib.load(cal_path)
    return bundle["clf"], cal_clf, bundle["le"], bundle["feature_cols"]
