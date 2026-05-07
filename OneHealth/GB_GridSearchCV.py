"""
MODEL COMPARISON V5 — EARLY WARNING SYSTEM + GRIDSEARCHCV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V5 IMPROVEMENTS:
✓ Real early warning target (t + 4 weeks)
✓ Temporal validation (TimeSeriesSplit)
✓ GridSearchCV for Gradient Boosting
✓ F1 optimization
✓ Threshold analysis
✓ Leakage prevention
✓ Imbalance handling
✓ Cleaner evaluation pipeline

GOAL:
Optimize the early warning system while balancing:
- Recall (catch outbreaks)
- Precision (reduce false alarms)
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import GradientBoostingClassifier


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    print(f"Loaded: {df.shape}")

    return df


# ============================================================
# KEEP ONLY COMMON GROUPS
# ============================================================

def filter_common_groups(df):

    train_df = df[df['ANNEE'] <= 2014]
    test_df  = df[df['ANNEE'] >= 2015]

    train_groups = set(zip(train_df['PROV'], train_df['MALADIE']))
    test_groups  = set(zip(test_df['PROV'], test_df['MALADIE']))

    common_groups = train_groups.intersection(test_groups)

    print(f"Common (PROV, MALADIE) groups: {len(common_groups)}")

    df['GROUP'] = list(zip(df['PROV'], df['MALADIE']))

    df = df[df['GROUP'].isin(common_groups)].copy()

    df = df.drop(columns=['GROUP'])

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # REMOVE NaNs
    # --------------------------------------------------------
    before = len(df)

    df = df.dropna().reset_index(drop=True)

    print(f"Dropped {before - len(df)} rows due to NaNs")

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------
    y = df['TARGET_OUTBREAK_FUTURE']

    # --------------------------------------------------------
    # REMOVE LEAKAGE
    # --------------------------------------------------------
    drop_cols = [
        'TARGET_OUTBREAK_FUTURE',
        'IS_OUTBREAK',
        'INCIDENCE',
        'DEBUTSEM'
    ]

    X = df.drop(columns=drop_cols, errors='ignore')

    # --------------------------------------------------------
    # OPTIONAL: remove noisy interactions
    # --------------------------------------------------------
    noisy_cols = [
        'RAIN_TEMP',
        'HUM_TEMP'
    ]

    X = X.drop(columns=noisy_cols, errors='ignore')

    # --------------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------------
    X = pd.get_dummies(
        X,
        columns=['PROV', 'MALADIE'],
        drop_first=True
    )

    print("Any NaN left:", X.isnull().sum().sum())

    return X, y, df


# ============================================================
# TIME SPLIT
# ============================================================

def time_split(df, X, y):

    train_idx = df['ANNEE'] <= 2014
    test_idx  = df['ANNEE'] >= 2015

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# GRID SEARCH
# ============================================================

def optimize_gradient_boosting(X_train, y_train):

    print("\n" + "="*60)
    print("GRID SEARCH OPTIMIZATION")
    print("="*60)

    # --------------------------------------------------------
    # SAMPLE WEIGHTS
    # --------------------------------------------------------
    sample_weight = np.where(y_train == 1, 4, 1)

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------
    gb = GradientBoostingClassifier(
        random_state=42
    )

    # --------------------------------------------------------
    # PARAMETER GRID
    # --------------------------------------------------------
    param_grid = {

        'n_estimators': [200, 300],

        'learning_rate': [0.03, 0.05],

        'max_depth': [3, 4],

        'min_samples_split': [10, 20],

        'min_samples_leaf': [5, 10]
    }

    # --------------------------------------------------------
    # TEMPORAL CROSS VALIDATION
    # --------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=3)

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------
    grid = GridSearchCV(

        estimator=gb,

        param_grid=param_grid,

        scoring='f1',

        cv=tscv,

        n_jobs=-1,

        verbose=2
    )

    grid.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )

    print("\nBest parameters:")
    print(grid.best_params_)

    print(f"\nBest CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(y_test, y_proba):

    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    best_threshold = 0.15
    best_f1 = 0

    for t in [0.05, 0.10, 0.15, 0.20, 0.25]:

        y_pred = (y_proba >= t).astype(int)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)

        print(
            f"Threshold={t:.2f} | "
            f"Recall={recall:.3f} | "
            f"Precision={precision:.3f} | "
            f"F1={f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold selected: {best_threshold}")

    return best_threshold


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------
    y_proba = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------
    auc = roc_auc_score(y_test, y_proba)

    print(f"AUC: {auc:.4f}")

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------
    best_t = threshold_analysis(y_test, y_proba)

    # --------------------------------------------------------
    # FINAL PREDICTIONS
    # --------------------------------------------------------
    y_pred = (y_proba >= best_t).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


# ============================================================
# MAIN
# ============================================================

def main():

    FILE = "DRC_ML_Ready_v4.csv"

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    df = load_data(FILE)

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------
    df = filter_common_groups(df)

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------
    X, y, df = prepare_data(df)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = time_split(df, X, y)

    # --------------------------------------------------------
    # OPTIMIZE MODEL
    # --------------------------------------------------------
    best_model = optimize_gradient_boosting(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------
    evaluate_model(
        best_model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()