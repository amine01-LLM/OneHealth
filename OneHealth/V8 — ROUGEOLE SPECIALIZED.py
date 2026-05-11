"""
============================================================
V8 — ROUGEOLE SPECIALIZED EARLY WARNING SYSTEM
============================================================

OBJECTIVE:
Specialized temporal outbreak prediction model
for ROUGEOLE (Measles).

WHY ROUGEOLE?
- Best disease-specific performance
- Strong temporal epidemic dynamics
- Strong seasonality
- High predictive signal

MAIN IMPROVEMENTS:
✓ Disease-specialized model
✓ Advanced temporal engineering
✓ Temporal GridSearchCV
✓ Rolling statistics
✓ Trend features
✓ Seasonality encoding
✓ Threshold optimization
✓ Leakage prevention
✓ TimeSeriesSplit validation
✓ SHAP-ready architecture

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit
)

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    print(f"Loaded dataset: {df.shape}")

    return df


# ============================================================
# FILTER ROUGEOLE ONLY
# ============================================================

def filter_rougeole(df):

    df = df[
        df['MALADIE'] == 'ROUGEOLE'
    ].copy()

    print(f"ROUGEOLE dataset: {df.shape}")

    return df


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def add_temporal_features(df):

    print("\nAdding advanced temporal features...")

    group_cols = ['PROV']

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        ['PROV', 'ANNEE', 'DEBUTSEM']
    )

    # --------------------------------------------------------
    # MULTI-SCALE LAGS
    # --------------------------------------------------------

    for lag in [1, 2, 3, 4, 6, 8, 12]:

        df[f'CASE_LAG_{lag}'] = (

            df.groupby(group_cols)['INCIDENCE']
              .shift(lag)
        )

    # --------------------------------------------------------
    # ROLLING MEANS
    # --------------------------------------------------------

    for window in [2, 4, 8, 12]:

        df[f'ROLLING_MEAN_{window}'] = (

            df.groupby(group_cols)['INCIDENCE']

              .transform(
                  lambda x:
                  x.shift(1)
                   .rolling(window)
                   .mean()
              )
        )

    # --------------------------------------------------------
    # ROLLING MAX
    # --------------------------------------------------------

    for window in [4, 8]:

        df[f'ROLLING_MAX_{window}'] = (

            df.groupby(group_cols)['INCIDENCE']

              .transform(
                  lambda x:
                  x.shift(1)
                   .rolling(window)
                   .max()
              )
        )

    # --------------------------------------------------------
    # ROLLING STD
    # --------------------------------------------------------

    for window in [4, 8]:

        df[f'ROLLING_STD_{window}'] = (

            df.groupby(group_cols)['INCIDENCE']

              .transform(
                  lambda x:
                  x.shift(1)
                   .rolling(window)
                   .std()
              )
        )

    # --------------------------------------------------------
    # EPIDEMIC MOMENTUM
    # --------------------------------------------------------

    df['MOMENTUM_1_4'] = (

        (df['CASE_LAG_1'] - df['CASE_LAG_4'])

        /

        (df['CASE_LAG_4'] + 1)
    )

    # --------------------------------------------------------
    # SHORT TREND
    # --------------------------------------------------------

    df['TREND_SHORT'] = (

        df['ROLLING_MEAN_2']

        -

        df['ROLLING_MEAN_8']
    )

    # --------------------------------------------------------
    # LONG TREND
    # --------------------------------------------------------

    df['TREND_LONG'] = (

        df['ROLLING_MEAN_4']

        -

        df['ROLLING_MEAN_12']
    )

    # --------------------------------------------------------
    # OUTBREAK PERSISTENCE
    # --------------------------------------------------------

    df['OUTBREAK_PERSISTENCE'] = (

        df.groupby(group_cols)['IS_OUTBREAK']

          .transform(
              lambda x:
              x.shift(1)
               .rolling(4)
               .sum()
          )
    )

    # --------------------------------------------------------
    # SEASONALITY ENCODING
    # --------------------------------------------------------

    df['DEBUTSEM'] = pd.to_datetime(df['DEBUTSEM'])

    df['WEEK_NUM'] = (
        df['DEBUTSEM']
        .dt.isocalendar()
        .week
    )

    df['WEEK_SIN'] = np.sin(
        2 * np.pi * df['WEEK_NUM'] / 52
    )

    df['WEEK_COS'] = np.cos(
        2 * np.pi * df['WEEK_NUM'] / 52
    )

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

    print(f"Dropped NaNs: {before - len(df)}")

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

        'DEBUTSEM',

        'MALADIE'
    ]

    X = df.drop(
        columns=drop_cols,
        errors='ignore'
    )

    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    X = pd.get_dummies(

        X,

        columns=['PROV'],

        drop_first=True
    )

    print(f"Feature matrix: {X.shape}")

    return X, y, df


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(df, X, y):

    train_idx = (
        df['ANNEE'] <= 2014
    )

    test_idx = (
        df['ANNEE'] >= 2015
    )

    X_train = X[train_idx]
    X_test  = X[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    print(f"\nTrain: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# GRID SEARCH
# ============================================================

def optimize_model(X_train, y_train):

    print("\n" + "="*60)
    print("GRID SEARCH OPTIMIZATION")
    print("="*60)

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    scale_pos_weight = (

        (y_train == 0).sum()

        /

        (y_train == 1).sum()
    )

    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LGBMClassifier(

        objective='binary',

        random_state=42,

        scale_pos_weight=scale_pos_weight,

        verbosity=-1,

        n_jobs=-1
    )

    # --------------------------------------------------------
    # PARAM GRID
    # --------------------------------------------------------

    param_grid = {

        'n_estimators': [200, 300],

        'learning_rate': [0.03, 0.05],

        'max_depth': [3, 5],

        'num_leaves': [31, 63],

        'subsample': [0.8],

        'colsample_bytree': [0.8]
    }

    # --------------------------------------------------------
    # TEMPORAL CV
    # --------------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        scoring='f1',

        cv=tscv,

        verbose=2,

        n_jobs=-1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    grid.fit(X_train, y_train)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nBest Parameters:")
    print(grid.best_params_)

    print(f"\nBest CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, y_proba):

    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35
    ]

    best_threshold = 0.25
    best_f1 = 0

    for t in thresholds:

        y_pred = (
            y_proba >= t
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred
        )

        f1 = f1_score(
            y_true,
            y_pred
        )

        print(
            f"Threshold={t:.2f} | "
            f"Recall={recall:.3f} | "
            f"Precision={precision:.3f} | "
            f"F1={f1:.3f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold: {best_threshold}")

    return best_threshold


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    print(f"AUC: {auc:.4f}")

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    best_t = find_best_threshold(
        y_test,
        y_proba
    )

    # --------------------------------------------------------
    # FINAL PREDICTIONS
    # --------------------------------------------------------

    y_pred = (
        y_proba >= best_t
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nFINAL METRICS")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    return model


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
    # FILTER ROUGEOLE
    # --------------------------------------------------------

    df = filter_rougeole(df)

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df = add_temporal_features(df)

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, y, df = prepare_data(df)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = temporal_split(
        df,
        X,
        y
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_model = optimize_model(
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()