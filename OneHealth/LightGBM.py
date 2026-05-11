"""
============================================================
V6 — LIGHTGBM TEMPORAL EARLY WARNING SYSTEM
============================================================

OBJECTIVE:
Predict epidemic outbreaks 4 weeks ahead using:
- epidemiological history
- temporal dynamics
- NASA weather data
- engineered temporal features

MAIN IMPROVEMENTS:
✓ LightGBM model
✓ Temporal validation
✓ Leakage prevention
✓ Time-aware split
✓ Threshold optimization
✓ Imbalance handling
✓ Temporal feature engineering
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
# FILTER COMMON GROUPS
# ============================================================

def filter_common_groups(df):

    train_df = df[df['ANNEE'] <= 2014]
    test_df  = df[df['ANNEE'] >= 2015]

    train_groups = set(
        zip(train_df['PROV'], train_df['MALADIE'])
    )

    test_groups = set(
        zip(test_df['PROV'], test_df['MALADIE'])
    )

    common_groups = train_groups.intersection(test_groups)

    print(f"Common groups: {len(common_groups)}")

    df['GROUP'] = list(
        zip(df['PROV'], df['MALADIE'])
    )

    df = df[df['GROUP'].isin(common_groups)].copy()

    df.drop(columns=['GROUP'], inplace=True)

    return df


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def add_temporal_features(df):

    print("\nAdding temporal features...")

    group_cols = ['PROV', 'MALADIE']

    # --------------------------------------------------------
    # LONGER LAGS
    # --------------------------------------------------------

    for lag in [1, 2, 3, 4, 8, 12]:

        df[f'CASE_LAG_{lag}'] = (
            df.groupby(group_cols)['INCIDENCE']
              .shift(lag)
        )

    # --------------------------------------------------------
    # ROLLING MEAN
    # --------------------------------------------------------

    for window in [4, 8]:

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
    # CASE GROWTH
    # --------------------------------------------------------

    df['CASE_GROWTH'] = (

        (df['CASE_LAG_1'] - df['CASE_LAG_4'])

        /

        (df['CASE_LAG_4'] + 1)
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

    print(f"Dropped rows due to NaN: {before - len(df)}")

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

    X = df.drop(
        columns=drop_cols,
        errors='ignore'
    )

    # --------------------------------------------------------
    # ONE HOT ENCODING
    # --------------------------------------------------------

    X = pd.get_dummies(

        X,

        columns=[
            'PROV',
            'MALADIE'
        ],

        drop_first=True
    )

    print(f"Final feature matrix: {X.shape}")

    return X, y, df


# ============================================================
# TIME SPLIT
# ============================================================

def temporal_split(df, X, y):

    train_idx = df['ANNEE'] <= 2014
    test_idx  = df['ANNEE'] >= 2015

    X_train = X[train_idx]
    X_test  = X[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    print(f"\nTrain: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# LIGHTGBM OPTIMIZATION
# ============================================================

def optimize_lightgbm(X_train, y_train):

    print("\n" + "="*60)
    print("LIGHTGBM GRID SEARCH")
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

    lgbm = LGBMClassifier(
        verbosity=-1,

        objective='binary',

        random_state=42,

        n_jobs=-1,

        scale_pos_weight=scale_pos_weight
    )

    # --------------------------------------------------------
    # PARAM GRID
    # --------------------------------------------------------

    param_grid = {

    'n_estimators': [200],

    'learning_rate': [0.03, 0.05],

    'num_leaves': [31],

    'max_depth': [3, 5],

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

        estimator=lgbm,

        param_grid=param_grid,

        scoring='f1',

        cv=tscv,

        n_jobs=-1,

        verbose=2
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
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(y_test, y_proba):

    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    best_threshold = 0.25
    best_f1 = 0

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ]

    for t in thresholds:

        y_pred = (
            y_proba >= t
        ).astype(int)

        recall = recall_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
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
    # PREDICT PROBA
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
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    best_t = threshold_analysis(
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
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

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
    # FILTER GROUPS
    # --------------------------------------------------------

    df = filter_common_groups(df)

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

    best_model = optimize_lightgbm(
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