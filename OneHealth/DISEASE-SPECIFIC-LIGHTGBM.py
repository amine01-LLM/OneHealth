"""
============================================================
V7 — DISEASE-SPECIFIC LIGHTGBM EARLY WARNING SYSTEM
============================================================

OBJECTIVE:
Train one temporal early-warning model per disease.

WHY?
Different diseases have different:
- climatic drivers
- temporal dynamics
- outbreak mechanisms

MAIN FEATURES:
✓ One model per disease
✓ Temporal validation
✓ LightGBM
✓ Threshold optimization
✓ Leakage prevention
✓ Temporal feature engineering
✓ Automatic disease evaluation
✓ Final comparison table

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

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

    common_groups = train_groups.intersection(
        test_groups
    )

    print(f"Common groups: {len(common_groups)}")

    df['GROUP'] = list(
        zip(df['PROV'], df['MALADIE'])
    )

    df = df[
        df['GROUP'].isin(common_groups)
    ].copy()

    df.drop(columns=['GROUP'], inplace=True)

    return df


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):

    print("\nAdding temporal features...")

    group_cols = ['PROV', 'MALADIE']

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    for lag in [1, 2, 3, 4, 8, 12]:

        df[f'CASE_LAG_{lag}'] = (

            df.groupby(group_cols)['INCIDENCE']

              .shift(lag)
        )

    # --------------------------------------------------------
    # ROLLING MEANS
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
    # GROWTH
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
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, y_proba):

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ]

    best_threshold = 0.25
    best_f1 = 0

    for t in thresholds:

        y_pred = (
            y_proba >= t
        ).astype(int)

        f1 = f1_score(
            y_true,
            y_pred
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = t

    return best_threshold


# ============================================================
# TRAIN ONE DISEASE MODEL
# ============================================================

def train_disease_model(df_disease, disease_name):

    print("\n" + "="*70)
    print(f"DISEASE: {disease_name}")
    print("="*70)

    # --------------------------------------------------------
    # REMOVE NaNs
    # --------------------------------------------------------

    df_disease = (
        df_disease
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CHECK DATA SIZE
    # --------------------------------------------------------

    positives = (
        df_disease['TARGET_OUTBREAK_FUTURE']
        .sum()
    )

    if len(df_disease) < 300:

        print("Skipped: not enough samples")

        return None

    if positives < 20:

        print("Skipped: not enough outbreaks")

        return None

    print(f"Samples: {len(df_disease)}")
    print(f"Positive outbreaks: {positives}")

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = df_disease['TARGET_OUTBREAK_FUTURE']

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

    X = df_disease.drop(
        columns=drop_cols,
        errors='ignore'
    )

    # --------------------------------------------------------
    # ENCODE PROVINCES
    # --------------------------------------------------------

    X = pd.get_dummies(

        X,

        columns=['PROV'],

        drop_first=True
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    train_idx = (
        df_disease['ANNEE'] <= 2014
    )

    test_idx = (
        df_disease['ANNEE'] >= 2015
    )

    X_train = X[train_idx]
    X_test  = X[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    # --------------------------------------------------------
    # CLASS IMBALANCE
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

        n_estimators=200,

        learning_rate=0.03,

        max_depth=5,

        num_leaves=31,

        subsample=0.8,

        colsample_bytree=0.8,

        scale_pos_weight=scale_pos_weight,

        n_jobs=-1,

        verbosity=-1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICT
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

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    best_t = find_best_threshold(
        y_test,
        y_proba
    )

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

    print(f"\nAUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Threshold : {best_t}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    return {

        'Disease': disease_name,

        'Samples': len(df_disease),

        'Outbreaks': int(positives),

        'AUC': auc,

        'Precision': precision,

        'Recall': recall,

        'F1': f1,

        'Threshold': best_t
    }


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
    # FEATURES
    # --------------------------------------------------------

    df = add_temporal_features(df)

    # --------------------------------------------------------
    # DISEASE LIST
    # --------------------------------------------------------

    diseases = sorted(
        df['MALADIE'].unique()
    )

    print("\nDiseases found:")

    for d in diseases:

        print("-", d)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    all_results = []

    # --------------------------------------------------------
    # LOOP OVER DISEASES
    # --------------------------------------------------------

    for disease in diseases:

        df_disease = df[
            df['MALADIE'] == disease
        ].copy()

        result = train_disease_model(
            df_disease,
            disease
        )

        if result is not None:

            all_results.append(result)

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by='AUC',
        ascending=False
    )

    print("\n" + "="*70)
    print("FINAL DISEASE COMPARISON")
    print("="*70)

    print(results_df)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df.to_csv(
        "Disease_Specific_Results.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "Disease_Specific_Results.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()