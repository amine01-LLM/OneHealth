"""
MODEL COMPARISON V4 — EARLY WARNING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY CHANGES:
✓ Uses TARGET_OUTBREAK_FUTURE (t + H)
✓ Removes leakage features (INCIDENCE, IS_OUTBREAK)
✓ Optimizes for recall (early detection)
✓ Includes threshold tuning
✓ Keeps multi-model comparison

GOAL:
Find the best model for predicting outbreaks 4 weeks in advance
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    print(f"Loaded: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):

     # 🔥 DROP ALL NaNs
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows due to NaNs")

    # --------------------------------------------------------
    # TARGET (EARLY WARNING)
    # --------------------------------------------------------
    y = df['TARGET_OUTBREAK_FUTURE']

    # --------------------------------------------------------
    # REMOVE LEAKAGE FEATURES
    # --------------------------------------------------------
    drop_cols = [
        'TARGET_OUTBREAK_FUTURE',
        'IS_OUTBREAK',
        'INCIDENCE',   # CRITICAL REMOVE
        'DEBUTSEM'
    ]

    X = df.drop(columns=drop_cols, errors='ignore')

    # --------------------------------------------------------
    # HANDLE CATEGORICAL
    # --------------------------------------------------------
    X = pd.get_dummies(X, columns=['PROV', 'MALADIE'], drop_first=True)

    return X, y, df


# ============================================================
# TIME SPLIT (NO LEAKAGE)
# ============================================================

def time_split(df, X, y):

    train_idx = df['ANNEE'] <= 2014
    test_idx  = df['ANNEE'] >= 2015

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# MODELS
# ============================================================

def get_models(scale_pos_weight):

    models = {

        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=42
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        ),

        "SVM": SVC(
            probability=True,
            class_weight='balanced'
        ),

        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42
        )
    }

    return models


# ============================================================
# THRESHOLD SEARCH (RECALL FOCUSED)
# ============================================================

def find_best_threshold(y_true, y_proba):

    thresholds = np.arange(0.05, 0.5, 0.05)

    best_recall = 0
    best_threshold = 0.2

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        recall = tp / (tp + fn + 1e-6)

        if recall > best_recall:
            best_recall = recall
            best_threshold = t

    return best_threshold, best_recall


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "="*60)
    print(f"MODEL: {name}")
    print("="*60)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")

    # Threshold tuning
    best_t, best_recall = find_best_threshold(y_test.values, y_proba)

    print(f"Best threshold (recall-focused): {best_t}")
    print(f"Recall at best threshold: {best_recall:.4f}")

    # Final predictions
    y_pred = (y_proba >= best_t).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model": name,
        "auc": auc,
        "recall": best_recall,
        "threshold": best_t
    }

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
# MAIN
# ============================================================

def main():

    FILE = "DRC_ML_Ready_v4.csv"

    df = load_data(FILE)

    # 🔥 IMPORTANT FILTER
    df = filter_common_groups(df)
    X, y, df = prepare_data(df)

    X_train, X_test, y_train, y_test = time_split(df, X, y)

    # --------------------------------------------------------
    # HANDLE IMBALANCE
    # --------------------------------------------------------
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / (pos + 1e-6)

    print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

    models = get_models(scale_pos_weight)

    results = []

    for name, model in models.items():
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    df_results = pd.DataFrame(results)
    print(df_results.sort_values(by="recall", ascending=False))


if __name__ == "__main__":
    main()