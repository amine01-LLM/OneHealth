# ============================================================
# V10 — TRUE CLIMATE-DRIVEN EARLY WARNING MODEL
# ============================================================
# OBJECTIVE:
# Build a REAL early warning system using ONLY:
# - climate anomalies
# - climate volatility
# - selected temporal lags
# - geography
# - seasonality
#
# NO epidemiological leakage
#
# TARGET:
# TARGET_OUTBREAK_FUTURE
#
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "DRC_Health_Weather_Master.csv"

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# FILTER ROUGEOLE
# ============================================================

df = df[df['MALADIE'] == 'ROUGEOLE'].copy()

print(f"Rougeole dataset: {df.shape}")

# ============================================================
# CREATE MONTH CYCLICAL FEATURES
# ============================================================

if 'MOIS' in df.columns:

    df['MONTH_SIN'] = np.sin(
        2 * np.pi * df['MOIS'] / 12
    )

    df['MONTH_COS'] = np.cos(
        2 * np.pi * df['MOIS'] / 12
    )

# ============================================================
# CREATE VOLATILITY FEATURES
# ============================================================

# temperature volatility
df['TEMP_VOLATILITY'] = (
    df['T2M_LAG4'] - df['T2M']
).abs()

# rain volatility
df['RAIN_VOLATILITY'] = (
    df['PRECTOTCORR_LAG4'] - df['PRECTOTCORR']
).abs()

# humidity volatility
df['RH_VOLATILITY'] = (
    df['RH2M_LAG4'] - df['RH2M']
).abs()

# ============================================================
# TARGET
# ============================================================

TARGET = 'TARGET_OUTBREAK_FUTURE'

# ============================================================
# FEATURES (NO LEAKAGE)
# ============================================================

FEATURES = [

    # --------------------------------------------------------
    # Geography
    # --------------------------------------------------------
    'LAT',
    'LON',
    'POP',

    # --------------------------------------------------------
    # Raw climate
    # --------------------------------------------------------
    'T2M',
    'RH2M',
    'PRECTOTCORR',

    # --------------------------------------------------------
    # Selected lags
    # --------------------------------------------------------
    'T2M_LAG4',
    'RH2M_LAG4',
    'PRECTOTCORR_LAG4',

    # --------------------------------------------------------
    # Climate anomalies
    # --------------------------------------------------------
    'T2M_ANOM',
    'RH2M_ANOM',
    'PRECTOTCORR_ANOM',

    # --------------------------------------------------------
    # Climate volatility
    # --------------------------------------------------------
    'TEMP_VOLATILITY',
    'RAIN_VOLATILITY',
    'RH_VOLATILITY',

    # --------------------------------------------------------
    # Seasonal encoding
    # --------------------------------------------------------
    'MONTH_SIN',
    'MONTH_COS'
]

# keep only existing columns
FEATURES = [f for f in FEATURES if f in df.columns]

print("\nSelected features:")
for f in FEATURES:
    print(f"- {f}")

# ============================================================
# REMOVE NaNs
# ============================================================

df = df.dropna(subset=FEATURES + [TARGET])

print(f"\nDataset after NaN removal: {df.shape}")

# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# TEMPORAL SPLIT
# ============================================================

# sort by time if available
sort_cols = []

if 'ANNEE' in df.columns:
    sort_cols.append('ANNEE')

if 'MOIS' in df.columns:
    sort_cols.append('MOIS')

if len(sort_cols) > 0:
    df = df.sort_values(sort_cols)

X = df[FEATURES]
y = df[TARGET]

split_index = int(len(df) * 0.70)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain/Test shapes")
print(X_train.shape)
print(X_test.shape)

# ============================================================
# CLASS IMBALANCE
# ============================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print(f"\nScale pos weight: {scale_pos_weight:.2f}")

# ============================================================
# LIGHTGBM MODEL
# ============================================================

model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

# ============================================================
# GRID SEARCH
# ============================================================

param_grid = {

    'n_estimators': [100, 200],

    'learning_rate': [0.03, 0.05],

    'max_depth': [3, 5],

    'num_leaves': [15, 31],

    'subsample': [0.8],

    'colsample_bytree': [0.8]
}

print("\n" + "=" * 60)
print("GRID SEARCH")
print("=" * 60)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# ============================================================
# BEST MODEL
# ============================================================

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(f"\nBest CV F1: {grid.best_score_:.4f}")

# ============================================================
# PREDICTIONS
# ============================================================

y_proba = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

print(f"AUC: {auc:.4f}")

# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

best_f1 = 0
best_threshold = 0.5

for t in thresholds:

    y_pred = (y_proba >= t).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
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

# ============================================================
# FINAL THRESHOLD
# ============================================================

print(f"\nBest threshold: {best_threshold}")

y_pred_final = (
    y_proba >= best_threshold
).astype(int)

# ============================================================
# FINAL METRICS
# ============================================================

cm = confusion_matrix(y_test, y_pred_final)

precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_final
    )
)

print("\nFINAL METRICS")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    'Feature': FEATURES,
    'Importance': best_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTOP FEATURES")
print("=" * 60)

print(importance_df.head(20))

# ============================================================
# SAVE RESULTS
# ============================================================

importance_df.to_csv(
    "V10_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("- V10_feature_importance.csv")

# ============================================================
# END
# ============================================================