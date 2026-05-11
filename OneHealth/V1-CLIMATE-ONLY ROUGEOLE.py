# ============================================================
# V11 — CLIMATE-ONLY ROUGEOLE EARLY WARNING MODEL
# ============================================================
# Goal:
# Predict future outbreaks WITHOUT using current epidemiological
# information such as current cases or incidence.
#
# This creates a TRUE early-warning system based on:
# - Weather
# - Geography
# - Population
# - Seasonality
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

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "DRC_ML_Ready_v4.csv"

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded dataset: {df.shape}")


# ============================================================
# FILTER ROUGEOLE ONLY
# ============================================================

df = df[df["MALADIE"] == "ROUGEOLE"].copy()

print(f"Rougeole samples: {len(df)}")


# ============================================================
# DATE PROCESSING
# ============================================================

df["DEBUTSEM"] = pd.to_datetime(df["DEBUTSEM"])

df = df.sort_values(
    ["PROV", "DEBUTSEM"]
).reset_index(drop=True)


# ============================================================
# CREATE TARGET
# ============================================================

# Predict outbreak in future weeks

PREDICT_WEEKS_AHEAD = 2

df["TARGET_OUTBREAK_FUTURE"] = (
    df.groupby(["PROV"])["IS_OUTBREAK"]
    .shift(-PREDICT_WEEKS_AHEAD)
)

df = df.dropna(subset=["TARGET_OUTBREAK_FUTURE"])

df["TARGET_OUTBREAK_FUTURE"] = (
    df["TARGET_OUTBREAK_FUTURE"]
    .astype(int)
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("\nAdding temporal features...")


# WEEK
df["WEEK"] = df["DEBUTSEM"].dt.isocalendar().week.astype(int)

# MONTH
df["MONTH"] = df["DEBUTSEM"].dt.month


# Cyclical encoding
df["SIN_WEEK"] = np.sin(
    2 * np.pi * df["WEEK"] / 52
)

df["COS_WEEK"] = np.cos(
    2 * np.pi * df["WEEK"] / 52
)

df["SIN_MONTH"] = np.sin(
    2 * np.pi * df["MONTH"] / 12
)

df["COS_MONTH"] = np.cos(
    2 * np.pi * df["MONTH"] / 12
)


# ============================================================
# WEATHER TEMPORAL FEATURES
# ============================================================

WEATHER_COLS = [

    "PRECTOTCORR",
    "T2M",
    "RH2M",
    "WS2M"

]

for col in WEATHER_COLS:

    if col not in df.columns:
        continue

    # LAGS
    for lag in [1, 2, 4, 6]:

        df[f"{col}_LAG{lag}"] = (
            df.groupby(["PROV"])[col]
            .shift(lag)
        )

    # ROLLING MEAN
    df[f"{col}_ROLL4"] = (
        df.groupby(["PROV"])[col]
        .transform(
            lambda x: x.rolling(4).mean()
        )
    )

    # ANOMALY
    df[f"{col}_ANOM"] = (
        df[col]
        - df.groupby("MONTH")[col]
        .transform("mean")
    )


# ============================================================
# REMOVE NaNs
# ============================================================

before = len(df)

df = df.dropna()
df["PROV"] = df["PROV"].astype("category").cat.codes

after = len(df)

print(f"\nDropped {before - after} rows due to NaNs")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Temporal split

split_date = df["DEBUTSEM"].quantile(0.80)

train_df = df[df["DEBUTSEM"] <= split_date].copy()
test_df  = df[df["DEBUTSEM"] > split_date].copy()

print(f"\nTrain size: {train_df.shape}")
print(f"Test size : {test_df.shape}")


# ============================================================
# REMOVE EPIDEMIOLOGICAL FEATURES
# ============================================================

REMOVE_FEATURES = [

    # identifiers
    "DEBUTSEM",
    "MALADIE",

    # targets
    "IS_OUTBREAK",
    "TARGET_OUTBREAK_FUTURE",

    # direct epidemiology
    "TOTALCAS",
    "TOTALDECES",
    "INCIDENCE",

    # incidence lags
    "INC_LAG1",
    "INC_LAG2",
    "INC_LAG3",
    "INC_LAG4",
    "INC_LAG5",
    "INC_LAG6",
    "INC_LAG7",
    "INC_LAG8",

    # rolling incidence
    "INC_ROLL2",
    "INC_ROLL4",
    "INC_ROLL8",
    "INC_ROLL12",

    "ROLL_MEAN_2",
    "ROLL_MEAN_4",
    "ROLL_MEAN_8",

    # trends
    "INC_TREND",
    "INC_GROWTH",
    "INC_ACCELERATION",

    # anomalies
    "INC_ANOMALY"

]


FEATURES = [

    col for col in train_df.columns
    if col not in REMOVE_FEATURES

]


X_train = train_df[FEATURES]
y_train = train_df["TARGET_OUTBREAK_FUTURE"]

X_test = test_df[FEATURES]
y_test = test_df["TARGET_OUTBREAK_FUTURE"]


print(f"\nFeatures used: {len(FEATURES)}")


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


print("\nRunning GridSearchCV...\n")


grid = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    scoring='f1',

    cv=3,

    verbose=1,

    n_jobs=-1
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

print("\n============================================================")
print("FINAL EVALUATION")
print("============================================================")

print(f"AUC: {auc:.4f}")


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\n============================================================")
print("THRESHOLD ANALYSIS")
print("============================================================")


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


print(f"\nBest threshold: {best_threshold}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

y_pred = (
    y_proba >= best_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nFINAL METRICS")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": best_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP FEATURES")
print("============================================================")

print(
    importance_df.head(20)
)