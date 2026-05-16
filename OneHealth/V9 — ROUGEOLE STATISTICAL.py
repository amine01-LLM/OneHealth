# ============================================================
# V9 — ROUGEOLE STATISTICAL VALIDATION PIPELINE
# ============================================================
# OBJECTIVES:
# 1. Correlation Matrix Analysis
# 2. Mutual Information Feature Selection
# 3. Statistical Significance Tests
# 4. Class Imbalance Analysis
#
# OUTPUTS:
# - Correlation heatmap
# - Highly correlated features
# - MI feature ranking
# - Mann-Whitney significance tests
# - Imbalance statistics
# - CSV exports
#
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "DRC_ML_Ready_v4.csv"

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# FILTER ROUGEOLE ONLY
# ============================================================

df = df[df['MALADIE'] == 'ROUGEOLE'].copy()

print(f"Rougeole dataset: {df.shape}")

# ============================================================
# REMOVE LEAKAGE FEATURES
# ============================================================

LEAKAGE_FEATURES = [
    'TOTALCAS',
    'TOTALDECES',
    'INCIDENCE',
    'INC_LAG1',
    'INC_LAG2',
    'INC_LAG3',
    'INC_LAG4',
    'INC_ROLL4',
    'INC_ROLL8',
    'INC_ROLL12',
    'INC_ACCELERATION',
    'INC_TREND',
    'INC_ANOMALY'
]

existing_leaks = [c for c in LEAKAGE_FEATURES if c in df.columns]

df.drop(columns=existing_leaks, inplace=True)

print("\nRemoved leakage features:")
for c in existing_leaks:
    print(f"- {c}")

# ============================================================
# TARGET
# ============================================================

TARGET = 'TARGET_OUTBREAK_FUTURE'

# ============================================================
# KEEP NUMERIC FEATURES ONLY
# ============================================================

numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

# remove target temporarily
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

# remove IDs if exist
REMOVE_COLS = [
    'YEARWEEK',
    'ANNEE'
]

numeric_cols = [
    c for c in numeric_cols
    if c not in REMOVE_COLS
]

print("\nNumeric features:")
print(len(numeric_cols))

# ============================================================
# DROP NaNs
# ============================================================

df_clean = df[numeric_cols + [TARGET]].dropna()

print(f"\nAfter NaN removal: {df_clean.shape}")

# ============================================================
# 1. CORRELATION MATRIX ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("1. CORRELATION MATRIX ANALYSIS")
print("=" * 60)

corr_matrix = df_clean[numeric_cols].corr()

# ------------------------------------------------------------
# SAVE HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(18, 14))

sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Saved: correlation_heatmap.png")

# ------------------------------------------------------------
# FIND HIGHLY CORRELATED FEATURES
# ------------------------------------------------------------

high_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):

        corr_value = corr_matrix.iloc[i, j]

        if abs(corr_value) > 0.90:

            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]

            high_corr.append([
                feature_1,
                feature_2,
                corr_value
            ])

high_corr_df = pd.DataFrame(
    high_corr,
    columns=['Feature_1', 'Feature_2', 'Correlation']
)

high_corr_df.to_csv(
    "high_correlation_features.csv",
    index=False
)

print("\nHighly correlated features:")
print(high_corr_df.head(20))

# ============================================================
# 2. MUTUAL INFORMATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("2. MUTUAL INFORMATION ANALYSIS")
print("=" * 60)

X = df_clean[numeric_cols]
y = df_clean[TARGET]

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    'Feature': numeric_cols,
    'MI_Score': mi_scores
})

mi_df = mi_df.sort_values(
    by='MI_Score',
    ascending=False
)

mi_df.to_csv(
    "mutual_information_scores.csv",
    index=False
)

print("\nTop Mutual Information Features:")
print(mi_df.head(20))

# ------------------------------------------------------------
# PLOT MI SCORES
# ------------------------------------------------------------

top_mi = mi_df.head(20)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top_mi,
    y='Feature',
    x='MI_Score'
)

plt.title("Top Mutual Information Features")

plt.tight_layout()

plt.savefig(
    "mutual_information_ranking.png",
    dpi=300
)

plt.close()

print("\nSaved: mutual_information_ranking.png")

# ============================================================
# 3. STATISTICAL SIGNIFICANCE TESTS
# ============================================================

print("\n" + "=" * 60)
print("3. STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

outbreak_df = df_clean[df_clean[TARGET] == 1]
normal_df = df_clean[df_clean[TARGET] == 0]

stat_results = []

for feature in numeric_cols:

    try:

        outbreak_values = outbreak_df[feature]
        normal_values = normal_df[feature]

        # Mann-Whitney U test
        stat, p_value = mannwhitneyu(
            outbreak_values,
            normal_values,
            alternative='two-sided'
        )

        significance = (
            "SIGNIFICANT"
            if p_value < 0.05
            else "NOT_SIGNIFICANT"
        )

        stat_results.append([
            feature,
            p_value,
            significance,
            outbreak_values.mean(),
            normal_values.mean()
        ])

    except:
        pass

stats_df = pd.DataFrame(
    stat_results,
    columns=[
        'Feature',
        'P_Value',
        'Result',
        'Outbreak_Mean',
        'Normal_Mean'
    ]
)

stats_df = stats_df.sort_values(
    by='P_Value'
)

stats_df.to_csv(
    "statistical_significance_tests.csv",
    index=False
)

print("\nTop Significant Features:")
print(stats_df.head(20))

# ============================================================
# 4. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("4. CLASS IMBALANCE ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# GLOBAL IMBALANCE
# ------------------------------------------------------------

class_counts = df_clean[TARGET].value_counts()

print("\nClass distribution:")
print(class_counts)

positive_ratio = (
    class_counts[1] / len(df_clean)
)

print(f"\nPositive ratio: {positive_ratio:.4f}")

# ------------------------------------------------------------
# PER YEAR ANALYSIS
# ------------------------------------------------------------

if 'ANNEE' in df.columns:

    yearly = df.groupby('ANNEE')[TARGET].mean()

    plt.figure(figsize=(10, 5))

    yearly.plot(marker='o')

    plt.title("Outbreak Ratio Per Year")
    plt.ylabel("Outbreak Ratio")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "yearly_outbreak_ratio.png",
        dpi=300
    )

    plt.close()

    print("\nSaved: yearly_outbreak_ratio.png")

# ------------------------------------------------------------
# PER PROVINCE ANALYSIS
# ------------------------------------------------------------

if 'PROV' in df.columns:

    province_ratio = (
        df.groupby('PROV')[TARGET]
        .mean()
        .sort_values(ascending=False)
    )

    province_ratio.to_csv(
        "province_outbreak_ratio.csv"
    )

    plt.figure(figsize=(12, 6))

    province_ratio.plot(kind='bar')

    plt.title("Outbreak Ratio Per Province")

    plt.ylabel("Outbreak Ratio")

    plt.tight_layout()

    plt.savefig(
        "province_outbreak_ratio.png",
        dpi=300
    )

    plt.close()

    print("Saved: province_outbreak_ratio.png")

# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE RECOMMENDATIONS")
print("=" * 60)

# ------------------------------------------------------------
# HIGH MI FEATURES
# ------------------------------------------------------------

good_features = mi_df[
    mi_df['MI_Score'] > 0.01
]['Feature'].tolist()

print("\nRecommended features from MI:")
for f in good_features:
    print(f"- {f}")

# ------------------------------------------------------------
# SIGNIFICANT FEATURES
# ------------------------------------------------------------

significant_features = stats_df[
    stats_df['P_Value'] < 0.05
]['Feature'].tolist()

print("\nStatistically significant features:")
for f in significant_features:
    print(f"- {f}")

# ------------------------------------------------------------
# FINAL INTERSECTION
# ------------------------------------------------------------

final_features = list(
    set(good_features)
    &
    set(significant_features)
)

print("\nFINAL RECOMMENDED FEATURES:")
for f in final_features:
    print(f"- {f}")

# save final feature list
pd.DataFrame({
    'Selected_Features': final_features
}).to_csv(
    "final_selected_features.csv",
    index=False
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("V9 ANALYSIS COMPLETE")
print("=" * 60)

print("\nGenerated files:")
print("- correlation_heatmap.png")
print("- high_correlation_features.csv")
print("- mutual_information_scores.csv")
print("- mutual_information_ranking.png")
print("- statistical_significance_tests.csv")
print("- province_outbreak_ratio.csv")
print("- province_outbreak_ratio.png")
print("- yearly_outbreak_ratio.png")
print("- final_selected_features.csv")