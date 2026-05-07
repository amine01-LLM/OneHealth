"""
FEATURE ENGINEERING v4 — TRUE EARLY WARNING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY UPGRADES:
✓ Predicts future outbreaks (t + H weeks)
✓ Removes leakage from current incidence
✓ Adds long weather lags (biological delays)
✓ Adds early signals (growth, acceleration, anomaly)
✓ Keeps epidemiologically meaningful features only

OUTPUT:
    DRC_ML_Ready_v4.csv
"""

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

HORIZON = 4  # Predict outbreak 4 weeks ahead


# ============================================================
# TARGET: Tukey (per PROV × MALADIE)
# ============================================================

def tukey_outbreak(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1

    tukey = q3 + 1.5 * iqr
    p90   = group.quantile(0.90)

    threshold = max(tukey, p90)
    return (group > threshold).astype(int)


# ============================================================
# MAIN PIPELINE
# ============================================================

def feature_engineering(input_file):

    print(f"Loading {input_file}")
    df = pd.read_csv(input_file)

    # --------------------------------------------------------
    # 1. SORT (CRITICAL)
    # --------------------------------------------------------
    df = df.sort_values(['PROV', 'MALADIE', 'DEBUTSEM']).reset_index(drop=True)
    grp = df.groupby(['PROV', 'MALADIE'])

    # --------------------------------------------------------
    # 2. INCIDENCE (BASE SIGNAL — NOT USED DIRECTLY)
    # --------------------------------------------------------
    df['INCIDENCE'] = df['TOTALCAS'] / df['POP']

    # --------------------------------------------------------
    # 3. CURRENT TARGET (for shifting only)
    # --------------------------------------------------------
    df['IS_OUTBREAK'] = grp['INCIDENCE'].transform(tukey_outbreak)

    # --------------------------------------------------------
    # 4. FUTURE TARGET (EARLY WARNING)
    # --------------------------------------------------------
    df['TARGET_OUTBREAK_FUTURE'] = (
        grp['IS_OUTBREAK'].shift(-HORIZON)
    )

    # --------------------------------------------------------
    # 5. EPIDEMIOLOGICAL LAGS (NO LEAKAGE)
    # --------------------------------------------------------
    for lag in [1, 2, 3, 4]:
        df[f'INC_LAG{lag}'] = grp['INCIDENCE'].shift(lag)

    # --------------------------------------------------------
    # 6. ROLLING BASELINES (context)
    # --------------------------------------------------------
    inc_shift = grp['INCIDENCE'].shift(1)

    df['INC_ROLL4']  = inc_shift.transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['INC_ROLL8']  = inc_shift.transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['INC_ROLL12'] = inc_shift.transform(lambda x: x.rolling(12, min_periods=1).mean())

    # --------------------------------------------------------
    # 7. EARLY SIGNAL FEATURES (VERY IMPORTANT)
    # --------------------------------------------------------

    # Acceleration (change over time)
    df['INC_ACCELERATION'] = df['INC_LAG1'] - df['INC_LAG3']

    # Growth rate
    df['INC_GROWTH'] = (
        (df['INC_LAG1'] - df['INC_LAG2']) /
        (df['INC_LAG2'] + 1e-6)
    )

    # Anomaly vs long-term baseline
    df['INC_ANOMALY'] = df['INC_LAG1'] - df['INC_ROLL12']

    # Trend
    df['INC_TREND'] = df['INC_LAG1'] - df['INC_ROLL4']

    # --------------------------------------------------------
    # 8. WEATHER LAGS (EXTENDED — KEY FIX)
    # --------------------------------------------------------
    weather_cols = ['PRECTOTCORR', 'T2M', 'RH2M']

    for col in weather_cols:
        for lag in [1, 2, 3, 4, 6, 8]:
            df[f'{col}_LAG{lag}'] = grp[col].shift(lag)

    # --------------------------------------------------------
    # 9. CLIMATE ANOMALIES
    # --------------------------------------------------------
    for col in weather_cols:
        mean = grp[col].transform('mean')
        df[f'{col}_ANOM'] = df[col] - mean

    # --------------------------------------------------------
    # 10. CLIMATE INTERACTIONS
    # --------------------------------------------------------
    # df['RAIN_TEMP'] = df['PRECTOTCORR'] * df['T2M']
    # df['HUM_TEMP']  = df['RH2M'] * df['T2M']

    # --------------------------------------------------------
    # 11. VOLATILITY
    # --------------------------------------------------------
    df['TEMP_VOLATILITY'] = grp['T2M'].transform(lambda x: x.rolling(4).std())
    df['RAIN_VOLATILITY'] = grp['PRECTOTCORR'].transform(lambda x: x.rolling(4).std())

    # --------------------------------------------------------
    # 12. SEASONALITY
    # --------------------------------------------------------
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12)

    # --------------------------------------------------------
    # 13. CLEANING
    # --------------------------------------------------------
    before = len(df)

    df = df.dropna(subset=[
        'INC_LAG4',
        'TARGET_OUTBREAK_FUTURE'
    ]).reset_index(drop=True)

    print(f"Dropped {before - len(df)} rows due to lagging & horizon shift")

    return df


# ============================================================
# VALIDATION REPORT
# ============================================================

def report(df):

    print("\n" + "="*60)
    print("FINAL DATASET REPORT — V4 (EARLY WARNING)")
    print("="*60)

    print(f"Shape: {df.shape}")
    print(f"Years: {df['ANNEE'].min()} - {df['ANNEE'].max()}")
    print(f"Provinces: {df['PROV'].nunique()}")
    print(f"Diseases: {df['MALADIE'].nunique()}")

    rate = df['TARGET_OUTBREAK_FUTURE'].mean() * 100
    print(f"\nFuture outbreak rate (t+{HORIZON}): {rate:.2f}%")

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None ✓")

    print("="*60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    INPUT  = "DRC_Health_Weather_Master.csv"
    OUTPUT = "DRC_ML_Ready_v4.csv"

    df = feature_engineering(INPUT)
    report(df)

    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")