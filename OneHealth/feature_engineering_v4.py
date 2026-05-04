"""
feature_engineering_v4.py
─────────────────────────
Changes vs v3:
  • Incidence lags extended to 8 weeks  (INC_LAG1 … INC_LAG8)
  • Weather lags extended to 8 weeks    (PRECTOTCORR/T2M/RH2M _LAG1 … _LAG8)
  • Rolling windows extended to 16 wks  (INC_ROLL4/8/12/16)
  • INCIDENCE itself is kept for target computation and lags but is
    explicitly tagged so the model pipeline can drop it from X.
  • dropna now requires INC_LAG8 (the longest lag).

OUTPUT:
    DRC_ML_Ready_v4.csv
"""

import pandas as pd
import numpy as np


# ============================================================
# TARGET: Hybrid Tukey + P90 (per PROV × MALADIE)
# ============================================================

def tukey_outbreak(group):
    q1    = group.quantile(0.25)
    q3    = group.quantile(0.75)
    iqr   = q3 - q1
    tukey = q3 + 1.5 * iqr
    p90   = group.quantile(0.90)
    # Use the stricter of the two so we don't inflate outbreak rate
    threshold = max(tukey, p90)
    return (group > threshold).astype(int)


# ============================================================
# MAIN PIPELINE
# ============================================================

def feature_engineering(input_file):

    print(f"Loading {input_file}")
    df = pd.read_csv(input_file)

    # --------------------------------------------------------
    # 1. SORT (CRITICAL for correct lag alignment)
    # --------------------------------------------------------
    df = df.sort_values(['PROV', 'MALADIE', 'DEBUTSEM']).reset_index(drop=True)

    grp = df.groupby(['PROV', 'MALADIE'])

    # --------------------------------------------------------
    # 2. INCIDENCE  — used for target & lags; NOT a raw feature
    #    (the model pipeline must exclude it from X)
    # --------------------------------------------------------
    df['INCIDENCE'] = df['TOTALCAS'] / df['POP']

    # --------------------------------------------------------
    # 3. TARGET (Hybrid Tukey+P90 per PROV × MALADIE on INCIDENCE)
    # --------------------------------------------------------
    df['IS_OUTBREAK'] = grp['INCIDENCE'].transform(tukey_outbreak)

    # --------------------------------------------------------
    # 4. INCIDENCE LAGS  — 1 to 8 weeks
    # --------------------------------------------------------
    for lag in range(1, 9):          # 1,2,3,4,5,6,7,8
        df[f'INC_LAG{lag}'] = grp['INCIDENCE'].shift(lag)

    # --------------------------------------------------------
    # 5. ROLLING MEANS OF INCIDENCE  — 4 / 8 / 12 / 16 weeks
    #    (always based on lag-1 shift to avoid data leakage)
    # --------------------------------------------------------
    inc_shift = grp['INCIDENCE'].shift(1)

    for window in [4, 8, 12, 16]:
        df[f'INC_ROLL{window}'] = inc_shift.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # --------------------------------------------------------
    # 6. INCIDENCE TREND  (short-term momentum)
    # --------------------------------------------------------
    df['INC_TREND']      = df['INC_LAG1'] - df['INC_ROLL4']
    df['INC_TREND_LONG'] = df['INC_LAG1'] - df['INC_ROLL12']   # new in v4

    # --------------------------------------------------------
    # 7. WEATHER LAGS  — 1 to 8 weeks  (extended from 1-2 in v3)
    # --------------------------------------------------------
    weather_cols = ['PRECTOTCORR', 'T2M', 'RH2M']

    for col in weather_cols:
        for lag in range(1, 9):
            df[f'{col}_LAG{lag}'] = grp[col].shift(lag)

    # --------------------------------------------------------
    # 8. CLIMATE ANOMALIES  (deviation from historical mean)
    # --------------------------------------------------------
    for col in weather_cols:
        historical_mean = grp[col].transform('mean')
        df[f'{col}_ANOM'] = df[col] - historical_mean

    # --------------------------------------------------------
    # 9. CLIMATE INTERACTIONS
    # --------------------------------------------------------
    df['RAIN_TEMP'] = df['PRECTOTCORR'] * df['T2M']
    df['HUM_TEMP']  = df['RH2M']        * df['T2M']

    # --------------------------------------------------------
    # 10. CLIMATE VOLATILITY  (4-week rolling std)
    # --------------------------------------------------------
    df['TEMP_VOLATILITY'] = grp['T2M'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )
    df['RAIN_VOLATILITY'] = grp['PRECTOTCORR'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )

    # --------------------------------------------------------
    # 11. SEASONALITY  (cyclic encoding)
    # --------------------------------------------------------
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12)

    # --------------------------------------------------------
    # 12. DROP NA  (rows that cannot have full 8-week lag history)
    # --------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=['INC_LAG8']).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} rows due to 8-week lag requirement")

    return df


# ============================================================
# VALIDATION REPORT
# ============================================================

def report(df):

    print("\n" + "=" * 60)
    print("FINAL DATASET REPORT  (v4)")
    print("=" * 60)

    print(f"Shape            : {df.shape}")
    print(f"Years            : {df['ANNEE'].min()} – {df['ANNEE'].max()}")
    print(f"Provinces        : {df['PROV'].nunique()}")
    print(f"Diseases         : {df['MALADIE'].nunique()}")

    rate = df['IS_OUTBREAK'].mean() * 100
    print(f"Outbreak rate    : {rate:.2f}%")

    lag_cols = [c for c in df.columns if 'LAG' in c or 'ROLL' in c]
    print(f"Lag/rolling cols : {len(lag_cols)}  {lag_cols}")

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None ✓")

    print("=" * 60)


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
