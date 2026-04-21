"""


OUTPUT:
    DRC_ML_Ready_v3.csv
"""

import pandas as pd
import numpy as np


# ============================================================
# TARGET: Tukey (per PROV × MALADIE)
# ============================================================

def tukey_outbreak(group):
    # q1 = group.quantile(0.25)
    # q3 = group.quantile(0.75)
    # iqr = q3 - q1
    # threshold = q3 + 1.5 * iqr
    # return (group > threshold).astype(int)
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1

        tukey = q3 + 1.5 * iqr
        p90   = group.quantile(0.90)

        # Hybrid threshold
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
    # 2. INCIDENCE (core epidemiological signal)
    # --------------------------------------------------------
    df['INCIDENCE'] = df['TOTALCAS'] / df['POP']

    # --------------------------------------------------------
    # 3. TARGET (Tukey per PROV × MALADIE)
    # --------------------------------------------------------
    df['IS_OUTBREAK'] = grp['INCIDENCE'].transform(tukey_outbreak)

    # --------------------------------------------------------
    # 4. TEMPORAL FEATURES (incidence)
    # --------------------------------------------------------
    for lag in [1, 2, 3, 4]:
        df[f'INC_LAG{lag}'] = grp['INCIDENCE'].shift(lag)

    inc_shift = grp['INCIDENCE'].shift(1)

    df['INC_ROLL4']  = inc_shift.transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['INC_ROLL8']  = inc_shift.transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['INC_ROLL12'] = inc_shift.transform(lambda x: x.rolling(12, min_periods=1).mean())

    df['INC_TREND'] = df['INC_LAG1'] - df['INC_ROLL4']

    # --------------------------------------------------------
    # 5. WEATHER LAGS
    # --------------------------------------------------------
    weather_cols = ['PRECTOTCORR', 'T2M', 'RH2M']

    for col in weather_cols:
        df[f'{col}_LAG1'] = grp[col].shift(1)
        df[f'{col}_LAG2'] = grp[col].shift(2)

    # --------------------------------------------------------
    # 6. CLIMATE ANOMALIES (VERY IMPORTANT)
    # --------------------------------------------------------
    for col in weather_cols:
        mean = grp[col].transform('mean')
        df[f'{col}_ANOM'] = df[col] - mean

    # --------------------------------------------------------
    # 7. CLIMATE INTERACTIONS
    # --------------------------------------------------------
    df['RAIN_TEMP'] = df['PRECTOTCORR'] * df['T2M']
    df['HUM_TEMP']  = df['RH2M'] * df['T2M']

    # --------------------------------------------------------
    # 8. VOLATILITY (instability signal)
    # --------------------------------------------------------
    df['TEMP_VOLATILITY'] = grp['T2M'].transform(lambda x: x.rolling(4).std())
    df['RAIN_VOLATILITY'] = grp['PRECTOTCORR'].transform(lambda x: x.rolling(4).std())

    # --------------------------------------------------------
    # 9. SEASONALITY
    # --------------------------------------------------------
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12)

    # --------------------------------------------------------
    # 10. DROP NA (due to lags)
    # --------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=['INC_LAG4']).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows due to lagging")

    return df


# ============================================================
# VALIDATION REPORT
# ============================================================

def report(df):

    print("\n" + "="*60)
    print("FINAL DATASET REPORT")
    print("="*60)

    print(f"Shape: {df.shape}")
    print(f"Years: {df['ANNEE'].min()} - {df['ANNEE'].max()}")
    print(f"Provinces: {df['PROV'].nunique()}")
    print(f"Diseases: {df['MALADIE'].nunique()}")

    rate = df['IS_OUTBREAK'].mean() * 100
    print(f"\nOutbreak rate: {rate:.2f}%")

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None ✓")

    print("="*60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    INPUT  = "DRC_Health_Weather_Master.csv"
    OUTPUT = "DRC_ML_Ready_v3.csv"

    df = feature_engineering(INPUT)
    report(df)

    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")