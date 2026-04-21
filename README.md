# 🌍 OneHealth DRC — Epidemic Early-Warning System

A machine learning pipeline that predicts disease outbreaks across the Democratic Republic of Congo by combining epidemiological surveillance data with satellite-derived climate variables. Built on the **One Health** framework, which recognizes the interconnection between human health and environmental conditions.

---

## 📋 Overview

| | |
|---|---|
| **Country** | Democratic Republic of Congo (DRC) |
| **Period** | 2006 – 2017 |
| **Provinces** | 29 |
| **Diseases** | 19 |
| **Records** | ~36,700 (ML-ready dataset) |
| **Task** | Binary outbreak classification (`IS_OUTBREAK`) |
| **Outbreak rate** | ~6% of records |

---

## 🗂️ Repository Structure

```
Git_OneHealth/
│
├── weather_fetcher.py           # Step 1 — Fetch climate data from NASA POWER API
├── feature_engineering_v3.py   # Step 2 — Build ML-ready features from master dataset
├── compare_multiple_models_v2.py# Step 3 — Train & evaluate 6 ML classifiers
│
├── DRC_Health_Final.csv         # Raw health surveillance data (cases, deaths, population)
├── DRC_Health_Weather_Master.csv# Health data merged with climate variables
└── DRC_ML_Ready_v3.csv          # Final feature-engineered dataset (model input)
```

---

## ⚙️ Pipeline

### Step 1 — `weather_fetcher.py`
Fetches monthly climate data from the **[NASA POWER API](https://power.larc.nasa.gov/)** for each province's GPS coordinates and merges it with the health records.

**Climate variables collected:**
- `PRECTOTCORR` — Precipitation (mm/day)
- `T2M` — Temperature at 2 metres (°C)
- `RH2M` — Relative humidity at 2 metres (%)

**Output:** `DRC_Health_Weather_Master.csv`

---

### Step 2 — `feature_engineering_v3.py`
Transforms the master dataset into a rich feature matrix for ML. Features are engineered per `PROVINCE × DISEASE` group to avoid data leakage across regions or diseases.

| Feature Group | Features | Description |
|---|---|---|
| **Incidence** | `INCIDENCE` | Cases / Population |
| **Outbreak label** | `IS_OUTBREAK` | Hybrid Tukey + P90 threshold per PROV × DISEASE |
| **Incidence lags** | `INC_LAG1..4` | Incidence from 1–4 periods back |
| **Rolling averages** | `INC_ROLL4/8/12` | 4, 8, 12-week rolling mean of incidence |
| **Trend** | `INC_TREND` | `INC_LAG1 − INC_ROLL4` |
| **Weather lags** | `*_LAG1/2` | Precipitation, temperature & humidity lagged 1–2 periods |
| **Climate anomalies** | `*_ANOM` | Deviation from historical province-level mean |
| **Climate interactions** | `RAIN_TEMP`, `HUM_TEMP` | Multiplicative interaction terms |
| **Volatility** | `TEMP_VOLATILITY`, `RAIN_VOLATILITY` | 4-period rolling std |
| **Seasonality** | `MONTH_SIN`, `MONTH_COS` | Cyclic encoding of month |

**Outbreak labeling — Hybrid Tukey/P90:**
```
threshold = max(Q3 + 1.5 × IQR,  P90)
IS_OUTBREAK = 1  if  INCIDENCE > threshold
```

**Output:** `DRC_ML_Ready_v3.csv` (36,727 rows × 38 columns)

---

### Step 3 — `compare_multiple_models_v2.py`
Trains and evaluates 6 classifiers with a **temporal train/test split** to simulate real-world prospective prediction.

| Split | Years | Usage |
|---|---|---|
| Train | 2006 – 2014 | Model fitting |
| Test | 2015 – 2017 | Evaluation |

**Models trained:**
- XGBoost (with `scale_pos_weight` for class imbalance)
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM
- Neural Network (MLP)

**Evaluation metrics:** AUC-ROC, Recall, Precision, F1-Score, Accuracy

**Outputs:** `DRC_Model_Comparison_v2.png`, `DRC_Feature_Importance_v2.png`, `model_comparison_v2.csv`

---

## 🦠 Diseases Covered

`CHOLERA` · `PALUDISME` (Malaria) · `ROUGEOLE` (Measles) · `MENINGITE` · `FIEVRE JAUNE` (Yellow Fever) · `FIEVRE TYPHOIDE` · `MONKEY POX` · `IRA` · `COQUELUCHE` · `DIARR SANGLANTE` · `FHA` · `FHV` · `RAGE` · `PESTE` · `PFA` · `TNN` · `DRACUNCULOSE` · `DECES MATERNEL` · `DECES MATERNELS`

---

## 🗺️ Provinces Covered

29 provinces including: `KINSHASA` · `NORD-KIVU` · `SUD-KIVU` · `KATANGA` · `EQUATEUR` · `ORIENTALE` · `BANDUNDU` · `KASAI` · `MANIEMA` · `ITURI` and more.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib requests
```

### Run the full pipeline

```bash
# 1. Fetch weather data (requires internet access to NASA POWER API)
python weather_fetcher.py

# 2. Engineer features
python feature_engineering_v3.py

# 3. Train models and generate comparison report
python compare_multiple_models_v2.py
```

> **Note:** If `DRC_Health_Weather_Master.csv` already exists, you can skip Step 1 and go straight to Step 2.

---

## 📊 Key Design Choices

**Why One Health?** Infectious disease dynamics in the DRC are strongly shaped by environmental variables — rainfall drives cholera and malaria, temperature anomalies affect vector populations. Integrating climate data alongside case counts substantially improves predictive power.

**Why a temporal split?** A random train/test split would leak future information. Splitting at 2014/2015 reflects how a real early-warning system would be deployed: trained on historical data, evaluated on genuinely unseen future periods.

**Why the hybrid Tukey/P90 threshold?** Standard Tukey fences can be too permissive for highly skewed epidemiological data. The hybrid threshold `max(Q3 + 1.5×IQR, P90)` ensures that outbreaks are flagged only when incidence is both statistically extreme *and* in the top decile, reducing label noise.

---

## 📁 Data Sources

- **Health surveillance:** DRC national epidemiological surveillance records (weekly case & death counts per province and disease)
- **Climate data:** [NASA POWER](https://power.larc.nasa.gov/) — satellite-derived monthly climate parameters at province centroids

---

## 📄 License

This project is for research and public health purposes. Please cite appropriately if you use this pipeline or dataset in your work.
