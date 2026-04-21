

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    recall_score, precision_score, accuracy_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-MODEL COMPARISON v2 — DRC EPIDEMIC WARNING SYSTEM")
print("=" * 80)
print()


# ============================================================================
# 1. LOAD AND BUILD ENRICHED FEATURES
# ============================================================================

print("1. Loading data and building enriched features...")

# Accept either the pre-built v2 file or the base file
import os
if os.path.exists('DRC_ML_Ready_v3.csv'):
    df = pd.read_csv('DRC_ML_Ready_v3.csv')
    print("  Loaded DRC_ML_Ready_v3.csv")
else:
    print("  DRC_ML_Ready_v2.csv not found — computing features from base file")
    df = pd.read_csv('DRC_General_Risk_Updated.csv')

    # ── Target ──────────────────────────────────────────────────────────
    def tukey_outbreak_threshold(group):
        # q1    = group.quantile(0.25)
        # q3    = group.quantile(0.75)
        # fence = q3 + 1.5 * (q3 - q1)
        # return (group > fence).astype(int)
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1

        tukey = q3 + 1.5 * iqr
        p90   = group.quantile(0.90)

        # Hybrid threshold
        threshold = max(tukey, p90)

        return (group > threshold).astype(int)

    df['IS_OUTBREAK'] = (
        df.groupby('MALADIE')['TOTALCAS']
          .transform(tukey_outbreak_threshold)
    )

    df = df.sort_values(['PROV', 'MALADIE', 'DEBUTSEM']).reset_index(drop=True)
    grp_pd = df.groupby(['PROV', 'MALADIE'])

    # ── Weather lags ────────────────────────────────────────────────────
    for col in ['PRECTOTCORR', 'T2M', 'RH2M']:
        df[f'{col}_LAG1'] = grp_pd[col].shift(1)
        df[f'{col}_LAG2'] = grp_pd[col].shift(2)

    # ── Case lags ───────────────────────────────────────────────────────
    for lag in [1, 2, 3, 4]:
        df[f'CASES_LAG{lag}'] = grp_pd['TOTALCAS'].shift(lag)

    # ── Rolling means ───────────────────────────────────────────────────
    shifted = grp_pd['TOTALCAS'].shift(1)
    df['CASES_ROLL4']  = shifted.transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['CASES_ROLL8']  = shifted.transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['CASES_ROLL12'] = shifted.transform(lambda x: x.rolling(12, min_periods=1).mean())

    # ── Trend ────────────────────────────────────────────────────────────
    df['CASES_TREND'] = df['CASES_LAG1'] - df['CASES_ROLL4']

    # ── Autoregressive outbreak flags ────────────────────────────────────
    grp_pm = df.groupby(['PROV', 'MALADIE'])
    df['OUTBREAK_LAG1'] = grp_pm['IS_OUTBREAK'].shift(1)
    df['OUTBREAK_LAG2'] = grp_pm['IS_OUTBREAK'].shift(2)

    # ── Cyclic month ─────────────────────────────────────────────────────
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12)

    df = df.dropna(subset=['CASES_LAG4', 'OUTBREAK_LAG2']).reset_index(drop=True)

print(f"  {len(df):,} rows after feature engineering")
print(f"  Overall outbreak rate: {df['IS_OUTBREAK'].mean()*100:.2f}%")
print()

# ── Disease filter: keep only diseases present in BOTH train and test ────────
pre  = set(df[df['ANNEE'] <= 2014]['MALADIE'].unique())
post = set(df[df['ANNEE'] >  2014]['MALADIE'].unique())
consistent_diseases = sorted(pre & post)

print(f"  Diseases in both train and test periods: {consistent_diseases}")
excluded = sorted(set(df['MALADIE'].unique()) - set(consistent_diseases))
print(f"  Excluded (only in one period): {excluded}")
print()

df_m = df[df['MALADIE'].isin(consistent_diseases)].copy()
print(f"  Dataset after disease filter: {len(df_m):,} rows")
print()

# ── Features ─────────────────────────────────────────────────────────────────
EXCLUDE = {
    'IS_OUTBREAK', 'IS_OUTBREAK_NEW', 'IS_OUTBREAK_OLD',
    'TARGET',
    'TOTALCAS',    # outcome — must not be a feature
    'TOTALDECES',  # outcome
    'DEBUTSEM',    # raw date string
}
feature_cols = [c for c in df_m.columns if c not in EXCLUDE]

# X = df_m[feature_cols].copy()
# y = df_m['IS_OUTBREAK'].copy()

# le_prov = LabelEncoder()
# le_mal  = LabelEncoder()
# X['PROV']    = le_prov.fit_transform(X['PROV'])
# X['MALADIE'] = le_mal.fit_transform(X['MALADIE'])

train_mask = df_m['ANNEE'] <= 2014
test_mask  = df_m['ANNEE'] >= 2015
train_diseases = set(df_m.loc[train_mask, "MALADIE"].unique())
test_diseases  = set(df_m.loc[test_mask,  "MALADIE"].unique())

common_diseases = train_diseases.intersection(test_diseases)

print("Common diseases:", common_diseases)
print("Kept:", len(common_diseases), "diseases")
df_m = df_m[df_m["MALADIE"].isin(common_diseases)].copy()
# FINAL dataset after all filtering
X = df_m[feature_cols].copy()
y = df_m["IS_OUTBREAK"].copy()

# Encode categorical variables
le_prov = LabelEncoder()
le_mal  = LabelEncoder()

X['PROV']    = le_prov.fit_transform(X['PROV'])
X['MALADIE'] = le_mal.fit_transform(X['MALADIE'])



# ── Train / test split ────────────────────────────────────────────────────────
# 2006–2014 → train  |  2015–2017 → test
train_mask = df_m['ANNEE'] <= 2014
X_train, X_test = X[train_mask],  X[~train_mask]
y_train, y_test = y[train_mask],  y[~train_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"  Train (2006–2014): {len(X_train):,} rows | "
      f"outbreak rate: {y_train.mean()*100:.2f}%")
print(f"  Test  (2015–2017): {len(X_test):,} rows | "
      f"outbreak rate: {y_test.mean()*100:.2f}%")
print(f"  Class imbalance ratio: {pos_weight:.1f}x")
print(f"  Feature count: {len(feature_cols)}")
print()

# ============================================================================
# 2. TRAIN MODELS
# ============================================================================

print("2. Training models...")
print()

models     = {}
use_scaled = {}

print("   Training XGBoost...")
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss'
)
models['XGBoost'].fit(X_train, y_train)
use_scaled['XGBoost'] = False

print("   Training Random Forest...")
models['Random Forest'] = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
models['Random Forest'].fit(X_train, y_train)
use_scaled['Random Forest'] = False

print("   Training Gradient Boosting...")
models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42
)
models['Gradient Boosting'].fit(X_train, y_train)
use_scaled['Gradient Boosting'] = False

print("   Training Logistic Regression...")
models['Logistic Regression'] = LogisticRegression(
    max_iter=1000, class_weight='balanced', C=0.1,
    random_state=42, n_jobs=-1
)
models['Logistic Regression'].fit(X_train_scaled, y_train)
use_scaled['Logistic Regression'] = True

print("   Training SVM...")
models['SVM'] = SVC(
    kernel='rbf', C=1.0, class_weight='balanced',
    probability=True, random_state=42
)
models['SVM'].fit(X_train_scaled, y_train)
use_scaled['SVM'] = True

print("   Training Neural Network...")
models['Neural Network'] = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), max_iter=500,
    random_state=42, early_stopping=True, validation_fraction=0.1,
    learning_rate_init=0.001
)
models['Neural Network'].fit(X_train_scaled, y_train)
use_scaled['Neural Network'] = True

print()
print("✓ All models trained!")
print()

# ============================================================================
# 3. EVALUATE
# ============================================================================

print("3. Evaluating models...")

results = {}
total_outbreaks = int(y_test.sum())

for name, model in models.items():
    X_eval = X_test_scaled if use_scaled[name] else X_test
    y_pred_proba = model.predict_proba(X_eval)[:, 1]

    threshold = 0.2  # try 0.2 / 0.25 / 0.35
    y_pred = (y_pred_proba > threshold).astype(int)


    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred,    zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred,        zero_division=0),
        'auc':       roc_auc_score(y_test, y_pred_proba),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
    }

print("✓ All models evaluated!")
print()

# ============================================================================
# 4. PRINT RESULTS TABLE
# ============================================================================

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

results_df = pd.DataFrame({
    'Model':     list(results.keys()),
    'AUC':       [results[m]['auc']       for m in results],
    'Recall':    [results[m]['recall']    for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'F1-Score':  [results[m]['f1']        for m in results],
    'Accuracy':  [results[m]['accuracy']  for m in results],
}).sort_values('AUC', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))
print()

for name in results_df['Model']:
    r = results[name]
    print(f"{name}:")
    print(f"  AUC:       {r['auc']:.3f}")
    print(f"  Recall:    {r['recall']:.1%}  "
          f"(catches {int(r['tp'])} of {total_outbreaks} outbreaks, "
          f"misses {int(r['fn'])})")
    print(f"  Precision: {r['precision']:.1%}  "
          f"({int(r['tp'])} true positives vs {int(r['fp'])} false alarms)")
    print(f"  F1-Score:  {r['f1']:.3f}")
    print(f"  Accuracy:  {r['accuracy']:.1%}")
    print()

# ============================================================================
# 5. FEATURE IMPORTANCE (tree-based models)
# ============================================================================

print("Feature importance (XGBoost — top 15):")
fi = pd.Series(
    models['XGBoost'].feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

for feat, imp in fi.head(15).items():
    bar = '█' * int(imp * 60)
    print(f"  {feat:<22} {imp:.4f}  {bar}")
print()

# ============================================================================
# 6. VISUALISATION
# ============================================================================

print("4. Creating visualisation...")

COLORS = {
    'XGBoost':            '#2E75B6',
    'Random Forest':      '#1E8449',
    'Gradient Boosting':  '#7D3C98',
    'Logistic Regression':'#CA6F1E',
    'SVM':                '#17A589',
    'Neural Network':     '#C0392B',
}
model_order = list(results_df['Model'])  # sorted by AUC

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#FAFAFA')

# Title
fig.suptitle(
    'Multi-Model Comparison — DRC Epidemic Warning System  (v2: enriched features)',
    fontsize=15, fontweight='bold', y=0.98
)

# ── Panel layout ─────────────────────────────────────────────────────────────
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

# 1. ROC Curves
ax_roc = fig.add_subplot(gs[0, 0])
for name in model_order:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    ax_roc.plot(fpr, tpr, linewidth=2, color=COLORS[name],
                label=f"{name}  (AUC={results[name]['auc']:.3f})")
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax_roc.set_xlabel('False Positive Rate', fontsize=10)
ax_roc.set_ylabel('True Positive Rate', fontsize=10)
ax_roc.set_title('ROC Curves', fontweight='bold')
ax_roc.legend(fontsize=7.5, loc='lower right')
ax_roc.grid(alpha=0.25)
ax_roc.set_facecolor('#F8F8F8')

# 2–6. Metric bar charts
metric_panels = [
    ('auc',       'AUC',                          gs[0, 1], '.3f'),
    ('recall',    'Recall  (catching outbreaks)',  gs[0, 2], '.1%'),
    ('precision', 'Precision',                     gs[1, 0], '.1%'),
    ('f1',        'F1-Score',                      gs[1, 1], '.3f'),
    ('accuracy',  'Accuracy',                      gs[1, 2], '.1%'),
]

for key, title, gridspec, fmt in metric_panels:
    ax = fig.add_subplot(gridspec)
    ax.set_facecolor('#F8F8F8')
    vals  = [results[m][key] for m in model_order]
    cols  = [COLORS[m] for m in model_order]
    bars  = ax.barh(model_order, vals, color=cols, height=0.6, edgecolor='white')
    ax.set_xlim(0, min(max(vals) * 1.2, 1.0))
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis='x', alpha=0.25)
    for bar, v in zip(bars, vals):
        ax.text(
            v + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
            f'{v:{fmt}}', va='center', fontsize=8.5, fontweight='500'
        )

# ── Feature importance inset inside AUC panel ────────────────────────────────
ax_fi = fig.add_subplot(gs[1, 1])   # reuse F1 panel — add separate inset
# Actually make it a standalone 7th panel via inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes   # noqa

# Replace F1 panel with feature importance  (swap position gs[1,1] → fi)
ax_f1_existing = fig.axes[-1]  # last added = F1 panel
# Leave F1 panel as is; add feature importance below the figure as text
# (keeping 6 panels clean)

plt.savefig('DRC_Model_Comparison_v2.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("✓ Saved: DRC_Model_Comparison_v2.png")
plt.close()

# ── Feature importance separate figure ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.patch.set_facecolor('#FAFAFA')
top_features = fi.head(15)
bars2 = ax2.barh(
    top_features.index[::-1],
    top_features.values[::-1],
    color=['#C0392B' if 'OUTBREAK' in f
           else '#2E75B6' if 'CASES' in f
           else '#7D3C98' if 'MONTH' in f
           else '#888888'
           for f in top_features.index[::-1]],
    edgecolor='white', height=0.65
)
ax2.set_xlabel('Feature Importance (XGBoost)', fontsize=11)
ax2.set_title('Feature Importance — Top 15  (XGBoost)', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.25)
ax2.set_facecolor('#F8F8F8')
for bar, v in zip(bars2, top_features.values[::-1]):
    ax2.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{v:.4f}', va='center', fontsize=9)

# Legend
legend_elements = [
    mpatches.Patch(color='#C0392B', label='Autoregressive outbreak flags'),
    mpatches.Patch(color='#2E75B6', label='Case count lags / rolling'),
    mpatches.Patch(color='#7D3C98', label='Seasonal encoding'),
    mpatches.Patch(color='#888888', label='Location / time / weather'),
]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('DRC_Feature_Importance_v2.png', dpi=300, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("✓ Saved: DRC_Feature_Importance_v2.png")
plt.close()

# ============================================================================
# 7. SAVE CSV
# ============================================================================

results_df.to_csv('model_comparison_v2.csv', index=False)
print("✓ Saved: model_comparison_v2.csv")
print()

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

best_auc    = results_df.iloc[0]
best_recall = results_df.loc[results_df['Recall'].idxmax()]
best_f1     = results_df.loc[results_df['F1-Score'].idxmax()]

print(f"Best overall (AUC):       {best_auc['Model']:<22} AUC={best_auc['AUC']:.3f}")
print(f"Best for detection:       {best_recall['Model']:<22} Recall={best_recall['Recall']:.1%}")
print(f"Best balanced:            {best_f1['Model']:<22} F1={best_f1['F1-Score']:.3f}")
print()
print("Diseases evaluated:", consistent_diseases)
print()
print("Feature groups added in v2:")
print("  Autoregressive : OUTBREAK_LAG1, OUTBREAK_LAG2  (corr ≈ 0.64 with target)")
print("  Case lags      : CASES_LAG1..4")
print("  Rolling means  : CASES_ROLL4, CASES_ROLL8, CASES_ROLL12")
print("  Trend          : CASES_TREND")
print("  Seasonal       : MONTH_SIN, MONTH_COS")
print()
print("=" * 80)
print("✓ DONE")
print("=" * 80)
