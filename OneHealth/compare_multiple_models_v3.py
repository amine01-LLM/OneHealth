"""
compare_multiple_models_v3.py
─────────────────────────────
Changes vs v2:
  • Loads DRC_ML_Ready_v4.csv (8-week lags)
  • INCIDENCE excluded from training features (tagged in EXCLUDE)
  • Feature selection stage added:
      – Step 1: drop near-zero-variance features
      – Step 2: drop highly correlated pairs (|r| > 0.95)
      – Step 3: mutual-information ranking → keep top-K
  • All downstream model code unchanged.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    recall_score, precision_score, accuracy_score, f1_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-MODEL COMPARISON v3 — DRC EPIDEMIC WARNING SYSTEM")
print("  (8-week lags · feature selection · INCIDENCE excluded from X)")
print("=" * 80)
print()


# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("1. Loading data...")

if os.path.exists('DRC_ML_Ready_v4.csv'):
    df = pd.read_csv('DRC_ML_Ready_v4.csv')
    print("   Loaded DRC_ML_Ready_v4.csv")
else:
    raise FileNotFoundError(
        "DRC_ML_Ready_v4.csv not found. "
        "Run feature_engineering_v4.py first."
    )

print(f"   {len(df):,} rows | outbreak rate: {df['IS_OUTBREAK'].mean()*100:.2f}%")
print()


# ============================================================================
# 2. DISEASE FILTER  (keep only diseases present in both train & test)
# ============================================================================

print("2. Disease filter...")

pre  = set(df[df['ANNEE'] <= 2014]['MALADIE'].unique())
post = set(df[df['ANNEE'] >  2014]['MALADIE'].unique())
consistent_diseases = sorted(pre & post)
excluded_diseases   = sorted(set(df['MALADIE'].unique()) - set(consistent_diseases))

print(f"   Kept  : {consistent_diseases}")
print(f"   Dropped (one period only): {excluded_diseases}")

df = df[df['MALADIE'].isin(consistent_diseases)].copy()
print(f"   Dataset after filter: {len(df):,} rows")
print()


# ============================================================================
# 3. BUILD FEATURE MATRIX
# ============================================================================

print("3. Building feature matrix...")

# ── Columns that must NEVER enter the model ──────────────────────────────────
EXCLUDE = {
    'IS_OUTBREAK',                   # target
    'TOTALCAS', 'TOTALDECES',        # raw outcomes (leakage)
    'INCIDENCE',                     # derived from TOTALCAS/POP → leakage
    'DEBUTSEM',                      # raw date string
}

feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"   Raw feature count after EXCLUDE: {len(feature_cols)}")

X_raw = df[feature_cols].copy()
y     = df['IS_OUTBREAK'].copy()

# Encode categoricals
le_prov = LabelEncoder()
le_mal  = LabelEncoder()
X_raw['PROV']    = le_prov.fit_transform(X_raw['PROV'])
X_raw['MALADIE'] = le_mal.fit_transform(X_raw['MALADIE'])


# ============================================================================
# 4. FEATURE SELECTION  (run on train split to prevent leakage)
# ============================================================================

print("4. Feature selection...")

train_mask = df['ANNEE'] <= 2014
X_tr_raw   = X_raw[train_mask]
y_tr       = y[train_mask]

# ── Step 4a: Near-zero variance filter ───────────────────────────────────────
variances   = X_tr_raw.var()
nzv_mask    = variances > 1e-6
dropped_nzv = list(variances[~nzv_mask].index)
X_tr_raw    = X_tr_raw.loc[:, nzv_mask]
print(f"   [4a] Near-zero variance dropped : {len(dropped_nzv)}  {dropped_nzv}")

# ── Step 4b: High-correlation filter  (|r| > 0.95) ───────────────────────────
corr_matrix = X_tr_raw.corr().abs()
upper       = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X_tr_raw     = X_tr_raw.drop(columns=to_drop_corr)
print(f"   [4b] High-correlation dropped   : {len(to_drop_corr)}  "
      f"(kept {X_tr_raw.shape[1]} features)")

# ── Step 4c: Mutual information ranking ──────────────────────────────────────
# Fill any remaining NaNs for MI computation
X_tr_filled = X_tr_raw.fillna(X_tr_raw.median())

mi_scores = mutual_info_classif(
    X_tr_filled, y_tr, discrete_features=False, random_state=42
)
mi_series = pd.Series(mi_scores, index=X_tr_raw.columns).sort_values(ascending=False)

# Keep features with MI > 0 AND top-40 (whichever is smaller)
mi_positive  = mi_series[mi_series > 0].index.tolist()
TOP_K        = min(40, len(mi_positive))
selected_features = mi_series.head(TOP_K).index.tolist()

print(f"   [4c] MI ranking: {len(mi_positive)} features with MI>0; keeping top {TOP_K}")
print()
print("   Top-20 features by mutual information:")
for feat, score in mi_series.head(20).items():
    bar = '█' * int(score * 80)
    print(f"     {feat:<30} {score:.4f}  {bar}")
print()

# Restrict X to selected features only
X = X_raw[selected_features].copy()

# ============================================================================
# 5. TRAIN / TEST SPLIT  +  SCALING
# ============================================================================

print("5. Preparing train/test split...")

X_train, X_test = X[train_mask],  X[~train_mask]
y_train, y_test = y[train_mask],  y[~train_mask]

# Fill NaN with column median (same split to avoid leakage)
col_medians  = X_train.median()
X_train      = X_train.fillna(col_medians)
X_test       = X_test.fillna(col_medians)

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"   Train (2006–2014): {len(X_train):,} rows | "
      f"outbreak rate: {y_train.mean()*100:.2f}%")
print(f"   Test  (2015–2017): {len(X_test):,} rows  | "
      f"outbreak rate: {y_test.mean()*100:.2f}%")
print(f"   Class imbalance ratio : {pos_weight:.1f}x")
print(f"   Final feature count   : {len(selected_features)}")
print()


# ============================================================================
# 6. TRAIN MODELS
# ============================================================================

print("6. Training models...")

models     = {}
use_scaled = {}

print("   XGBoost...")
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss',
)
models['XGBoost'].fit(X_train, y_train)
use_scaled['XGBoost'] = False

print("   Random Forest...")
models['Random Forest'] = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
models['Random Forest'].fit(X_train, y_train)
use_scaled['Random Forest'] = False

print("   Gradient Boosting...")
models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42,
)
models['Gradient Boosting'].fit(X_train, y_train)
use_scaled['Gradient Boosting'] = False

print("   Logistic Regression...")
models['Logistic Regression'] = LogisticRegression(
    max_iter=1000, class_weight='balanced', C=0.1,
    random_state=42, n_jobs=-1,
)
models['Logistic Regression'].fit(X_train_sc, y_train)
use_scaled['Logistic Regression'] = True

print("   SVM...")
models['SVM'] = SVC(
    kernel='rbf', C=1.0, class_weight='balanced',
    probability=True, random_state=42,
)
models['SVM'].fit(X_train_sc, y_train)
use_scaled['SVM'] = True

print("   Neural Network...")
models['Neural Network'] = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), max_iter=500,
    random_state=42, early_stopping=True, validation_fraction=0.1,
    learning_rate_init=0.001,
)
models['Neural Network'].fit(X_train_sc, y_train)
use_scaled['Neural Network'] = True

print("\n✓ All models trained!")
print()


# ============================================================================
# 7. EVALUATE
# ============================================================================

print("7. Evaluating models...")

THRESHOLD       = 0.2
results         = {}
total_outbreaks = int(y_test.sum())

for name, model in models.items():
    X_eval        = X_test_sc if use_scaled[name] else X_test
    y_proba       = model.predict_proba(X_eval)[:, 1]
    y_pred        = (y_proba > THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results[name] = {
        'accuracy'  : accuracy_score(y_test, y_pred),
        'recall'    : recall_score(y_test, y_pred,    zero_division=0),
        'precision' : precision_score(y_test, y_pred, zero_division=0),
        'f1'        : f1_score(y_test, y_pred,        zero_division=0),
        'auc'       : roc_auc_score(y_test, y_proba),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'y_pred_proba': y_proba,
        'y_pred'      : y_pred,
    }

print("✓ All models evaluated!")
print()


# ============================================================================
# 8. RESULTS TABLE
# ============================================================================

print("=" * 80)
print("RESULTS")
print("=" * 80)

results_df = pd.DataFrame({
    'Model'    : list(results.keys()),
    'AUC'      : [results[m]['auc']       for m in results],
    'Recall'   : [results[m]['recall']    for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'F1-Score' : [results[m]['f1']        for m in results],
    'Accuracy' : [results[m]['accuracy']  for m in results],
}).sort_values('AUC', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))
print()

for name in results_df['Model']:
    r = results[name]
    print(f"{name}:")
    print(f"  AUC       : {r['auc']:.3f}")
    print(f"  Recall    : {r['recall']:.1%}  "
          f"(catches {int(r['tp'])} / {total_outbreaks} outbreaks, "
          f"misses {int(r['fn'])})")
    print(f"  Precision : {r['precision']:.1%}  "
          f"({int(r['tp'])} true pos vs {int(r['fp'])} false alarms)")
    print(f"  F1-Score  : {r['f1']:.3f}")
    print(f"  Accuracy  : {r['accuracy']:.1%}")
    print()


# ============================================================================
# 9. XGBoost FEATURE IMPORTANCE  (over selected features)
# ============================================================================

print("XGBoost feature importance (top 20 of selected):")
fi = pd.Series(
    models['XGBoost'].feature_importances_,
    index=selected_features,
).sort_values(ascending=False)

for feat, imp in fi.head(20).items():
    bar = '█' * int(imp * 60)
    print(f"  {feat:<30} {imp:.4f}  {bar}")
print()


# ============================================================================
# 10. VISUALISATION
# ============================================================================

print("8. Creating visualisations...")

COLORS = {
    'XGBoost'           : '#2E75B6',
    'Random Forest'     : '#1E8449',
    'Gradient Boosting' : '#7D3C98',
    'Logistic Regression': '#CA6F1E',
    'SVM'               : '#17A589',
    'Neural Network'    : '#C0392B',
}
model_order = list(results_df['Model'])

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle(
    'Multi-Model Comparison v3 — DRC Epidemic Warning System\n'
    '(8-week lags · feature selection · INCIDENCE excluded from training)',
    fontsize=14, fontweight='bold', y=0.98,
)

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

# ROC curves
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

# Metric bar charts
metric_panels = [
    ('auc',       'AUC',                         gs[0, 1], '.3f'),
    ('recall',    'Recall  (catching outbreaks)', gs[0, 2], '.1%'),
    ('precision', 'Precision',                    gs[1, 0], '.1%'),
    ('f1',        'F1-Score',                     gs[1, 1], '.3f'),
    ('accuracy',  'Accuracy',                     gs[1, 2], '.1%'),
]

for key, title, gridspec, fmt in metric_panels:
    ax = fig.add_subplot(gridspec)
    ax.set_facecolor('#F8F8F8')
    vals = [results[m][key] for m in model_order]
    cols = [COLORS[m] for m in model_order]
    bars = ax.barh(model_order, vals, color=cols, height=0.6, edgecolor='white')
    ax.set_xlim(0, min(max(vals) * 1.2, 1.0))
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis='x', alpha=0.25)
    for bar, v in zip(bars, vals):
        ax.text(
            v + max(vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{v:{fmt}}', va='center', fontsize=8.5, fontweight='500',
        )

plt.savefig('DRC_Model_Comparison_v3.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("✓ Saved: DRC_Model_Comparison_v3.png")
plt.close()

# Feature importance (XGBoost, top 20)
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor('#FAFAFA')
top_fi = fi.head(20)

bar_colors = [
    '#C0392B' if 'OUTBREAK' in f
    else '#2E75B6' if ('INC' in f or 'CASES' in f)
    else '#E67E22' if any(w in f for w in ['LAG5','LAG6','LAG7','LAG8'])
    else '#7D3C98' if 'MONTH' in f
    else '#888888'
    for f in top_fi.index[::-1]
]

bars2 = ax2.barh(
    top_fi.index[::-1], top_fi.values[::-1],
    color=bar_colors, edgecolor='white', height=0.65,
)
ax2.set_xlabel('Feature Importance (XGBoost)', fontsize=11)
ax2.set_title(
    'Feature Importance — Top 20 (XGBoost, v3)\n'
    '8-week lags · INCIDENCE excluded · after feature selection',
    fontweight='bold', fontsize=11,
)
ax2.grid(axis='x', alpha=0.25)
ax2.set_facecolor('#F8F8F8')
for bar, v in zip(bars2, top_fi.values[::-1]):
    ax2.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{v:.4f}', va='center', fontsize=8.5)

legend_elements = [
    mpatches.Patch(color='#C0392B', label='Autoregressive outbreak flags'),
    mpatches.Patch(color='#2E75B6', label='Incidence lags / rolling means'),
    mpatches.Patch(color='#E67E22', label='Extended lags (wk 5–8)'),
    mpatches.Patch(color='#7D3C98', label='Seasonal encoding'),
    mpatches.Patch(color='#888888', label='Location / time / weather'),
]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('DRC_Feature_Importance_v3.png', dpi=300, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("✓ Saved: DRC_Feature_Importance_v3.png")
plt.close()

# MI ranking plot
fig3, ax3 = plt.subplots(figsize=(10, 8))
fig3.patch.set_facecolor('#FAFAFA')
top_mi = mi_series.head(30)
mi_colors = [
    '#C0392B' if 'OUTBREAK' in f
    else '#2E75B6' if 'INC' in f
    else '#E67E22' if any(f'LAG{k}' in f for k in [5,6,7,8])
    else '#7D3C98' if 'MONTH' in f
    else '#888888'
    for f in top_mi.index[::-1]
]
ax3.barh(top_mi.index[::-1], top_mi.values[::-1],
         color=mi_colors, edgecolor='white', height=0.65)
ax3.set_xlabel('Mutual Information Score', fontsize=11)
ax3.set_title('Feature Selection — Mutual Information Ranking (top 30)',
              fontweight='bold', fontsize=11)
ax3.grid(axis='x', alpha=0.25)
ax3.set_facecolor('#F8F8F8')
ax3.legend(handles=legend_elements, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('DRC_MI_Ranking_v3.png', dpi=300, bbox_inches='tight',
            facecolor=fig3.get_facecolor())
print("✓ Saved: DRC_MI_Ranking_v3.png")
plt.close()


# ============================================================================
# 11. SAVE CSV
# ============================================================================

results_df.to_csv('model_comparison_v3.csv', index=False)
print("✓ Saved: model_comparison_v3.csv")
print()


# ============================================================================
# 12. SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)

best_auc    = results_df.iloc[0]
best_recall = results_df.loc[results_df['Recall'].idxmax()]
best_f1     = results_df.loc[results_df['F1-Score'].idxmax()]

print(f"Best overall (AUC)   : {best_auc['Model']:<22} AUC={best_auc['AUC']:.3f}")
print(f"Best for detection   : {best_recall['Model']:<22} Recall={best_recall['Recall']:.1%}")
print(f"Best balanced (F1)   : {best_f1['Model']:<22} F1={best_f1['F1-Score']:.3f}")
print()
print(f"Diseases evaluated   : {consistent_diseases}")
print(f"Features selected    : {len(selected_features)} (from {len(feature_cols)} raw)")
print()
print("v3 changes summary:")
print("  • INCIDENCE excluded from X (was target-leaking in v2)")
print("  • INC_LAG1..8 + weather LAG1..8  (extended from 4/2 in v3)")
print("  • INC_ROLL16 added")
print("  • INC_TREND_LONG (lag1 - roll12) added")
print("  • Feature selection: NZV filter → corr filter → MI top-40")
print()
print("=" * 80)
print("✓ DONE")
print("=" * 80)
