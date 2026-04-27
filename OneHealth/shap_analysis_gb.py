"""
SHAP Analysis — Gradient Boosting Model
OneHealth DRC — Epidemic Early-Warning System

This script:
  1. Rebuilds the same dataset / train-test split used in compare_multiple_models_v2.py
  2. Trains the Gradient Boosting classifier
  3. Computes SHAP values (TreeExplainer — exact, fast for tree models)
  4. Produces four publication-ready figures:
       shap_summary_beeswarm.png   — global feature impact (beeswarm)
       shap_summary_bar.png        — mean |SHAP| bar chart
       shap_dependence_top4.png    — dependence plots for top-4 features
       shap_waterfall_examples.png — waterfall plots for 2 individual predictions
  5. Exports shap_values.csv       — raw SHAP matrix for further analysis

Requirements:
    pip install shap pandas numpy scikit-learn matplotlib
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

print("=" * 70)
print("SHAP ANALYSIS — GRADIENT BOOSTING  |  DRC EPIDEMIC WARNING SYSTEM")
print("=" * 70)
print()

# ============================================================
# 1.  LOAD DATA  (mirrors compare_multiple_models_v2.py)
# ============================================================

print("1. Loading data …")

if os.path.exists('DRC_ML_Ready_v3.csv'):
    df = pd.read_csv('DRC_ML_Ready_v3.csv')
    print("   Loaded DRC_ML_Ready_v3.csv")
else:
    print("   DRC_ML_Ready_v3.csv not found — computing features from base file")
    df = pd.read_csv('DRC_General_Risk_Updated.csv')

    def tukey_outbreak_threshold(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        threshold = max(q3 + 1.5 * iqr, group.quantile(0.90))
        return (group > threshold).astype(int)

    df['IS_OUTBREAK'] = (
        df.groupby('MALADIE')['TOTALCAS']
          .transform(tukey_outbreak_threshold)
    )

    df = df.sort_values(['PROV', 'MALADIE', 'DEBUTSEM']).reset_index(drop=True)
    grp_pd = df.groupby(['PROV', 'MALADIE'])

    for col in ['PRECTOTCORR', 'T2M', 'RH2M']:
        df[f'{col}_LAG1'] = grp_pd[col].shift(1)
        df[f'{col}_LAG2'] = grp_pd[col].shift(2)

    for lag in [1, 2, 3, 4]:
        df[f'CASES_LAG{lag}'] = grp_pd['TOTALCAS'].shift(lag)

    shifted = grp_pd['TOTALCAS'].shift(1)
    df['CASES_ROLL4']  = shifted.transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['CASES_ROLL8']  = shifted.transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['CASES_ROLL12'] = shifted.transform(lambda x: x.rolling(12, min_periods=1).mean())

    df['CASES_TREND']   = df['CASES_LAG1'] - df['CASES_ROLL4']
    grp_pm = df.groupby(['PROV', 'MALADIE'])
    df['OUTBREAK_LAG1'] = grp_pm['IS_OUTBREAK'].shift(1)
    df['OUTBREAK_LAG2'] = grp_pm['IS_OUTBREAK'].shift(2)

    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12)

    df = df.dropna(subset=['CASES_LAG4', 'OUTBREAK_LAG2']).reset_index(drop=True)

print(f"   {len(df):,} rows  |  outbreak rate: {df['IS_OUTBREAK'].mean()*100:.2f}%")

# ── Keep only diseases present in BOTH train and test ──────────────────
pre  = set(df[df['ANNEE'] <= 2014]['MALADIE'].unique())
post = set(df[df['ANNEE'] >  2014]['MALADIE'].unique())
consistent_diseases = sorted(pre & post)
df_m = df[df['MALADIE'].isin(consistent_diseases)].copy()

# ── Feature matrix ──────────────────────────────────────────────────────
EXCLUDE = {
    'IS_OUTBREAK', 'IS_OUTBREAK_NEW', 'IS_OUTBREAK_OLD', 'TARGET',
    'TOTALCAS', 'TOTALDECES', 'DEBUTSEM',
}
feature_cols = [c for c in df_m.columns if c not in EXCLUDE]

X = df_m[feature_cols].copy()
y = df_m['IS_OUTBREAK'].copy()

le_prov = LabelEncoder()
le_mal  = LabelEncoder()
X['PROV']    = le_prov.fit_transform(X['PROV'])
X['MALADIE'] = le_mal.fit_transform(X['MALADIE'])

train_mask = df_m['ANNEE'] <= 2014
X_train, X_test = X[train_mask],  X[~train_mask]
y_train, y_test = y[train_mask],  y[~train_mask]

print(f"   Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print()

# ============================================================
# 2.  TRAIN GRADIENT BOOSTING
# ============================================================

print("2. Training Gradient Boosting …")

gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)

y_prob = gb.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.2).astype(int)

print(f"   AUC       : {roc_auc_score(y_test, y_prob):.4f}")
print(f"   Recall    : {recall_score(y_test, y_pred, zero_division=0):.1%}")
print(f"   Precision : {precision_score(y_test, y_pred, zero_division=0):.1%}")
print(f"   F1        : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print()

# ============================================================
# 3.  COMPUTE SHAP VALUES  (TreeExplainer — exact for GBM)
# ============================================================

print("3. Computing SHAP values …")

explainer   = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)   # shape: (n_test, n_features)

# GradientBoostingClassifier returns expected_value as a 1-element array
# for binary classification — extract the scalar safely
_ev = explainer.expected_value
if hasattr(_ev, '__len__'):
    expected_value = float(_ev[-1])   # last element = positive class (works for size 1 or 2)
else:
    expected_value = float(_ev)

# shap_values may also come back as a list [neg_class, pos_class] — keep pos class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"   SHAP matrix shape : {shap_values.shape}")
print(f"   Base value        : {expected_value:.4f}")
print()

# ── Mean absolute SHAP per feature (global importance) ──────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

print("   Top-15 features by mean |SHAP|:")
for feat, val in shap_importance.head(15).items():
    bar = '█' * int(val * 400)
    print(f"   {feat:<26} {val:.5f}  {bar}")
print()

# ── Export raw SHAP matrix ──────────────────────────────────────────────
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df.to_csv('shap_values.csv', index=False)
print("   Saved: shap_values.csv")
print()

# ============================================================
# 4.  FIGURE 1 — BEESWARM SUMMARY PLOT
# ============================================================

print("4. Figure 1 — Beeswarm summary plot …")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_cols,
    max_display=15,
    show=False,
    plot_type='dot',
    color_bar_label='Feature value',
)
plt.title(
    'SHAP Summary — Gradient Boosting\n'
    'DRC Epidemic Early-Warning System  (test period 2015–2017)',
    fontsize=12, fontweight='bold', pad=12
)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_summary_beeswarm.png")

# ============================================================
# 5.  FIGURE 2 — BAR CHART (mean |SHAP|)
# ============================================================

print("5. Figure 2 — Bar chart (mean |SHAP|) …")

PALETTE = {
    'OUTBREAK': '#C0392B',   # autoregressive outbreak flags
    'INC':      '#2E75B6',   # incidence-based features
    'CASES':    '#2E75B6',   # case-count features (v2 naming)
    'MONTH':    '#7D3C98',   # seasonal
    'ANOM':     '#D35400',   # climate anomaly
    'RAIN':     '#1E8449',   # rain / precipitation
    'TEMP':     '#1E8449',   # temperature
    'RH2M':     '#1E8449',   # humidity
    'HUM':      '#1E8449',   # humidity interaction
    'DEFAULT':  '#888888',
}

def get_color(feat):
    feat_upper = feat.upper()
    for key, col in PALETTE.items():
        if key in feat_upper:
            return col
    return PALETTE['DEFAULT']

top15  = shap_importance.head(15)
colors = [get_color(f) for f in top15.index[::-1]]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F8F8F8')

bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors,
               edgecolor='white', height=0.65)

for bar, val in zip(bars, top15.values[::-1]):
    ax.text(val + top15.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.5f}', va='center', fontsize=9)

ax.set_xlabel('Mean |SHAP value|  (impact on model output)', fontsize=11)
ax.set_title(
    'Feature Importance — Gradient Boosting (SHAP)\n'
    'DRC Epidemic Early-Warning System  (test period 2015–2017)',
    fontsize=12, fontweight='bold'
)
ax.grid(axis='x', alpha=0.25)

legend_elements = [
    mpatches.Patch(color='#C0392B', label='Autoregressive outbreak flags'),
    mpatches.Patch(color='#2E75B6', label='Case / Incidence lags & rolling'),
    mpatches.Patch(color='#7D3C98', label='Seasonal encoding'),
    mpatches.Patch(color='#D35400', label='Climate anomalies'),
    mpatches.Patch(color='#1E8449', label='Weather / Climate variables'),
    mpatches.Patch(color='#888888', label='Location / time'),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right')

plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("   Saved: shap_summary_bar.png")

# ============================================================
# 6.  FIGURE 3 — DEPENDENCE PLOTS (top-4 features)
# ============================================================

print("6. Figure 3 — Dependence plots (top-4 features) …")

top4 = shap_importance.head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle(
    'SHAP Dependence Plots — Top-4 Features  (Gradient Boosting)',
    fontsize=13, fontweight='bold', y=1.01
)

for ax, feat in zip(axes.flat, top4):
    feat_idx  = list(feature_cols).index(feat)
    feat_vals = X_test[feat].values
    sv        = shap_values[:, feat_idx]

    sc = ax.scatter(feat_vals, sv, c=feat_vals, cmap='RdYlBu_r',
                    alpha=0.35, s=12, linewidths=0)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('SHAP value', fontsize=10)
    ax.set_title(feat, fontweight='bold', fontsize=10)
    ax.set_facecolor('#F8F8F8')
    ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label('Feature value', fontsize=8)

plt.tight_layout()
plt.savefig('shap_dependence_top4.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("   Saved: shap_dependence_top4.png")

# ============================================================
# 7.  FIGURE 4 — WATERFALL PLOTS (2 individual predictions)
# ============================================================

print("7. Figure 4 — Waterfall plots (2 individual predictions) …")

# Pick one true outbreak and one true non-outbreak from test set
y_test_arr = y_test.values
tp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0]
tn_idx = np.where((y_test_arr == 0) & (y_pred == 0))[0]

sample_indices = {
    'True Positive\n(outbreak correctly predicted)': tp_idx[0] if len(tp_idx) else 0,
    'True Negative\n(no outbreak correctly predicted)': tn_idx[0] if len(tn_idx) else 1,
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle(
    'SHAP Waterfall Plots — Individual Predictions  (Gradient Boosting)',
    fontsize=13, fontweight='bold'
)

for ax, (label, idx) in zip(axes, sample_indices.items()):
    sv_row = shap_values[idx]                 # per-feature SHAP for this row
    base   = expected_value
    pred_p = y_prob[idx]

    # Keep top-10 by absolute magnitude, aggregate the rest
    order      = np.argsort(np.abs(sv_row))[::-1]
    top_n      = 10
    top_idx    = order[:top_n]
    rest_idx   = order[top_n:]

    top_feats  = [feature_cols[i] for i in top_idx]
    top_sv     = sv_row[top_idx]
    rest_sv    = sv_row[rest_idx].sum()

    feats = top_feats + ['Others']
    svs   = list(top_sv) + [rest_sv]

    # Sort by value for waterfall readability
    paired = sorted(zip(svs, feats), key=lambda x: x[0])
    svs_sorted   = [p[0] for p in paired]
    feats_sorted = [p[1] for p in paired]

    colors_wf = ['#C0392B' if v > 0 else '#2E75B6' for v in svs_sorted]

    ax.set_facecolor('#F8F8F8')
    bars_wf = ax.barh(feats_sorted, svs_sorted, color=colors_wf,
                      edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)

    for bar, v in zip(bars_wf, svs_sorted):
        offset = max(np.abs(svs_sorted)) * 0.02 * (1 if v >= 0 else -1)
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f'{v:+.4f}', va='center', fontsize=8,
                ha='left' if v >= 0 else 'right')

    ax.set_xlabel('SHAP value  (contribution to prediction)', fontsize=10)
    ax.set_title(
        f'{label}\n'
        f'Predicted probability = {pred_p:.3f}  |  '
        f'Base value = {base:.3f}',
        fontsize=9.5, fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.savefig('shap_waterfall_examples.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("   Saved: shap_waterfall_examples.png")

# ============================================================
# 8.  DISEASE-LEVEL SHAP BREAKDOWN
# ============================================================

print()
print("8. Disease-level mean |SHAP| breakdown …")

# Attach disease label back to test rows
disease_labels = le_mal.inverse_transform(X_test['MALADIE'].values)

shap_df_indexed = pd.DataFrame(shap_values, columns=feature_cols)
shap_df_indexed['MALADIE'] = disease_labels

# Exclude 'MALADIE' from numeric SHAP columns (it is now a string label)
numeric_shap_cols = [c for c in feature_cols if c != 'MALADIE']

disease_shap = (
    shap_df_indexed.groupby('MALADIE')[numeric_shap_cols]
    .apply(lambda g: g.abs().mean())
)

# Top-3 features per disease
print()
print("   Top-3 SHAP features per disease:")
for disease, row in disease_shap.iterrows():
    top3 = row.sort_values(ascending=False).head(3)
    feats_str = ' | '.join([f'{f} ({v:.4f})' for f, v in top3.items()])
    print(f"   {disease:<20} → {feats_str}")

disease_shap.to_csv('shap_by_disease.csv')
print()
print("   Saved: shap_by_disease.csv")

# ============================================================
# 9.  SUMMARY
# ============================================================

print()
print("=" * 70)
print("OUTPUTS")
print("=" * 70)
print("  shap_summary_beeswarm.png  — beeswarm plot (global feature impact)")
print("  shap_summary_bar.png       — mean |SHAP| bar chart")
print("  shap_dependence_top4.png   — dependence plots for top-4 features")
print("  shap_waterfall_examples.png— waterfall for 1 TP + 1 TN prediction")
print("  shap_values.csv            — raw SHAP matrix (n_test × n_features)")
print("  shap_by_disease.csv        — mean |SHAP| aggregated per disease")
print()
print("=" * 70)
print("KEY FINDINGS (inspect plots for full detail)")
print("=" * 70)
print()
print(f"  Model AUC (GB, threshold=0.20): {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Base value (E[f(X)])           : {expected_value:.4f}")
print()
print("  Top-5 global SHAP drivers:")
for feat, val in shap_importance.head(5).items():
    print(f"    {feat:<26} mean|SHAP| = {val:.5f}")
print()
print("  Interpretation guide:")
print("    • Positive SHAP → pushes prediction toward OUTBREAK=1")
print("    • Negative SHAP → pushes prediction toward OUTBREAK=0")
print("    • Beeswarm color: red=high feature value, blue=low feature value")
print()
print("=" * 70)
print("✓ SHAP ANALYSIS COMPLETE")
print("=" * 70)
