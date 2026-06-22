import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('DRC_Health_Weather_CLEANED.csv')
if not pd.api.types.is_datetime64_any_dtype(df['DEBUTSEM']):
    df['DEBUTSEM'] = pd.to_datetime(df['DEBUTSEM'])

# --- Configuration ---
provinces_a_analyser = ['KATANGA', 'KINSHASA', 'SUD-KIVU', 'EQUATEUR', 'ORIENTALE']

# Lags optimaux par maladie (basés sur la biologie / corrélations)
# Paludisme : transmission vectorielle lente (8-16 sem)
# Choléra   : transmission hydrique rapide (1-3 sem)
# Rougeole  : transmission aérienne, moins liée au climat (2-4 sem)
# Typhoïde  : transmission hydrique (2-4 sem)
# Méningite : lié à la saison sèche / humidité basse (2-4 sem)
LAGS_PAR_MALADIE = {
    'PALUDISME'      : {'PRECTOTCORR': 8,  'RH2M': 3,  'T2M': 16},
    'CHOLERA'        : {'PRECTOTCORR': 2,  'RH2M': 1,  'T2M': 2 },
    'ROUGEOLE'       : {'PRECTOTCORR': 3,  'RH2M': 2,  'T2M': 3 },
    'FIEVRE TYPHOIDE': {'PRECTOTCORR': 3,  'RH2M': 2,  'T2M': 3 },
    'MENINGITE'      : {'PRECTOTCORR': 2,  'RH2M': 4,  'T2M': 2 },
    'IRA'            : {'PRECTOTCORR': 2,  'RH2M': 2,  'T2M': 1 },
}

# Seuil percentile par maladie
SEUIL_PAR_MALADIE = {
    'PALUDISME'      : 85,
    'CHOLERA'        : 85,
    'ROUGEOLE'       : 85,
    'FIEVRE TYPHOIDE': 85,
    'MENINGITE'      : 85,
    'IRA'            : 85,
}

# Maladies disponibles dans le dataset
maladies_disponibles = df['MALADIE'].unique().tolist()
print("Maladies disponibles dans le dataset :")
for m in sorted(maladies_disponibles):
    n = len(df[df['MALADIE'] == m])
    print(f"  - {m} ({n} lignes)")

# On garde les maladies qu'on a configurées ET qui existent dans le dataset
maladies_a_traiter = [m for m in LAGS_PAR_MALADIE.keys() if m in maladies_disponibles]
print(f"\nMaladies traitées : {maladies_a_traiter}")

# Dictionnaires de stockage globaux
tous_modeles  = {}   # tous_modeles[maladie][province] = rf
tous_scalers  = {}   # tous_scalers[maladie][province] = scaler
tous_seuils   = {}   # tous_seuils[maladie][province]  = tau
tous_resultats = []  # liste de dict pour le tableau comparatif final



for maladie in maladies_a_traiter:

    print("\n" + "=" * 70)
    print(f"  MALADIE : {maladie}")
    print("=" * 70)

    lags        = LAGS_PAR_MALADIE[maladie]
    percentile  = SEUIL_PAR_MALADIE[maladie]
    features    = [f'{var}_lag{lag}' for var, lag in lags.items()]

    tous_modeles[maladie] = {}
    tous_scalers[maladie] = {}
    tous_seuils[maladie]  = {}

    for province in provinces_a_analyser:

        print(f"\n  --- {province} ---")

        # Filtrage et tri
        data = df[(df['PROV'] == province) & (df['MALADIE'] == maladie)].copy()
        data = data.sort_values('DEBUTSEM').reset_index(drop=True)

        # Lags
        for var, lag in lags.items():
            data[f'{var}_lag{lag}'] = data[var].shift(lag)

        data_clean = data.dropna().reset_index(drop=True)

        if len(data_clean) < 50:
            print(f"  Donnees insuffisantes ({len(data_clean)} semaines) — ignore")
            continue

        # --- Seuil épidémique ---
        seuil_epidemie = np.percentile(data_clean['INCIDENCE'], percentile)
        data_clean['TARGET'] = (data_clean['INCIDENCE'] > seuil_epidemie).astype(int)
        n_epi     = data_clean['TARGET'].sum()
        n_non_epi = len(data_clean) - n_epi

        print(f"  Periode  : {data_clean['DEBUTSEM'].min().date()} -> {data_clean['DEBUTSEM'].max().date()}")
        print(f"  Semaines : {len(data_clean)}  |  Seuil P{percentile} : {seuil_epidemie:.2f} cas/1000h")
        print(f"  Epidemies: {n_epi} ({100*n_epi/len(data_clean):.1f}%)  |  Non-epidemies: {n_non_epi}")

        # --- Features & target ---
        X = data_clean[features].values
        y = data_clean['TARGET'].values

        # --- Normalisation ---
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- SMOTE ---
        if n_epi >= 5:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, n_epi - 1))
                X_res, y_res = smote.fit_resample(X_scaled, y)
            except Exception as e:
                X_res, y_res = X_scaled, y
        else:
            X_res, y_res = X_scaled, y

        # --- Random Forest ---
        class_weight = {0: 1, 1: min(5, max(2, n_non_epi // max(1, n_epi)))}
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=5,
            class_weight=class_weight,
            random_state=42
        )
        rf.fit(X_res, y_res)

        # --- Optimisation tau 
        y_proba   = rf.predict_proba(X_scaled)[:, 1]
        best_f1, best_tau = 0, 0.5
        for tau in np.arange(0.20, 0.80, 0.05):
            y_pred_tau = (y_proba >= tau).astype(int)
            f1 = f1_score(y, y_pred_tau, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_tau = tau

        y_pred_final = (y_proba >= best_tau).astype(int)

        print(f"  Tau optimal : {best_tau:.2f}  |  F1-Score : {best_f1:.3f}")

        report = classification_report(
            y, y_pred_final,
            target_names=['Pas epidemie', 'Epidemie'],
            output_dict=True, zero_division=0
        )
        print(classification_report(
            y, y_pred_final,
            target_names=['Pas epidemie', 'Epidemie'],
            zero_division=0
        ))

        # Importance des variables
        imp = pd.DataFrame({
            'Variable':   features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("  Importance des variables :")
        print(imp.to_string(index=False))

        # Stockage
        tous_modeles[maladie][province] = rf
        tous_scalers[maladie][province] = scaler
        tous_seuils[maladie][province]  = best_tau

        tous_resultats.append({
            'Maladie'      : maladie,
            'Province'     : province,
            'N_semaines'   : len(data_clean),
            'N_epidemies'  : n_epi,
            'Seuil_epi'    : round(seuil_epidemie, 2),
            'F1_Score'     : round(best_f1, 3),
            'Recall'       : round(report['Epidemie']['recall'], 3),
            'Precision'    : round(report['Epidemie']['precision'], 3),
            'Tau'          : best_tau,
        })



print("\n" + "=" * 70)
print("  SYNTHESE COMPARATIVE — TOUTES MALADIES x TOUTES PROVINCES")
print("=" * 70)

df_res = pd.DataFrame(tous_resultats)

for maladie in maladies_a_traiter:
    subset = df_res[df_res['Maladie'] == maladie].sort_values('F1_Score', ascending=False)
    if len(subset) == 0:
        continue
    print(f"\n  {maladie}")
    print(f"  {'-'*60}")
    print(subset[['Province', 'N_semaines', 'N_epidemies',
                  'F1_Score', 'Recall', 'Precision', 'Tau']].to_string(index=False))
    print(f"  F1 moyen : {subset['F1_Score'].mean():.3f}  |  "
          f"Meilleur : {subset['F1_Score'].max():.3f} ({subset.iloc[0]['Province']})")

# Tableau global trié par F1
print("\n" + "=" * 70)
print("  TOP 10 — Meilleurs modeles (F1-Score)")
print("=" * 70)
top10 = df_res.sort_values('F1_Score', ascending=False).head(10)
print(top10[['Maladie', 'Province', 'F1_Score', 'Recall', 'Precision', 'Tau']].to_string(index=False))

print("\n" + "=" * 70)
print("  FLOP — Modeles a ameliorer (F1 < 0.5)")
print("=" * 70)
flop = df_res[df_res['F1_Score'] < 0.5].sort_values('F1_Score')
if len(flop) > 0:
    print(flop[['Maladie', 'Province', 'F1_Score', 'Recall', 'Precision']].to_string(index=False))
else:
    print("  Aucun modele sous 0.5 !")



# VISUALISATION

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Systeme d'alerte precoce — F1-Score par maladie et province", fontsize=14)
axes = axes.flatten()

colors_province = {
    'KATANGA': '#2196F3', 'KINSHASA': '#4CAF50',
    'SUD-KIVU': '#FF9800', 'EQUATEUR': '#9C27B0', 'ORIENTALE': '#F44336'
}

for idx, maladie in enumerate(maladies_a_traiter[:6]):
    ax     = axes[idx]
    subset = df_res[df_res['Maladie'] == maladie].sort_values('Province')
    if len(subset) == 0:
        ax.set_visible(False)
        continue

    provs  = subset['Province'].tolist()
    f1s    = subset['F1_Score'].tolist()
    cols   = [colors_province.get(p, 'gray') for p in provs]

    bars = ax.bar(provs, f1s, color=cols, alpha=0.82)
    ax.axhline(0.7, color='green',  linestyle='--', linewidth=1, label='0.7')
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, label='0.5')
    ax.set_title(maladie, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1-Score')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

# Masquer les axes vides
for idx in range(len(maladies_a_traiter), 6):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('resultats_multi_maladies.png', dpi=150, bbox_inches='tight')
plt.show()

# Heatmap F1-Score : maladies x provinces
pivot = df_res.pivot_table(
    index='Maladie', columns='Province', values='F1_Score', aggfunc='mean'
)
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30)
ax.set_yticks(range(len(pivot.index)));  ax.set_yticklabels(pivot.index)
plt.colorbar(im, ax=ax, label='F1-Score')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val < 0.4 else 'black')
ax.set_title("Heatmap F1-Score — Maladies x Provinces", fontsize=13)
plt.tight_layout()
plt.savefig('heatmap_maladies_provinces.png', dpi=150, bbox_inches='tight')
plt.show()



# SAUVEGARDE

print("  Resultats exportes  : resultats_multi_maladies.csv")



def alerte_precoce(maladie, province, precip, humidite, temperature):
    """
    maladie     : str  ex. 'PALUDISME', 'CHOLERA', 'ROUGEOLE'
    province    : str  ex. 'KATANGA', 'KINSHASA'
    precip      : float  precipitations au lag optimal (mm/jour)
    humidite    : float  humidite relative au lag optimal (%)
    temperature : float  temperature au lag optimal (C)
    """
    if maladie not in tous_modeles or province not in tous_modeles.get(maladie, {}):
        return None, None, None, f"Modele indisponible : {maladie} / {province}"

    rf     = tous_modeles[maladie][province]
    scaler = tous_scalers[maladie][province]
    tau    = tous_seuils[maladie][province]

    X      = scaler.transform([[precip, humidite, temperature]])
    proba  = rf.predict_proba(X)[0, 1]
    alerte = int(proba >= tau)

    if   proba >= 0.70: niveau, action = "ROUGE",  "Distributions d'urgence, mobiliser personnel"
    elif proba >= 0.50: niveau, action = "ORANGE", "Prepositionnement intrants, renforcer surveillance"
    elif proba >= 0.30: niveau, action = "JAUNE",  "Sensibilisation communautaire, suivi des cas"
    else:               niveau, action = "VERT",   "Surveillance de routine"

    return alerte, round(proba, 3), niveau, action


# Demo
print("\n" + "=" * 70)
print("  DEMONSTRATION — Scenario debut de saison des pluies")
print("  Precip=6mm | Humidite=80% | Temperature=23.5C")
print("=" * 70)
for maladie in maladies_a_traiter:
    print(f"\n  {maladie}")
    for province in provinces_a_analyser:
        lags = LAGS_PAR_MALADIE[maladie]
        res  = alerte_precoce(maladie, province, 6.0, 80.0, 23.5)
        if res[0] is not None:
            alerte, proba, niveau, action = res
            print(f"    {province:12s} : {proba:.0%}  ->  {niveau}")
