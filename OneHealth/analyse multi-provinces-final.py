import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('DRC_Health_Weather_CLEANED.csv')

# Liste des provinces à analyser
provinces_a_analyser = ['KINSHASA', 'EQUATEUR', 'KATANGA', 'SUD-KIVU', 'ORIENTALE']
maladie_etude = 'PALUDISME'
max_lag = 16  # Augmenter à 16 semaines pour capturer des délais plus longs

# Dictionnaire pour stocker les résultats
resultats_provinces = {}

# Fonction pour calculer les corrélations croisées
def analyse_lag_province(df, province, maladie, max_lag):
    # Filtrer les données
    data = df[(df['PROV'] == province) & (df['MALADIE'] == maladie)].copy()
    data = data.sort_values('DEBUTSEM')
    
    if len(data) < 50:  # Pas assez de données
        return None
    
    variables_climat = ['PRECTOTCORR', 'T2M', 'RH2M']
    results = {}
    
    for var in variables_climat:
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = data['INCIDENCE'].corr(data[var])
            else:
                var_lagged = data[var].shift(lag)
                corr = data['INCIDENCE'].corr(var_lagged)
            correlations.append(corr)
        results[var] = correlations
    
    return results

# Calculer pour chaque province
for province in provinces_a_analyser:
    print(f"\n--- Analyse de {province} ---")
    resultats = analyse_lag_province(df, province, maladie_etude, max_lag)
    if resultats:
        resultats_provinces[province] = resultats
        
        # Afficher les meilleurs décalages
        for var, corrs in resultats.items():
            # Trouver le meilleur décalage (exclure lag 0)
            corrs_array = np.array(corrs)
            best_lag = np.argmax(corrs_array[1:]) + 1
            best_corr = corrs_array[best_lag]
            lag0_corr = corrs_array[0]
            
            print(f"  {var}:")
            print(f"    Meilleur décalage : {best_lag} semaines")
            print(f"    Corrélation max : {best_corr:.3f}")
            print(f"    Corrélation à lag 0 : {lag0_corr:.3f}")
    else:
        print(f"  Données insuffisantes pour {province}")

# Visualisation comparative
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

colors = {'PRECTOTCORR': 'blue', 'T2M': 'red', 'RH2M': 'green'}
linestyles = {'PRECTOTCORR': '-', 'T2M': '--', 'RH2M': ':'}

for idx, province in enumerate(provinces_a_analyser[:5]):
    if province in resultats_provinces:
        ax = axes[idx]
        resultats = resultats_provinces[province]
        
        for var, corrs in resultats.items():
            lags = range(len(corrs))
            ax.plot(lags, corrs, label=var, color=colors[var], 
                   linestyle=linestyles[var], linewidth=2, marker='o', markersize=4)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Décalage (semaines)')
        ax.set_ylabel('Corrélation')
        ax.set_title(f'{province} - {maladie_etude}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_lag)

# Supprimer l'axe vide s'il y a moins de 5 provinces
if len(provinces_a_analyser) < 5:
    axes[5].set_visible(False)

plt.suptitle(f'Comparaison des corrélations croisées - {maladie_etude}', fontsize=14)
plt.tight_layout()
plt.show()

# Graphique synthétique des meilleurs décalages par province
best_lags_summary = {}
for province, resultats in resultats_provinces.items():
    best_lags_summary[province] = {}
    for var, corrs in resultats.items():
        best_lag = np.argmax(corrs[1:]) + 1
        best_corr = corrs[best_lag]
        best_lags_summary[province][var] = {'lag': best_lag, 'corr': best_corr}

# Créer un tableau synthétique
summary_df = pd.DataFrame({
    province: {var: f"{best_lags_summary[province][var]['lag']} sem ({best_lags_summary[province][var]['corr']:.2f})"
               for var in ['PRECTOTCORR', 'T2M', 'RH2M']}
    for province in best_lags_summary.keys()
}).T

print("\n=== SYNTHÈSE DES MEILLEURS DÉCALAGES PAR PROVINCE ===\n")
print(summary_df.to_string())

# Barplot comparatif des corrélations maximales
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(resultats_provinces.keys()))
width = 0.25

for i, var in enumerate(['PRECTOTCORR', 'T2M', 'RH2M']):
    corr_values = [best_lags_summary[p][var]['corr'] for p in resultats_provinces.keys()]
    ax.bar(x + i*width, corr_values, width, label=var)

ax.set_xlabel('Province')
ax.set_ylabel('Corrélation maximale')
ax.set_title('Comparaison des corrélations maximales par province')
ax.set_xticks(x + width)
ax.set_xticklabels(resultats_provinces.keys())
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculer les p-values pour KATANGA (la plus forte corrélation)
from scipy import stats

# Pour KATANGA, vérifier si la corrélation à lag 8 est significative
katanga_data = df[(df['PROV'] == 'KATANGA') & (df['MALADIE'] == 'PALUDISME')].copy()
katanga_data = katanga_data.sort_values('DEBUTSEM')

# Créer les séries avec décalage
precip_lag8 = katanga_data['PRECTOTCORR'].shift(8)
incidence = katanga_data['INCIDENCE']

# Nettoyer les NaN
valid_mask = ~(precip_lag8.isna() | incidence.isna())
corr, p_value = stats.pearsonr(precip_lag8[valid_mask], incidence[valid_mask])

print(f"KATANGA - Précipitations lag 8 : corrélation={corr:.3f}, p-value={p_value:.4f}")
print(f"Significatif à 95% : {'OUI' if p_value < 0.05 else 'NON'}")