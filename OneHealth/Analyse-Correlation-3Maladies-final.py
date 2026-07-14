import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = pd.read_csv('DRC_Health_Weather_CLEANED.csv')

# Configuration
province_etude = 'KINSHASA'
maladies_etude = ['PALUDISME', 'ROUGEOLE', 'CHOLERA', 'FIEVRE JAUNE']
variables_climat = ['PRECTOTCORR', 'T2M', 'RH2M']
max_lag = 12  # Analyser jusqu'à 12 semaines de décalage

colors_maladie = {
    'PALUDISME': '#1f77b4',      
    'ROUGEOLE': '#ff7f0e',        
    'CHOLERA': '#2ca02c',         
    'FIEVRE JAUNE': '#d62728'     
}

# =============================================================================
# PARTIE 1 : CORRÉLATIONS CROISÉES PAR MALADIE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, maladie in enumerate(maladies_etude):
    ax = axes[idx]
    
    # Filtrer les données
    data = df[(df['PROV'] == province_etude) & (df['MALADIE'] == maladie)].copy()
    data = data.sort_values('DEBUTSEM')
    
    if len(data) == 0:
        ax.set_visible(False)
        print(f"\n Données insuffisantes pour {maladie} à {province_etude}")
        continue
    
    # Calculer les corrélations croisées
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
    
    # Tracer les courbes
    for var, corrs in results.items():
        ax.plot(range(max_lag + 1), corrs, marker='o', label=var, linewidth=2, markersize=4)
    
    ax.set_xlabel('Décalage (semaines)')
    ax.set_ylabel('Corrélation avec l\'incidence')
    ax.set_title(f'{maladie} - {province_etude}', fontweight='bold', color=colors_maladie[maladie])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-1, 1)
    
    # Afficher les meilleurs décalages dans la console
    print(f"\n=== MEILLEURS DÉCALAGES POUR {maladie} À {province_etude} ===")
    for var, corrs in results.items():
        if len(corrs) > 1:
            best_lag = np.argmax(np.abs(corrs[1:])) + 1
            best_corr = corrs[best_lag]
            print(f"  {var:12s} : lag {best_lag:2d} sem  |  corrélation = {best_corr:+.3f}")

plt.suptitle(f'Corrélations croisées climat-incidence par maladie\n{province_etude}', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('correlations_croisees_4maladies.png', dpi=150, bbox_inches='tight')
plt.show()



# PARTIE 2 : MATRICES DE CORRÉLATION GLOBALES PAR MALADIE


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, maladie in enumerate(maladies_etude):
    ax = axes[idx]
    
    data = df[(df['PROV'] == province_etude) & (df['MALADIE'] == maladie)].copy()
    
    if len(data) == 0:
        ax.set_visible(False)
        continue
    
    variables_analyse = ['INCIDENCE', 'PRECTOTCORR', 'T2M', 'RH2M']
    corr_matrix = data[variables_analyse].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f',
                ax=ax, vmin=-1, vmax=1, cbar=True)
    ax.set_title(f'Matrice corrélation - {maladie}\n{province_etude}', 
                 fontweight='bold', color=colors_maladie[maladie])
    
    # Afficher dans la console
    print(f"\n=== MATRICE DE CORRÉLATION {maladie} (sans décalage) ===")
    print(corr_matrix.round(3))

plt.suptitle('Matrices de corrélation globales par maladie', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('matrices_correlation_4maladies.png', dpi=150, bbox_inches='tight')
plt.show()



# PARTIE 3 : TABLEAU RÉCAPITULATIF COMPARATIF


print("\n" + "="*70)
print("  SYNTHÈSE COMPARATIVE - LAGS OPTIMAUX PAR MALADIE")
print("="*70)

recap = []
for maladie in maladies_etude:
    data = df[(df['PROV'] == province_etude) & (df['MALADIE'] == maladie)].copy()
    data = data.sort_values('DEBUTSEM')
    
    if len(data) == 0:
        continue
    
    for var in variables_climat:
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = data['INCIDENCE'].corr(data[var])
            else:
                var_lagged = data[var].shift(lag)
                corr = data['INCIDENCE'].corr(var_lagged)
            correlations.append(corr)
        
        best_lag = np.argmax(np.abs(correlations[1:])) + 1
        best_corr = correlations[best_lag]
        lag0_corr = correlations[0]
        
        recap.append({
            'Maladie': maladie,
            'Variable': var,
            'Lag_optimal': best_lag,
            'Correl_max': round(best_corr, 3),
            'Correl_lag0': round(lag0_corr, 3)
        })

df_recap = pd.DataFrame(recap)
print("\n")
print(df_recap.to_string(index=False))

# Pivot pour meilleure lisibilité
print("\n" + "="*70)
print("  TABLEAU PIVOT - LAGS OPTIMAUX")
print("="*70)
pivot_lag = df_recap.pivot(index='Maladie', columns='Variable', values='Lag_optimal')
print("\nLag optimal (semaines) :")
print(pivot_lag)

pivot_corr = df_recap.pivot(index='Maladie', columns='Variable', values='Correl_max')
print("\nCorrélation maximale :")
print(pivot_corr.round(3))



# PARTIE 4 : VISUALISATION COMPARATIVE DES LAGS OPTIMAUX


fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(variables_climat))
width = 0.2

for i, maladie in enumerate(maladies_etude):
    subset = df_recap[df_recap['Maladie'] == maladie]
    lags = [subset[subset['Variable'] == var]['Lag_optimal'].values[0] 
            for var in variables_climat]
    ax.bar(x + i*width, lags, width, label=maladie, color=colors_maladie[maladie])

ax.set_xlabel('Variables climatiques')
ax.set_ylabel('Lag optimal (semaines)')
ax.set_title(f'Comparaison des lags optimaux par maladie\n{province_etude}', fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(variables_climat)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('lags_optimaux_comparatif.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Analyse terminée ! Graphiques sauvegardés.")
