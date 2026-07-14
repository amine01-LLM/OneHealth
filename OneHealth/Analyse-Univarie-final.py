import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates



df = pd.read_csv("DRC_Health_Weather_CLEANED.csv")

df["DEBUTSEM"] = pd.to_datetime(df["DEBUTSEM"])

# Liste des maladies présentes
maladies = df['MALADIE'].unique()
print("Maladies disponibles :", maladies)

# figure avec sous-graphiques pour chaque maladie
# fig, axes = plt.subplots(len(maladies), 2, figsize=(14, 4*len(maladies)))
# if len(maladies) == 1:
#     axes = [axes]  # Pour gérer le cas d'une seule maladie

# for i, maladie in enumerate(maladies):
#     subset = df[df['MALADIE'] == maladie]
    
#     # Boxplot
#     axes[i][0].boxplot(subset['INCIDENCE'].dropna())
#     axes[i][0].set_title(f'Distribution de l\'incidence - {maladie}')
#     axes[i][0].set_ylabel('Cas pour 1000 habitants')
#     axes[i][0].grid(True, alpha=0.3)
    
#     # Histogramme
#     axes[i][1].hist(subset['INCIDENCE'].dropna(), bins=30, edgecolor='black', alpha=0.7)
#     axes[i][1].set_title(f'Histogramme de l\'incidence - {maladie}')
#     axes[i][1].set_xlabel('Cas pour 1000 habitants')
#     axes[i][1].set_ylabel('Fréquence (nombre de semaines)')
#     axes[i][1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# Statistiques descriptives par maladie
print("\n=== STATISTIQUES DESCRIPTIVES PAR MALADIE ===\n")
for maladie in maladies:
    subset = df[df['MALADIE'] == maladie]['INCIDENCE'].dropna()
    print(f"\n--- {maladie} ---")
    print(f"  Nombre de semaines : {len(subset)}")
    print(f"  Incidence moyenne : {subset.mean():.2f} cas/1000h")
    print(f"  Écart-type : {subset.std():.2f}")
    print(f"  Minimum : {subset.min():.2f}")
    print(f"  Maximum : {subset.max():.2f}")
    print(f"  Médiane : {subset.median():.2f}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Filtrer les 6 maladies les plus pertinentes pour une visualisation claire
maladies_principales = ['PALUDISME', 'FIEVRE JAUNE', 'ROUGEOLE', 'CHOLERA']
df_filtre = df[df['MALADIE'].isin(maladies_principales)]

# Créer une figure avec boxplots comparatifs
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_filtre, x='MALADIE', y='INCIDENCE')
plt.yscale('log')  # Échelle logarithmique pour mieux voir les différences
plt.title('Distribution de l\'incidence par maladie (échelle log)')
plt.ylabel('Incidence (cas/1000h) - échelle logarithmique')
plt.xlabel('Maladie')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# ============================================================
# Maladies étudiées
# ============================================================

maladies = [
    "PALUDISME",
    "ROUGEOLE",
    "CHOLERA"
]

# ============================================================
# Création de la figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
    sharex=True
)

axes = axes.flatten()

for ax, maladie in zip(axes, maladies):

    # Agrégation nationale
    weekly = (
        df[df["MALADIE"] == maladie]
        .groupby("DEBUTSEM")["INCIDENCE"]
        .mean()
        .reset_index()
        .sort_values("DEBUTSEM")
    )

    # Moyenne mobile 13 semaines
    weekly["MA13"] = (
        weekly["INCIDENCE"]
        .rolling(window=13, center=True)
        .mean()
    )

    # Série brute
    ax.plot(
        weekly["DEBUTSEM"],
        weekly["INCIDENCE"],
        alpha=0.35,
        linewidth=0.8,
        label="Incidence hebdomadaire"
    )

    # Tendance lissée
    ax.plot(
        weekly["DEBUTSEM"],
        weekly["MA13"],
        linewidth=2.5,
        label="Moyenne mobile (13 semaines)"
    )

    ax.set_title(maladie.replace("_", " "), fontsize=12)

    ax.grid(alpha=0.3)

    # Affichage d'une année tous les 2 ans
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y')
    )

# ============================================================
# Mise en forme
# ============================================================

fig.suptitle(
    "Évolution temporelle de l'incidence moyenne nationale des maladies étudiées",
    fontsize=16,
    y=0.98
)

for ax in axes:
    ax.set_ylabel(
        "Incidence moyenne\n(cas / 1000 habitants)"
    )

axes[2].set_xlabel("Année")
axes[3].set_xlabel("Année")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.94)
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# ============================================================
# Export haute qualité pour le mémoire
# ============================================================

plt.savefig(
    "evolution_temporelle_3_maladies.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
