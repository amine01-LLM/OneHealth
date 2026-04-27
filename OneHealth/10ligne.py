import pandas as pd

# Charger le fichier original
# On utilise 'nrows' pour ne lire que les 20 premières lignes (gain de mémoire)
df = pd.read_csv('DRC_ML_Ready_v3.csv', nrows=20)

# Enregistrer dans un nouveau fichier CSV
df.to_csv('extrait_20_lignes.csv', index=False)

print("Extraction terminée avec succès !")