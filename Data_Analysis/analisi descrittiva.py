import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento del dataset
df = pd.read_csv(r'D:\TESI\lid-data-samples\lid-data-samples\npz files\all_subjects_features.csv')

# Visualizza le prime righe per verificare il formato
df.head()

#1. Analisi e Plot per Paziente Singolo
#1.1. Calcolare le statistiche descrittive per ciascun paziente


# Verifica se le colonne sono correttamente etichettate
feature_columns = [col for col in df.columns if col.split()[0].isdigit()]

# Statistiche descrittive per ogni paziente
patient_stats = df.groupby(['Group', 'Subject', 'Brain region', 'Sleep Stage'])[feature_columns].describe().T

# Reset dell'indice per avere un dataframe più leggibile
patient_stats.reset_index(inplace=True)

# Salvataggio delle statistiche in un file CSV
patient_stats.to_csv(r'D:\TESI\lid-data-samples\lid-data-samples\npz files\analisi_pazienti_stats.csv', index=False)

# Conferma che il file è stato salvato correttamente
print("File CSV salvato con successo!")


# Visualizzare alcune righe delle statistiche
print(patient_stats.head())

#1.2. Creare un boxplot o violin plot per ogni paziente

# Creiamo un boxplot per una delle feature
feature = '1 Spectral energy'

# Boxplot per ogni paziente
plt.figure(figsize=(12, 6))
sns.boxplot(x='Subject', y=feature, data=df, hue='Group', palette='Set2')
plt.title(f'Boxplot per {feature} per ogni paziente')
plt.xlabel('Paziente')
plt.ylabel(feature)
plt.xticks(rotation=45)
plt.show()
