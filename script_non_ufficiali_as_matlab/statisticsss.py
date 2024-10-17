import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Lista di file e categorie corrispondenti per i dati delle features
file_ADV = 'D:\\TESI\\lid-data-samples\\lid-data-samples\\Results\\Features\\ADV\\PD002\\PD002_all_feats.npz'
file_CTL = 'D:\\TESI\\lid-data-samples\\lid-data-samples\\Results\\Features\\CTL\\PD009\\PD009_all_feats.npz'
file_DNV = 'D:\\TESI\\lid-data-samples\\lid-data-samples\\Results\\Features\\DNV\\PD005\\PD005_all_feats.npz'
file_DYS = 'D:\\TESI\\lid-data-samples\\lid-data-samples\\Results\\Features\\DYS\\PD012\\PD012_all_feats.npz'
files = [file_ADV, file_CTL, file_DNV, file_DYS]
categories = ['ADV', 'CTL', 'DNV', 'DYS']

all_data = []

# Loop per caricare e processare i dati di ciascun paziente
for file, category in zip(files, categories):
    npzfile = np.load(file)

    # Estrai i dati (epoche, features, regioni)
    data = npzfile['data']  # Shape: (6, 61, 868)
    print(f"Data shape for {category}: {data.shape}")
    features = npzfile['feats']  # Lista delle features
    regions = npzfile['regions']  # Lista delle regioni cerebrali

    # Carica il file degli sleep stage corrispondente
    patient_id = file.split('\\')[-1].split('_')[0]  # Ottieni l'ID paziente
    sleep_stage_file = f'D:\\TESI\\lid-data-samples\\lid-data-samples\\Labels\\{category}\\{patient_id}.npy'

    # Controlla se il file esiste e carica i dati
    try:
        sleep_stages = np.load(sleep_stage_file)
    except FileNotFoundError:
        print(f"File not found: {sleep_stage_file}")
        continue  # Salta al prossimo file se il file non esiste

    # Verifica se le dimensioni delle epoche coincidono
    if data.shape[2] != sleep_stages.shape[0]:
        print(
            f"Dimension mismatch for {patient_id}: data has {data.shape[2]} epochs, sleep_stages has {sleep_stages.shape[0]} epochs. Skipping this patient.")
        continue  # Salta a questo paziente se le dimensioni non corrispondono

    # Loop su ciascuna epoca e regione
    for region_idx in range(data.shape[0]):  # Loop sulle regioni (6 regioni)
        for epoch_idx in range(data.shape[2]):  # Loop sulle epoche
            row = data[region_idx, :, epoch_idx]  # Ottieni tutte le features per una singola epoca e regione

            # Crea una riga di dati con paziente, categoria, epoca, regione e tutte le features
            all_data.append([patient_id, category, epoch_idx, regions[region_idx], sleep_stages[epoch_idx]] + list(row))

# Crea un DataFrame con tutte le epoche, regioni e pazienti
columns = ['PatientID', 'Categoria', 'Epoca', 'Regione', 'SleepStage'] + list(features)
combined_data = pd.DataFrame(all_data, columns=columns)

# Visualizza le prime righe del DataFrame
print(combined_data.head())

# Modifica il PatientID per avere solo il nome del paziente (es. 'PD002')
combined_data['PatientID'] = combined_data['PatientID'].apply(lambda x: x.split('\\')[-1].split('_')[0])

# Ora puoi filtrare il DataFrame in base a 'Sleep Stage' o 'Categoria' come desideri
print(combined_data.head())  # Mostra i dati per verifica

# Filtra i dati per le epoche 301-400 e per le categorie DYS e CTL
filtered_data = combined_data[(combined_data['Categoria'].isin(['DYS', 'CTL'])) &
                              (combined_data['Epoca'].between(301, 400))]

# Inizializza un dizionario per memorizzare i risultati ANOVA
anova_results = {}

# Loop attraverso tutte le colonne delle features (escludendo le colonne non numeriche)
feature_columns = filtered_data.columns[5:]  # Escludi 'PatientID', 'Categoria', 'Epoca', 'Regione', 'Sleep Stage'

for feature in feature_columns:
    # Raggruppa i dati per categoria e crea una lista per ANOVA
    groups = [group[feature].values for name, group in filtered_data.groupby('Categoria')]

    # Esegui ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)

    # Memorizza i risultati
    anova_results[feature] = {'F-statistic': f_statistic, 'p-value': p_value}

# Converti i risultati in un DataFrame
anova_results_df = pd.DataFrame(anova_results).T

# Assicurati che tutte le colonne siano numeriche
anova_results_df['F-statistic'] = pd.to_numeric(anova_results_df['F-statistic'], errors='coerce')
anova_results_df['p-value'] = pd.to_numeric(anova_results_df['p-value'], errors='coerce')

# Aggiungi una colonna per il significato (0.05 come soglia)
anova_results_df['Significance'] = anova_results_df['p-value'] < 0.05

# Rimuovi eventuali righe con NaN
anova_results_df = anova_results_df.dropna()

# Visualizza i risultati come un heatmap per una migliore interpretazione
plt.figure(figsize=(12, 8))
sns.heatmap(anova_results_df[['F-statistic', 'p-value', 'Significance']], annot=True, cmap='coolwarm', cbar=True)
plt.title("Risultati ANOVA per tutte le Features tra DYS e CTL (Epoche 301-400)")
plt.show()
