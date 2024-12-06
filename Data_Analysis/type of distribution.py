import pandas as pd
from scipy.stats import shapiro

# Carica il dataset
df = pd.read_csv(r'C:\Users\Lorenzo\Desktop\all_subjects_features_with_period.csv')  # Carica i tuoi dati reali

# Filtrare per STAGE N3
df_n3 = df[df['Stage'] == 'N3']

# Test di normalità (Shapiro-Wilk) per ogni combinazione di gruppo, periodo e area cerebrale
print("Test di normalità (Shapiro-Wilk):")
for group in df_n3['Group'].unique():
    for period in df_n3['Period'].unique():
        for region in df_n3['Brain region'].unique():
            # Filtrare i dati per la combinazione corrente
            data = df_n3[(df_n3['Group'] == group) &
                         (df_n3['Period'] == period) &
                         (df_n3['Brain region'] == region)]['35 Spectral entropy']
            # Verifica che ci siano abbastanza dati
            data_length = len(data)
            if data_length >= 3:
                stat, p = shapiro(data)
                print(f"Group: {group}, Period: {period}, Region: {region}, Data points: {data_length}, p-value: {p:.12f}")
            else:
                print(f"Group: {group}, Period: {period}, Region: {region}, Data points: {data_length} - Non abbastanza dati per il test di normalità.")
print("###################################")

import pandas as pd
from scipy.stats import shapiro

# Carica il dataset
df = pd.read_csv(r'C:\Users\Lorenzo\Desktop\all_subjects_features_with_period.csv')  # Carica i tuoi dati reali

# Filtrare per STAGE N3
df_n3 = df[df['Stage'] == 'N3']

# Calcolare la media di '35 Spectral entropy' per ogni soggetto, gruppo, periodo e regione cerebrale
subject_means = df_n3.groupby(['Group', 'Period', 'Brain region', 'Subject'])['35 Spectral entropy'].mean().reset_index()

# Test di normalità (Shapiro-Wilk) sulla distribuzione delle medie per ogni combinazione di gruppo, periodo e regione cerebrale
print("Test di normalità (Shapiro-Wilk) sulla distribuzione delle medie per ogni soggetto:")
for group in subject_means['Group'].unique():
    for period in subject_means['Period'].unique():
        for region in subject_means['Brain region'].unique():
            # Filtrare i dati per la combinazione corrente
            data = subject_means[(subject_means['Group'] == group) &
                                 (subject_means['Period'] == period) &
                                 (subject_means['Brain region'] == region)]['35 Spectral entropy']
            # Verifica che ci siano abbastanza dati
            data_length = len(data)
            if data_length >= 3:
                stat, p = shapiro(data)
                print(f"Group: {group}, Period: {period}, Region: {region}, Data points: {data_length}, p-value: {p:.20f}")
            else:
                print(f"Group: {group}, Period: {period}, Region: {region}, Data points: {data_length} - Non abbastanza dati per il test di normalità.")
