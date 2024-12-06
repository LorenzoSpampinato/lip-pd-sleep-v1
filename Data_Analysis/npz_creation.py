
import numpy as np
import os
import random
'''
# Definizione dei nomi delle feature
features = [
    '1 Spectral energy', '2 Relative delta power band', '3 Relative theta power band',
    '4 Relative alpha power band', '5 Relative alpha1 power band',
    '6 Relative alpha2 power band', '7 Relative alpha3 power band',
    '8 Relative sigma power band', '9 Relative beta power band',
    '10 Relative beta1 power band', '11 Relative beta2 power band',
    '12 Relative gamma power band', '13 theta-delta power ratio',
    '14 theta-beta power ratio', '15 alpha-delta power ratio',
    '16 alpha-theta power ratio', '17 alpha-beta power ratio',
    '18 alpha3-alpha2 power ratio', '19 Spectral mean', '20 Spectral variance',
    '21 Spectral skewness', '22 Spectral kurtosis', '23 Spectral centroid',
    '24 Spectral crest factor', '25 Spectral flatness', '26 Spectral rolloff',
    '27 Spectral spread', '28 Mean', '29 Variance', '30 Skewness',
    '31 Kurtosis', '32 Zero-crossings', '33 Hjorth mobility',
    '34 Hjorth complexity', '35 Spectral entropy', '36 Renyi entropy',
    '37 Approximate entropy', '38 Sample entropy',
    '39 Singular value decomposition entropy', '40 Permutation entropy',
    '41 De-trended fluctuation analysis exponent', '42 Lempel–Ziv complexity',
    '43 Katz fractal dimension', '44 Higuchi fractal dimension',
    '45 Petrosian fractal dimension'
]

# Numero di regioni, epoche e feature
regions = ["Fp", "F", "C", "T", "P", "O"]
num_regions = len(regions)
num_features = len(features)
num_epochs = 140

# Definizione degli stadi del sonno
stages = ['W', 'N1', 'N2', 'N3', 'R']

# Creazione di un array casuale con shape (6, 45, 160)
# Valori casuali generati con distribuzione uniforme tra 0 e 1
data = np.random.rand(num_regions, num_features, num_epochs)

# Creazione degli stadi casuali per ogni epoca e regione
stage_data = np.array([[random.choice(stages) for _ in range(num_epochs)] for _ in range(num_regions)])

# Percorso della directory dove salvare il file
output_dir = r"D:\TESI\lid-data-samples\lid-data-samples\npz files"
os.makedirs(output_dir, exist_ok=True)  # Crea la directory se non esiste

# Salvataggio nel file NPZ
file_name = "PD008_all_feats.npz"
file_path = os.path.join(output_dir, file_name)
np.savez(file_path, regions=regions, features=features, data=data, stages=stage_data)

print(f"File {file_name} salvato con successo in {file_path}.")
'''
#####################################################################################################

import numpy as np

# Percorso del file da leggere
file_path = r"D:\TESI\lid-data-samples\lid-data-samples\npz files\Features\ADV\PD001\PD001_all_feats.npz"

# Caricamento dei dati dal file .npz
loaded_data = np.load(file_path)

# Accesso ai dati salvati
regions = loaded_data['regions']  # Lista delle regioni
features = loaded_data['features']  # Lista delle feature
data = loaded_data['data']  # Array dei dati (shape: 6, 45, 160)
stages = loaded_data['stages']  # Array degli stadi (shape: 6, 160)

# Stampa delle informazioni di base
print("Regioni:", regions)
print("Feature:", features)
print("Shape dei dati:", data.shape)  # Controlla la forma dell'array
print("Shape degli stadi:", stages.shape)  # Controlla la forma dell'array degli stadi

# Esempio: Accedi ai dati della prima regione, prima epoca e prima feature
first_region = regions[0]
first_epoch_data = data[0, 0, :]  # Dati della prima epoca nella prima regione
first_epoch_stage = stages[0, 0]  # Stadio della prima epoca nella prima regione

# Stampa un esempio dei dati e stadi
print(f"\nEsempio di dati della prima regione ({first_region}), prima epoca:")
print("Dati (tutte le epoche per la prima regione, prima feature):")
print(first_epoch_data)

print(f"\nStadio della prima epoca nella regione {first_region}: {first_epoch_stage}")

# Visualizza anche i primi 5 dati delle regioni
print("\nPrime 5 regioni:")
for i in range(min(5, len(regions))):  # Stampa fino a 5 regioni
    print(f"Regione {i+1}: {regions[i]}")

# Mostra i primi 5 stadi
print("\nPrime 5 epoche con il relativo stadio:")
for i in range(min(5, stages.shape[1])):  # Stampa i primi 5 stadi per le epoche
    print(f"Epoca {i+1}: Stadio {stages[0, i]}")

