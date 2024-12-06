import numpy as np
import pandas as pd

# Parametri
sampling_rate = 128  # Frequenza di campionamento in Hz
epoch_duration = 30  # Durata di ogni epoca in secondi
samples_per_epoch = sampling_rate * epoch_duration  # Campioni per epoca

# Carica il file CSV
file_path = r"C:\Users\Lorenzo\Downloads\stagesPD017.csv"
data = pd.read_csv(file_path, header=None)  # Nessun header presente
scores = data.iloc[:, 0]  # Considera solo la prima colonna (A1)

# Raggruppa i dati in epoche da 30 secondi
epochs = []
for epoch_index, i in enumerate(range(0, len(scores), samples_per_epoch)):
    epoch_data = scores[i:i+samples_per_epoch]
    if not epoch_data.empty:
        unique_values = epoch_data.unique()
        if len(unique_values) > 1:
            print(f"Epoca {epoch_index}: valori diversi trovati {unique_values}")
        # Assumiamo che, se i valori sono uniformi, prendiamo il primo valore
        epochs.append(epoch_data.iloc[0])

# Converti in array NumPy
epoch_scores = np.array(epochs)

# Salva in formato .npy
output_npy= r"C:\Users\Lorenzo\Downloads\PD017"
np.save(output_npy, epoch_scores)

print(f"File salvato come: {output_npy}")

# Carica il file .npy salvato
loaded_data = np.load(output_npy + ".npy")

# Mostra lo shape dei dati caricati
print(f"Shape del file .npy caricato: {loaded_data.shape}")
