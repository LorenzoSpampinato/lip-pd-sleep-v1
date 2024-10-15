import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend
import antropy as ant


# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Ottieni la frequenza di campionamento
fs = raw.info['sfreq']
print(f"Frequenza di campionamento: {fs} Hz")

# Seleziona il canale E8
raw_e8 = raw.copy().pick_channels(['E8'])

# Applica detrend e rimuovi il valore medio (per segnale grezzo)
data, times = raw_e8[:]
data_detrended = detrend(data, axis=1)

# Crea un nuovo oggetto Raw con i dati detrendati e senza valore medio
info = raw_e8.info
raw_e8_detrended = mne.io.RawArray(data_detrended, info)

# --- Applica il filtro passa-banda (0.5-40 Hz) ---
raw_e8_detrended.filter(l_freq=0.35, h_freq=40, method='fir',
                        fir_window='hamming', fir_design='firwin', verbose=False)

# --- Crea epoche di 30 secondi ---
epoch_duration = 30  # Durata delle epoche in secondi
events = mne.make_fixed_length_events(raw_e8_detrended, duration=epoch_duration)
epochs = mne.Epochs(raw_e8_detrended, events, event_id=None, tmin=0,
                    tmax=epoch_duration, baseline=None, preload=True)

# Calcola la sample entropy per ogni epoca
sample_entropy_values = []
for epoch in epochs:
    se = ant.sample_entropy(epoch[0])  # Calcola la sample entropy
    sample_entropy_values.append(se)

# Crea un DataFrame per memorizzare i risultati
df_entropy = pd.DataFrame({'Epoch': range(len(sample_entropy_values)),
                            'Sample Entropy': sample_entropy_values})

# Visualizza il DataFrame
print(df_entropy)

