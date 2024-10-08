import mne
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt

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
# Utilizzo del filtro FIR con finestra di Hamming
raw_e8_detrended.filter(l_freq=0.35, h_freq=40, method='fir',
                        fir_window='hamming', fir_design='firwin', verbose=False)

# --- Crea epoche di 30 secondi ---
epoch_duration = 30  # Durata delle epoche in secondi
events = mne.make_fixed_length_events(raw_e8_detrended, duration=epoch_duration)
epochs = mne.Epochs(raw_e8_detrended, events, event_id=None, tmin=0,
                    tmax=epoch_duration, baseline=None, preload=True)

# Stampa il numero di epoche estratte
print(f"Numero di epoche estratte: {len(epochs)}")

# --- Calcola la PSD normalizzata per ciascuna epoca e plottala ---
for i, epoch in enumerate(epochs):
    # Calcola la PSD per l'epoca corrente
    psd, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq=fs, fmin=0.5, fmax=40, n_fft=1024)

    # Normalizza la PSD per la somma totale della potenza
    psd_normalized = psd / np.sum(psd)

    # Plot della PSD normalizzata per ogni epoca
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs, psd_normalized.T)
    plt.title(f'PSD Normalizzata Epoca {i + 1}')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('PSD Normalizzata (Unità Arbitraria)')
    plt.grid(True)
    plt.show()
