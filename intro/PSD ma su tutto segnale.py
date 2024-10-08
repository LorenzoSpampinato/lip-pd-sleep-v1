import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Ottieni la frequenza di campionamento
fs = raw.info['sfreq']
print(f"Frequenza di campionamento: {fs} Hz")

# Seleziona il canale E8
raw_e8 = raw.copy().pick_channels(['E8'])

# Estrai i dati del canale E8
data, times = raw_e8[:]

# Applica detrend e rimuovi il valore medio (per segnale grezzo)
data_detrended = detrend(data, axis=1)

# Crea un nuovo oggetto Raw con i dati detrendati e senza valore medio
info = raw_e8.info
raw_e8_detrended = mne.io.RawArray(data_detrended, info)

##########################################################################
# Calcolo e plot della PSD prima del filtro (dati detrendati, senza filtro)
psd_before_filter, freqs_before_filter = raw_e8_detrended.compute_psd().get_data(return_freqs=True)

# Plot della PSD prima del filtro
plt.figure(figsize=(10, 6))
plt.plot(freqs_before_filter, psd_before_filter[0], label='PSD - Prima del filtro')
plt.title('Spettro della densità di potenza (PSD) - Canale E8 prima del filtro')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Densità di potenza (dB/Hz)')
plt.xlim([0, 50])
plt.grid(True)
plt.legend()
plt.show()

##########################################################################
# Applica il filtro band-pass FIR tra 0.5 e 40 Hz ai dati detrendati
raw_e8_detrended.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

##########################################################################
# Calcolo e plot della PSD dopo il filtro (dati filtrati)
psd_after_filter, freqs_after_filter = raw_e8_detrended.compute_psd().get_data(return_freqs=True)

# Plot della PSD dopo il filtro
plt.figure(figsize=(10, 6))
plt.plot(freqs_after_filter, psd_after_filter[0], label='PSD - Dopo il filtro')
plt.title('Spettro della densità di potenza (PSD) - Canale E8 dopo il filtro')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Densità di potenza (dB/Hz)')
plt.xlim([0, 50])
plt.grid(True)
plt.legend()
plt.show()

##########################################################################
# Calcolo della potenza relativa delle bande di frequenza dopo il filtro
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40)  # Opzionale
}

# Inizializza un dizionario per le potenze relative
relative_powers = {}
total_power_after = np.sum(psd_after_filter)  # Potenza totale del segnale filtrato

for band, (low_freq, high_freq) in bands.items():
    # Trova gli indici corrispondenti alle frequenze di interesse
    freq_idx = np.where((freqs_after_filter >= low_freq) & (freqs_after_filter <= high_freq))

    # Calcola la potenza della banda specifica
    band_power = np.sum(psd_after_filter[:, freq_idx])

    # Calcola la potenza relativa
    relative_power = band_power / total_power_after
    relative_powers[band] = relative_power

# Stampa la potenza relativa di ciascuna banda
for band, power in relative_powers.items():
    print(f"Potenza relativa nella banda {band}: {power * 100:.2f}%")
