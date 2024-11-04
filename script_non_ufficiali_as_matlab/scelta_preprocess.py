import mne
import matplotlib.pyplot as plt

# Parametri del filtro
fs = 250  # Frequenza di campionamento in Hz
l_freq = 0.3  # Frequenza di taglio inferiore
h_freq = 35 # Frequenza di taglio superiore
fir_window = 'hamming'  # Tipo di finestra
filter_length='auto'
l_trans_bandwidth=0.2
h_trans_bandwidth='auto'
fir_design='firwin'
method='fir'
phase= 'zero'

# Crea il filtro
filt_pars = mne.filter.create_filter(data=None, sfreq=fs, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length,
                                     l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                     method=method, phase= phase,
                                     fir_window=fir_window, fir_design=fir_design, verbose=None)

# Calcola la larghezza della banda di transizione
#l_trans_bandwidth = min(max(l_freq * 0.25, 2), l_freq)
h_trans_bandwidth = min(max(h_freq * 0.25, 2), fs/2 - h_freq)
#l_trans_bandwidth=0.3
#h_trans_bandwidth=8

# Risposta del filtro in secondi (larghezza del filtro in campioni)
ordine_filtro = len(filt_pars)
# Converti la lunghezza del filtro in secondi
filtro_length_seconds = ordine_filtro / fs

# Calcola la banda di roll-off
rolloff_bandwidth = l_trans_bandwidth + h_trans_bandwidth

# Visualizza la risposta in frequenza del filtro
mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')
plt.show()

# Stampa dei risultati e delle formule
print("Parametri del filtro FIR calcolati:")
print(f" - Frequenza di taglio inferiore: {l_freq} Hz")
print(f" - Frequenza di taglio superiore: {h_freq} Hz")
print(f" - Larghezza della banda di transizione inferiore: {l_trans_bandwidth:.2f} Hz")
print(f" - Larghezza della banda di transizione superiore: {h_trans_bandwidth:.2f} Hz")
print(f" - Ordine del filtro (numero di campioni): {ordine_filtro}")
print(f" - Tipo di finestra: {fir_window}")
print(f" - Risposta del filtro in secondi: {filtro_length_seconds:.4f} s")
print(f" - Banda di roll-off: {rolloff_bandwidth:.2f} Hz")

#####################################################################################################

# Path del file FIF (sostituisci con il percorso reale)
file_path = 'D:\TESI\lid-data-samples\lid-data-samples\Dataset\DYS\PD012.mff\PD012_cropped.fif'

# Carica il file FIF
raw = mne.io.read_raw_fif(file_path, preload=True)

# Applica il filtro FIR al segnale
raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, filter_length=filter_length,
                                     l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                     method=method, phase= phase,
                                     fir_window=fir_window, fir_design=fir_design, verbose=None)

# Seleziona un canale per visualizzare il segnale nel tempo (ad esempio il primo canale)
channel = 0  # Puoi scegliere un canale specifico
start, stop = raw.time_as_index([0,30])  # Intervallo di tempo (in secondi) da visualizzare

# Estrai i dati del segnale originale e filtrato
original_data, times = raw[channel, start:stop]
filtered_data, _ = raw_filtered[channel, start:stop]

# Plot dei segnali nel dominio del tempo
plt.figure(figsize=(12, 6))

# Segnale originale
plt.subplot(2, 1, 1)
plt.plot(times, original_data[0], label="Segnale Originale", color="blue")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza (uV)")
plt.title("Segnale Originale")
plt.legend()

# Segnale filtrato
plt.subplot(2, 1, 2)
plt.plot(times, filtered_data[0], label="Segnale Filtrato", color="orange")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza (uV)")
plt.title("Segnale Filtrato")
plt.legend()

plt.tight_layout()
plt.show()
