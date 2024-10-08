import mne
import matplotlib.pyplot as plt
from scipy.signal import detrend
import numpy as np


#apply detrending and filtering to the raw signal of channel E8.
# The pre and post filter signal is compared in a plot

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Ottieni la frequenza di campionamento (sampling frequency)
fs = raw.info['sfreq']
print(f"Frequenza di campionamento: {fs} Hz")

# Seleziona il canale E8
raw_e8 = raw.pick_channels(['E8'])

# Detrend dei dati
data, _ = raw_e8[:, :]  # Estrai i dati di E8
data_detrended = detrend(data, axis=1)  # Applica detrend

# Applica il filtro band-pass (FIR) [0.5 - 40 Hz]
raw_e8.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)
#filt_pars=mne.filter.create_filter(data, sfreq=fs, l_freq=0.5, h_freq=40, filter_length='auto', l_trans_bandwidth='auto',
                         #h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero',
                         #fir_window='hamming', fir_design='firwin', verbose=None)
#mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')

# Estrai i dati filtrati
data_filtered, _ = raw_e8[:, :]

# Controlla se i dati sono uguali prima e dopo il filtro
dati_uguali = np.array_equal(data_detrended, data_filtered)

# Stampa il risultato del controllo
if dati_uguali:
    print("I dati sono uguali prima e dopo il filtro.")
else:
    print("I dati sono diversi prima e dopo il filtro.")

# Imposta il tempo per la visualizzazione
start_time = 0  # Inizio della visualizzazione (puoi cambiare questo valore)
duration = 10   # Durata della visualizzazione in secondi

# Crea subplot per visualizzare il segnale grezzo e quello filtrato
plt.figure(figsize=(12, 6))

# Subplot 1: Segnale grezzo (detrended)
plt.subplot(2, 1, 1)
plt.plot(raw_e8.times[:int(duration * raw.info['sfreq'])], data_detrended[0, :int(duration * raw.info['sfreq'])], color='blue')
plt.title('Segnale Grezzo (Detrended) - Canale E8')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.xlim(start_time, start_time + duration)
plt.grid()

# Subplot 2: Segnale filtrato
plt.subplot(2, 1, 2)
plt.plot(raw_e8.times[:int(duration * raw.info['sfreq'])], data_filtered[0, :int(duration * raw.info['sfreq'])], color='red')
plt.title('Segnale Filtrato - Canale E8')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.xlim(start_time, start_time + duration)
plt.grid()

# Mostra i plot
plt.tight_layout()
plt.show()

