import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

#save 5% of the signal in local memory,
# view channel 8 in.fif format and
# perform detrending and filtering on all channels


# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Ottieni la frequenza di campionamento (sampling frequency)
fs = raw.info['sfreq']
print(f"Frequenza di campionamento: {fs} Hz")

# Ottieni la durata del segnale in secondi
durata_segnale = len(raw.times) / fs
print(f"Durata del segnale: {durata_segnale:.2f} secondi")

# Visualizza i primi 10 secondi del segnale
#raw.plot(duration=10, n_channels='E8', scalings='auto', show=True)
#plt.show()

# Seleziona il canale E8
raw_e8 = raw.pick_channels(['E8'])

# Visualizza il segnale per il canale E8 dall'istante 100s all'istante 200s
start_time = 100  # Secondi
duration = 100    # Durata (da 100s a 200s)

# Visualizza il plot

raw_e8.plot(duration=duration, start=start_time, scalings='auto',show=True)
plt.show()

# -------------------------------------  2. Average re-reference  --------------------------------------
# -----------------------------------------  3. Trend removal  -----------------------------------------
data, _ = raw[:, :]
raw._data = detrend(data, axis=1)
# Band-pass filter (FIR) [0.1 - 40 Hz]
# To plot filter mask
filt_pars = mne.filter.create_filter(data=None, sfreq=fs, l_freq=.1, h_freq=40,fir_window='hamming', fir_design='firwin')
mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')

raw.filter(l_freq=.1, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

