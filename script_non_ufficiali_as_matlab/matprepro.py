# import mne
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# # Imposta i parametri del filtro
# sfreq = 250.0  # Frequenza di campionamento in Hz
# l_freq = 0.3  # Frequenza di taglio inferiore per il filtro passa-banda
# h_freq = 35.0  # Frequenza di taglio superiore per il filtro passa-banda
#
# # Crea un filtro e ottieni i parametri
# data = np.empty((1, 5000))  # Dati fittizi per evitare il warning di lunghezza
# filter_params = mne.filter.create_filter(
#     data=data,
#     sfreq=sfreq,
#     l_freq=l_freq,
#     h_freq=h_freq,
#     fir_design='firwin'
# )
#
# # Calcolo dell'ordine del filtro
# filter_order = len(filter_params) - 1
#
# # Stampa dettagli del filtro per il documento
# print("Informazioni dettagliate sul filtro FIR applicato:")
# print(f"- Tipo di filtro: FIR (Finite Impulse Response), con fase lineare.")
# print(f"- Finestra: Hamming, che riduce l'ondulazione in banda passante (circa 1.94%) e garantisce un'attenuazione in banda di stop di circa 53 dB.")
# print(f"- Frequenza di campionamento: {sfreq} Hz, ovvero la frequenza a cui è stato registrato il segnale EEG.")
# print(f"- Passa-alto (frequenza di taglio inferiore): {l_freq} Hz. Questa frequenza preserva le componenti a bassa frequenza importanti del segnale EEG.")
# print(f"- Passa-basso (frequenza di taglio superiore): {h_freq} Hz. Impostata per attenuare le componenti ad alta frequenza come gli artefatti muscolari.")
# print(f"- Ordine del filtro: {filter_order} campioni.")
# print(f"- Banda di transizione:")
# print(f"  - Bordo inferiore di taglio: circa 0.15 Hz (-6 dB), con una banda di transizione di 0.3 Hz.")
# print(f"  - Bordo superiore di taglio: circa 39.38 Hz (-6 dB), con una banda di transizione di 8.75 Hz.")
# print(f"- Attenuazione in banda di stop: 53 dB, che riduce efficacemente i componenti fuori dalla banda passante per mantenere il segnale pulito.")
#
# # Visualizza la risposta in frequenza del filtro
# w, h = freqz(filter_params, worN=8000, fs=sfreq)
# plt.plot(w, 20 * np.log10(abs(h)), label="Risposta del filtro (dB)")
# plt.axhline(-6, color='red', linestyle='--', label="-6 dB (bordi di taglio)")
# plt.xlabel('Frequenza (Hz)')
# plt.ylabel('Ampiezza (dB)')
# plt.legend(loc="best")
# plt.title("Risposta in frequenza del filtro passa-banda (0.3 - 35 Hz)")
# plt.show()

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Parametri del filtro
sfreq = 250.0  # Frequenza di campionamento in Hz
l_freq = 0.3  # Frequenza di taglio inferiore
h_freq = 35.0  # Frequenza di taglio superiore

# Crea il filtro FIR
data = np.empty((1, 5000))  # Dati fittizi
filter_params = mne.filter.create_filter(
    data=data,
    sfreq=sfreq,
    l_freq=l_freq,
    h_freq=h_freq,
    fir_design='firwin'
)
# Visualizza la risposta in frequenza del filtro
mne.viz.plot_filter(filter_params, sfreq=sfreq, freq=None, gain='both')
# Calcola la risposta in frequenza

w, h = freqz(filter_params, worN=8000, fs=sfreq)

# Grafico della risposta in frequenza
plt.figure(figsize=(12, 6))
plt.plot(w, 20 * np.log10(abs(h)), label="Risposta del filtro (dB)", color="blue")
plt.axhline(-6, color='red', linestyle='--', label="-6 dB (bordi di taglio)")


# Evidenziazione della banda di transizione per il passa-alto e passa-basso
plt.fill_betweenx(
    y=[-60, 5],
    x1=0.15, x2=0.3,
    color='red', alpha=0.2
)
plt.fill_betweenx(
    y=[-60, 5],
    x1=35, x2=39.38,
    color='red', alpha=0.2, label="Banda di transizione"
)

# Impostazioni dei limiti dell'asse per ingrandire la banda utile
plt.xlim(0, 50)
plt.ylim(-60, 5)
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza (dB)')
plt.title("Risposta in frequenza del filtro passa-banda (0.3 - 35 Hz)")
plt.legend()
plt.grid()
plt.show()
# Calcola la risposta in frequenza del filtro
w, h = freqz(filter_params, worN=8000, fs=sfreq)

# Trova l'indice della frequenza più vicina a 0.1 Hz
freq_index = np.argmin(np.abs(w - 0.15))

# Calcola l'attenuazione in decibel a 0.1 Hz
attenuation_db = 20 * np.log10(abs(h[freq_index]))

# Visualizza il risultato
print(f"Attenuazione a 0.1 Hz: {attenuation_db:.2f} dB")