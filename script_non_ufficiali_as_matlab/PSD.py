import mne
from scipy.signal import detrend, welch
from mne_icalabel import label_components
from feature_extraction import EEGFeatureExtractor
from input_script import get_args
import numpy as np
import matplotlib.pyplot as plt

# Ottieni gli argomenti
args = get_args()

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_cropped.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Seleziona solo i canali desiderati
raw = raw.pick_channels(['E16', 'E22', 'E23', 'E24', 'E28', 'E29', 'E30'])  # Canali E8, E9, E10
raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg', verbose=False)

# Detrend e filtro
data, _ = raw[:, :]
raw._data = detrend(data, axis=1)
raw.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

# Applica ICA
ica = mne.preprocessing.ICA(n_components=None, method='fastica', verbose=False, random_state=0)
ica.fit(raw, verbose=False)

# Label ICA components
ic_labels = label_components(raw, ica, method="iclabel")
exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]

# Applica ICA escludendo i componenti di artefatti
raw = ica.apply(raw, exclude=exclude_idx, verbose=False)

# Inizializza l'estrattore di caratteristiche EEG
extractor = EEGFeatureExtractor(args.data_path, args.label_path, args.save_path, args.run_preprocess,
                                args.run_bad_interpolation)

# Segmenta le epoche
epochs = extractor._segment_epochs(raw)

# Specifica l'indice dell'epoca da analizzare
epoch_idx = 5  # Indice della sesta epoca (zero-based)

# Crea la figura
plt.figure(figsize=(12, 10))

# Estrai i dati per il canale E8
x_sig = epochs.get_data()[epoch_idx, 0, :]  # Dati della sesta epoca per il canale E8

# Calcola la PSD usando il primo metodo (finestra di 2 secondi)
f1, psd1 = welch(x=(x_sig - np.mean(x_sig)),  # Detrend la singola epoca
                 fs=raw.info['sfreq'], window='hamming', nperseg=int(3 * raw.info['sfreq']),  # Window di 2 secondi
                 noverlap=int((3 * raw.info['sfreq']) / 2),  # 50% overlap
                 average='median')

# Calcola la PSD usando il secondo metodo (finestra di 5 secondi)
nperseg_5s = int(5 * raw.info['sfreq'])
f2, psd2 = welch(x=x_sig,  # Usando il secondo metodo
                 fs=raw.info['sfreq'], window='hamming', nperseg=nperseg_5s,  # Window di 5 secondi
                 average='median')

# Normalizza la PSD
max_psd1 = np.max(psd1)
psd_normalized1 = psd1 / max_psd1 if max_psd1 > 0 else psd1  # Normalizza solo se max è maggiore di zero

max_psd2 = np.max(psd2)
psd_normalized2 = psd2 / max_psd2 if max_psd2 > 0 else psd2  # Normalizza solo se max è maggiore di zero

# Plotta i risultati
# Subplot per PSD normalizzata
plt.subplot(2, 1, 1)
plt.plot(f1, psd_normalized1, label='Finestra 3s overlap (Normalizzata)')
plt.plot(f2, psd_normalized2, label='Finestra 5s (Normalizzata)')
plt.axvline(x=0.5, color='r', linestyle='--', label='0.5 Hz')  # Linea verticale per 0.5 Hz
plt.title('PSD Normalizzata - Canale E16 (Epoca 6)')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('PSD Normalizzata (V^2/Hz)')
plt.xlim([0, 40])
plt.ylim([0, 1.1])  # Limiti per visualizzare meglio la normalizzazione
plt.grid()
plt.legend()

# Subplot per PSD non normalizzata
plt.subplot(2, 1, 2)
plt.plot(f1, psd1, label='Finestra 3s overlap (Non Normalizzata)')
plt.plot(f2, psd2, label='Finestra 5s (Non Normalizzata)')
plt.axvline(x=0.5, color='r', linestyle='--', label='0.5 Hz')  # Linea verticale per 0.5 Hz
plt.title('PSD Non Normalizzata - Canale E16 (Epoca 6)')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.xlim([0, 40])
plt.grid()
plt.legend()

# Mostra il grafico
plt.tight_layout()  # Ottimizza il layout
plt.show()

