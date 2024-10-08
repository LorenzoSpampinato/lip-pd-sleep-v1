import mne
import yasa
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
from utilities.hypnogram import hypnogram_definition
from lidpd_main import get_args

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Seleziona il canale E8
raw_e8 = raw.copy().pick_channels(['E8'])

# Applica detrend e rimuovi il valore medio
data, times = raw_e8[:]
data_detrended = detrend(data, axis=1)

# Crea un nuovo oggetto Raw con i dati detrendati
info = raw_e8.info
raw_e8_detrended = mne.io.RawArray(data_detrended, info)

# Applica filtro passa-banda (0.5-40 Hz)
raw_e8_detrended.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

# Crea epoche di 30 secondi
epoch_duration = 30  # Durata delle epoche in secondi
events = mne.make_fixed_length_events(raw_e8_detrended, duration=epoch_duration)
epochs = mne.Epochs(raw_e8_detrended, events, event_id=None, tmin=0, tmax=epoch_duration, baseline=None, preload=True)

# Utilizza YASA per la classificazione automatica del sonno
sls = yasa.SleepStaging(raw_e8_detrended, eeg_name='E8')

# Esegui la classificazione
hypno = sls.predict()  # L'output dovrebbe essere simile a quello che hai fornito

# Mappatura delle etichette a valori interi
stage_labels = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
hypno_int = np.array([stage_labels[label] for label in hypno])

# Plotta l'ipnogramma
fig, ax = plt.subplots(figsize=(10, 4))
yasa.plot_hypnogram(hypno_int, ax=ax)
plt.title("Ipnogramma delle fasi del sonno")
plt.xlabel("Tempo (s)")
plt.ylabel("Fase del sonno")
plt.show()

# Stampa l'ipnogramma (valori stringa)
print("Ipnogramma (valori stringa):", hypno)

# Conta quante volte appare ogni fase del sonno
unique_stages, counts = np.unique(hypno, return_counts=True)
stage_counts = dict(zip(unique_stages, counts))

print("Conteggio delle fasi del sonno:")
for stage, count in stage_counts.items():
    print(f"{stage}: {count}")

args = get_args()
hypnogram_definition(args.label_path, args.run_hypnogram)

#############################################################################################
#####################   Comparison between YASA and Labels   ################################
#############################################################################################
#############################################################################################
# Percorso del file con l'ipnogramma da hypnogram_definition
hypnogram_file_path = "D:/TESI/lid-data-samples/lid-data-samples/Labels/DYS/PD012.npy"
# Carica l'ipnogramma prodotto da hypnogram_definition
hypnogram_from_file = np.load(hypnogram_file_path)

# Verifica che hypno_int sia definito
if 'hypno_int' not in locals():
    raise ValueError("L'ipnogramma di YASA (hypno_int) non è definito.")

# Trova la lunghezza minima tra i due ipnogrammi
min_length = min(len(hypnogram_from_file), len(hypno_int))

# Truncare gli ipnogrammi per avere la stessa lunghezza
hypnogram_from_file_truncated = hypnogram_from_file[:min_length]
hypno_int_truncated = hypno_int[:min_length]

# Confronto
comparison = hypnogram_from_file_truncated == hypno_int_truncated

# Calcola la percentuale di concordanza
similarity_percentage = np.mean(comparison) * 100

# Visualizza i risultati
plt.figure(figsize=(12, 6))

# Plotta l'ipnogramma di hypnogram_definition
plt.subplot(2, 1, 1)
plt.plot(hypnogram_from_file_truncated, label='Hypnogram da hypnogram_definition', color='blue')
plt.title('Ipnogramma da hypnogram_definition')
plt.xlabel('Epochs (30s)')
plt.ylabel('Fase del sonno')
plt.xticks(ticks=np.arange(len(hypnogram_from_file_truncated)), labels=np.arange(len(hypnogram_from_file_truncated)))
plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['Awake', 'N1', 'N2', 'N3', 'REM'])
plt.legend()

# Plotta l'ipnogramma di YASA
plt.subplot(2, 1, 2)
plt.plot(hypno_int_truncated, label='Hypnogram da YASA', color='orange')
plt.title('Ipnogramma da YASA')
plt.xlabel('Epochs (30s)')
plt.ylabel('Fase del sonno')
plt.xticks(ticks=np.arange(len(hypno_int_truncated)), labels=np.arange(len(hypno_int_truncated)))
plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['Awake', 'N1', 'N2', 'N3', 'REM'])
plt.legend()

plt.tight_layout()
plt.show()

# Stampa la percentuale di similarità
print(f"Percentuale di concordanza tra gli ipnogrammi: {similarity_percentage:.2f}%")

