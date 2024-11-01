import mne
import EntropyHub as eh
import numpy as np

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_cropped.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Seleziona i canali desiderati
raw = raw.pick_channels(['E8', 'E9', 'E10'])


# Preprocessamento (se necessario)
# Qui puoi aggiungere le operazioni di preprocessamento del segnale

# Definisci la funzione per segmentare il segnale in epoche
def segment_epochs(raw):
    # Definisci gli eventi ogni 30 secondi
    events = mne.make_fixed_length_events(raw, duration=30.0)

    # Segmenta il segnale in epoche di 30 secondi
    epochs = mne.Epochs(raw=raw, events=events, tmin=0.0, tmax=30.0, baseline=None, preload=True, verbose=False)
    return epochs


# Segmenta il segnale in epoche
epochs = segment_epochs(raw)

# Ottieni i dati da tutte le epoche (n_epochs, n_channels, n_samples)
epochs_data = epochs.get_data()


# Funzione per calcolare le metriche di entropia su ogni canale
def calculate_entropy_metrics(epoch_data):
    """
    Calcola diverse metriche di entropia sui dati di ogni canale dell'epoca.
    :param epoch_data: Dati EEG per l'epoca, forma (n_channels, n_samples)
    :return: Dizionario con le varie entropie per ogni canale
    """
    n_channels = epoch_data.shape[0]
    entropies = {f'Channel_{i + 1}': {} for i in range(n_channels)}

    for i in range(n_channels):
        channel_data = epoch_data[i, :]  # Seleziona il singolo canale (un vettore 1D)

        # Fuzzy Entropy
        entropies[f'Channel_{i + 1}']['FuzzyEn'],_,_ = eh.FuzzEn(channel_data, m=1, r=(0.15*np.std(channel_data), 3))

        # Multiscale Entropy
        Mobj = eh.MSobject('FuzzEn', m=1, tau=1,
                        Fx="default", r=(0.15*np.std(channel_data), 3))
        entropies[f'Channel_{i + 1}']['MSEn'], _ = eh.MSEn(channel_data, Mbjx=Mobj, Scales=3, Methodx='coarse')

    return entropies


# Calcola le entropie per ogni epoca e canale
for epoch_data in epochs_data:
    # epoch_data ha dimensione (n_channels, n_samples) per ciascuna epoca
    entropy_metrics = calculate_entropy_metrics(epoch_data)

    # Stampa o salva i risultati come desiderato
    print(entropy_metrics)


