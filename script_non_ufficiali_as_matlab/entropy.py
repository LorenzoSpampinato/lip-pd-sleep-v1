'''
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
        ''' '''
        # Fuzzy Entropy
        entropies[f'Channel_{i + 1}']['FuzzyEn'],_,_ = eh.FuzzEn(channel_data, m=1, r=(0.15*np.std(channel_data), 3))

        # Multiscale Entropy
        Mobj = eh.MSobject('FuzzEn', m=1, tau=1,
                        Fx="default", r=(0.15*np.std(channel_data), 3))
        entropies[f'Channel_{i + 1}']['MSEn'], _ = eh.MSEn(channel_data, Mbjx=Mobj, Scales=3, Methodx='coarse')

        # Fuzzy Entropy
        entropies[f'Channel_{i + 1}']['FuzzyEn'],_,_ = eh.FuzzEn(channel_data, m=2)

        # Multiscale Entropy
        Mobj = eh.MSobject('FuzzEn', m=2, tau=1,
                        Fx="default")
        entropies[f'Channel_{i + 1}']['MSEn'], _ = eh.MSEn(channel_data, Mbjx=Mobj, Scales=3, Methodx='coarse')
    return entropies


# Calcola le entropie per ogni epoca e canale
for epoch_data in epochs_data:
    # epoch_data ha dimensione (n_channels, n_samples) per ciascuna epoca
    entropy_metrics = calculate_entropy_metrics(epoch_data)

    # Stampa o salva i risultati come desiderato
    print(entropy_metrics)
'''
import mne
import EntropyHub as eh
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_cropped.fif"

# Load the file with preload=True
raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

# Select the desired channel (e.g., 'E8')
raw = raw.pick_channels(['E8'])


# Define function to segment the signal into epochs
def segment_epochs(raw):
    # Define events every 30 seconds
    events = mne.make_fixed_length_events(raw, duration=30.0)

    # Segment the signal into 30-second epochs
    epochs = mne.Epochs(raw=raw, events=events, tmin=0.0, tmax=30.0, baseline=None, preload=True, verbose=False)
    return epochs


# Segment the signal into epochs
epochs = segment_epochs(raw)

# Get data from all epochs (n_epochs, n_channels, n_samples)
epochs_data = epochs.get_data()


# Custom Fuzzy Entropy function with specified parameters
def fuzzy_entropy(channel_data, m=1, r_ratio=0.15, r2=5):
    """
    Calculate Fuzzy Entropy for EEG data.

    Parameters:
    - channel_data: numpy array, EEG data for one channel.
    - m: int, embedding dimension (default=1).
    - r_ratio: float, ratio for setting r as r_ratio * std of channel data.
    - r2: int, fuzziness parameter for the exponential.

    Returns:
    - fuzzy_entropy: float, calculated fuzzy entropy value.
    """
    N = len(channel_data)
    std_dev = np.std(channel_data)
    r = r_ratio * std_dev

    # Step 1: Create m-dimensional vectors
    vectors = np.array([channel_data[i:i + m] for i in range(N - m + 1)])

    # Step 2: Calculate distances and similarity degrees
    similarity_sum = 0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d_ij = np.max(np.abs(vectors[i] - vectors[j]))  # max absolute difference
            similarity = np.exp(-(d_ij / r) ** r2)  # fuzzy similarity degree
            similarity_sum += similarity
            count += 1

    # Step 3: Calculate fuzzy entropy
    if count > 0:
        fuzzy_entropy_value = -np.log(similarity_sum / count)
    else:
        fuzzy_entropy_value = np.nan

    return fuzzy_entropy_value


# Calculate fuzzy entropy for the first 20 epochs and the selected channel
fuzzy_entropies = []

for epoch_index, epoch_data in enumerate(epochs_data[:20]):  # Process only the first 20 epochs
    # epoch_data has dimension (n_channels, n_samples) for each epoch
    channel_data = epoch_data[0, :]  # Select the first channel (e.g., 'E8')
    fuzzy_entropy_value = fuzzy_entropy(channel_data)
    fuzzy_entropies.append(fuzzy_entropy_value)

    # Print the fuzzy entropy for each epoch
    print(f"Epoch {epoch_index + 1}: Fuzzy Entropy = {fuzzy_entropy_value}")

# Plot the fuzzy entropy for the first 20 epochs
plt.figure(figsize=(10, 6))
plt.plot(fuzzy_entropies, label='Fuzzy Entropy')
plt.title('Fuzzy Entropy for Channel E8 (First 20 Epochs)')
plt.xlabel('Epochs')
plt.ylabel('Fuzzy Entropy')
plt.legend()
plt.show()



