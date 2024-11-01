import os
import numpy as np
import mne

# Percorsi dei file
data_path = 'D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012'  # Percorso della cartella senza estensione
percorso_cartella = f"{data_path}.mff"
signal2_path = f"{data_path}.mff/signal2.bin"  # Percorso corretto del file EOG

# Step 1: Carica i dati EEG
print(f"Loading EEG data from: {percorso_cartella}")
raw = mne.io.read_raw_egi(percorso_cartella, preload=True, verbose=False)

# Step 2: Carica il segnale EOG dal file signal2.bin
print(f"Loading EOG data from: {signal2_path}")
eog_data = np.fromfile(signal2_path, dtype=np.float64)

# Verifica la lunghezza dei dati EOG
n_samples = eog_data.shape[0]
expected_length = raw.n_times  # Lunghezza attesa deve corrispondere al numero di campioni in raw
print('n_samples:', n_samples)
print('expected_length:', expected_length)

# Controlla se la lunghezza dei dati EOG corrisponde alla lunghezza attesa
#if n_samples != expected_length:
#    raise ValueError(f"Length of EOG data ({n_samples}) does not match length of EEG data ({expected_length}).")

# Step 3: Reshape i dati EOG
# Assicurati che i dati EOG siano nella forma corretta (n_samples, n_channels)
eog_data = eog_data.reshape(-1, 1)  # Assicurati che sia in forma (n_samples, n_channels)
print('EOG data shape:', eog_data.shape)

# Step 4: Crea il canale EOG e aggiungilo a raw
eog_info = mne.create_info(ch_names=['EOG'], sfreq=raw.info['sfreq'], ch_types=['eog'])
eog_raw = mne.io.RawArray(eog_data.T, eog_info)  # Transponi per avere forma (1, n_samples)
raw.add_channels([eog_raw], force_update_info=True)

print("EOG channel added successfully.")

# Step 5: Salva i dati combinati come .edf
edf_output_file_path = os.path.join(data_path + '_signal_eeg_eog.edf')
print(f"Saving combined EEG and EOG data to: {edf_output_file_path}")
mne.export.export_raw(edf_output_file_path, raw, fmt='edf', overwrite=True)

# Step 6: Salva i dati combinati come .bin
bin_output_file_path = os.path.join(data_path + '_signal_eeg_eog.bin')
print(f"Saving combined EEG and EOG data to: {bin_output_file_path}")
raw_data = raw.get_data()  # Ottieni i dati come un array NumPy
raw_data.tofile(bin_output_file_path)  # Salva nel file binario

print("Finished processing subject.")



