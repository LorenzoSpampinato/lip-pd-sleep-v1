import numpy as np
import mne

# Definisci i percorsi dei file MFF (EEG) e del segnale 2 (EOG o altro)
percorso_cartella = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff"
percorso_signal2 = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/signal2.bin"


# Carica il file EEG in formato MFF
raw = mne.io.read_raw_egi(percorso_cartella, preload=True, verbose=False)

# Carica il file signal2 (presumibilmente un file binario)
signal2_data = np.fromfile(percorso_signal2, dtype=np.float32)

# Verifica la lunghezza e compatibilità del segnale con il file EEG
#n_samples = raw.n_times
#if len(signal2_data) != n_samples:
#    raise ValueError("La lunghezza del segnale 2 non corrisponde a quella del segnale EEG.")

# Aggiungi il signal2 come un canale separato nell'oggetto raw
signal2_data = signal2_data.reshape(1, n_samples)  # Reshape per avere un solo canale
signal2_info = mne.create_info(ch_names=['Signal2'], sfreq=raw.info['sfreq'], ch_types=['misc'])
signal2_raw = mne.io.RawArray(signal2_data, signal2_info)

# Aggiungi il canale signal2 ai dati EEG
raw.add_channels([signal2_raw], force_update_info=True)

# Definisci la terza ora in secondi (3 ore = 3 * 60 * 60 secondi)
terza_ora_inizio = 3 * 60 * 60
mezz_ora = 0.5 * 60 * 60

# Croppa il file a partire dalla terza ora per mezz'ora
raw_cropped = raw.crop(tmin=terza_ora_inizio, tmax=(terza_ora_inizio + mezz_ora))

# Salva il file croppato in formato .fif
raw_cropped.save(percorso_file_croppato, overwrite=True)

