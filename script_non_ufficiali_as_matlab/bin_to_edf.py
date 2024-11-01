import os
import glob
import time
import mne
import pyedflib
import numpy as np


def process_and_convert_eeg(data_path, only_class=None, only_patient=None, sfreq=250, exclude_channel="Vertex Reference"):
    # Estrai i percorsi dei file .mff per tutti i soggetti o per uno specifico paziente
    if only_class and only_patient:
        sub_folds = [os.path.join(data_path, only_class, only_patient)]
    else:
        sub_folds = glob.glob(os.path.join(data_path, '*/*'))

    for sub_fold in sub_folds:
        print(f"Start processing: {sub_fold}")  # Inizio della conversione

        # Rinomina il file in .mff se necessario
        if not sub_fold.endswith('.mff'):
            print(f"Renaming {sub_fold} to {sub_fold + '.mff'}...")
            os.rename(sub_fold, sub_fold + '.mff')
            sub_fold = sub_fold + '.mff'
            print(f"End renaming: {sub_fold}")

        # Misura il tempo di caricamento
        start_time_loading = time.time()
        print(f"Loading EEG data from: {sub_fold}")

        # Carica i dati EEG
        raw = mne.io.read_raw_egi(sub_fold, preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)  # Seleziona solo i canali EEG

        print(f"Data loaded in {time.time() - start_time_loading:.2f} seconds.")

        # Escludi il canale di riferimento, se presente
        if exclude_channel in raw.ch_names:
            exclude_idx = raw.ch_names.index(exclude_channel)
            raw.drop_channels([exclude_channel])
            print(f"Channel '{exclude_channel}' excluded from conversion.")

        # Imposta il percorso per il file .edf
        edf_path = sub_fold.replace('.mff', '.edf')

        # Misura il tempo di conversione
        start_time_conversion = time.time()

        # Estrai i dati e le informazioni
        data = raw.get_data()  # Dati EEG
        n_channels, n_samples = data.shape
        ch_names = raw.ch_names  # Nomi dei canali

        # Crea e salva il file .edf
        with pyedflib.EdfWriter(edf_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as edf_file:
            # Definisci le informazioni di ogni canale
            channel_info = []
            for i in range(n_channels):
                physical_min = np.min(data[i])
                physical_max = np.max(data[i])

                channel_info.append({
                    'label': ch_names[i],
                    'dimension': 'uV',
                    'sample_rate': sfreq,
                    'physical_min': float(f"{physical_min:.5f}"),  # Truncate to 5 decimal places
                    'physical_max': float(f"{physical_max:.5f}"),  # Truncate to 5 decimal places
                    'digital_min': -32768,
                    'digital_max': 32767,
                    'transducer': '',
                    'prefilter': ''
                })

            edf_file.setSignalHeaders(channel_info)
            edf_file.writeSamples(data)

        print(f"Conversion completed in {time.time() - start_time_conversion:.2f} seconds.")
        print(f"Total processing time for {sub_fold}: {time.time() - start_time_loading:.2f} seconds.\n")


# Utilizzo della funzione
data_path = r"D:\TESI\lid-data-samples\lid-data-samples\Dataset"
process_and_convert_eeg(data_path, only_class='ADV', only_patient='PD002')

