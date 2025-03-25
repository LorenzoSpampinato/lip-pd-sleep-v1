import numpy as np
import os
from utilities import EEGRegionsDivider
import matplotlib.pyplot as plt
from numba import config
config.DISABLE_JIT = True
import mne



class EEGDataLoader:
    def __init__(self, data_path, save_path, only_class=None, only_patient=None):

        self.data_path = data_path
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()
        self.save_path = save_path
        self.only_class = only_class
        self.only_patient = only_patient

        # Channel names and reference setup
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        #self.names[self.names.index('E257')] = 'Vertex Reference'

    def load_and_prepare_data(self, file_path, output_dir, file_name_base):
        """Load, prepare, and export EEG data."""
        #file_path= "D:\TESI\lid-data-samples\lid-data-samples\Dataset\DYS\PD012.mff\PD012_cropped.fif"
        raw = self.load_raw_data(file_path)
        #self.export_data(raw, output_dir, file_name_base)
        return raw

    def load_raw_data(self, file_path):
        """
        Load EEG data based on the file extension (.set, .fif, or .mff).

        Args:
            file_path (str): Path to the file to load.

        Returns:
            raw (mne.io.Raw): Loaded RAW object.
        """
        print(f"Caricamento del file grezzo: {file_path}")
        patient_name = os.path.basename(file_path.rstrip('/'))

        # Costruisci i percorsi possibili per diversi formati
        fif_file = os.path.join(file_path, f"{patient_name}.fif")
        set_file = os.path.join(file_path, f"{patient_name}.set")
        mff_file = os.path.join(file_path)

        # Prova a caricare il file FIF
        if os.path.exists(fif_file):
            print(f"Caricamento del file FIF per il paziente: {patient_name}")
            raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            print(f"La frequenza di campionamento è: {sfreq} Hz")
            n_times = raw.n_times
            print(f"Il numero totale di time points è: {n_times}")
            duration_in_seconds = raw.n_times / raw.info['sfreq']
            duration_in_hours = duration_in_seconds / 3600

            print(f"La durata totale del segnale è: {duration_in_hours:.2f} ore")


        elif os.path.exists(set_file):
            print(f"Caricamento del file SET per il paziente: {patient_name}")
            raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
            sfreq = raw.info['sfreq']  # Frequenza di campionamento
            durata_totale_sec = raw.n_times / sfreq  # Durata totale in secondi
            durata_totale_min = durata_totale_sec / 60  # Durata totale in minuti

            print(f"La frequenza di campionamento è: {sfreq} Hz")
            print(f"Durata totale del segnale: {durata_totale_min:.2f} minuti")

            # Rinomina i canali se necessario
            if all(ch_name.startswith("EEG") and ch_name.split()[-1].isdigit() for ch_name in raw.ch_names):
                mapping = {ch_name: f"E{int(ch_name.split()[-1])}" for ch_name in raw.ch_names}
                raw.rename_channels(mapping)
                print("Nomi dei canali aggiornati al formato 'E{numero}'.")

            # Calcolo del numero di epoche da 30 secondi
            epoch_length_sec = 30  # Durata di ogni epoca in secondi
            num_epochs = int(np.floor(durata_totale_sec / epoch_length_sec))

            print(f"Numero di epoche di 30 secondi: {num_epochs}")

        # Prova a caricare il file MFF
        elif os.path.exists(mff_file):
            print(f"Caricamento del file MFF per il paziente: {patient_name}")
            raw = mne.io.read_raw_egi(mff_file, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            print(f"La frequenza di campionamento è: {sfreq} Hz")

        else:
            raise FileNotFoundError(
                f"Non è stato trovato alcun file FIF, SET o MFF per il paziente: {patient_name} nel percorso: {file_path}")

        # Stampa alcune informazioni sul file caricato
        print(f"Numero di punti totali per canale: {raw.n_times}")
        print(f"Nomi dei canali: {raw.ch_names}")

        # Create a directory to save plots
        plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(plot_dir, exist_ok=True)

        print("Saving raw EEG data plot with subplots...")

        # Define channels for visualization and the time range
        channels_to_plot = ['E36', 'E224', 'E59', 'E183','E116']
        start_time = 10710  # seconds
        duration = 30  # seconds
        fs = int(raw.info['sfreq'])  # Sampling frequency

        # Calculate start and end indices
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)

        # Extract data for selected channels and time range
        data, times = raw.copy().pick(channels_to_plot).get_data(return_times=True)
        data = data[:, start_idx:end_idx]
        times = times[start_idx:end_idx]

        # Create subplots
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 2 * len(channels_to_plot)), sharex=True)

        for i, ax in enumerate(axes):
            ax.plot(times, data[i] * 1e6)  # Scale signal to µV
            ax.set_title(f"Channel: {channels_to_plot[i]}")
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True)

        # Set the x-label for the last subplot
        axes[-1].set_xlabel("Time (s)")

        # Adjust layout and save the figure
        plt.tight_layout()
        raw_plot_path = os.path.join(plot_dir, "raw_data_plot_subplots.png")
        plt.savefig(raw_plot_path)
        plt.close()

        print(f"Raw EEG data plot saved to {raw_plot_path}")

        return raw

    def export_data(self, raw, output_dir, file_name_base, overwrite=True):
        """
        Export raw EEG data to FIF format, organizing files in a preprocessed_bad_epochs structure.

        Args:
            raw (mne.io.Raw): Oggetto RAW contenente i dati EEG.
            output_dir (str): Directory base dove salvare i file esportati.
            file_name_base (str): Nome base del file.
            overwrite (bool): Se True, sovrascrive i file esistenti.
        """
        # Verifica che `only_class` e `only_patient` siano specificati
        if not self.only_class or not self.only_patient:
            raise ValueError("Entrambi 'only_class' e 'only_patient' devono essere specificati per salvare i file.")

        # Costruisci il percorso specifico per classe e paziente nella cartella preprocessed_bad_epochs
        preprocessed_dir = os.path.join(output_dir, "preprocessed_bad_epochs", self.only_class, self.only_patient)

        # Crea la directory, se non esiste
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Copia dei dati raw
        selected_raw = raw.copy()

        # Percorso per il salvataggio in formato FIF
        fif_path = os.path.join(preprocessed_dir, f"{os.path.basename(file_name_base)}.fif")

        # Salvataggio in formato FIF
        selected_raw.save(fif_path, overwrite=overwrite)
        print(f"Dati salvati in formato FIF in: {fif_path}")
