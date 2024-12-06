import numpy as np
import mne
import os
from utilities import EEGRegionsDivider
import matplotlib.pyplot as plt


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
        self.names[self.names.index('E257')] = 'Vertex Reference'

    def load_and_prepare_data(self, file_path, output_dir, file_name_base):
        """Load, prepare, and export EEG data."""
        #file_path= "D:\TESI\lid-data-samples\lid-data-samples\Dataset\DYS\PD012.mff\PD012_cropped.fif"
        raw = self.load_raw_data(file_path)
        self.add_bad_epochs(raw, output_dir,file_name_base)
        #self.export_data(raw, output_dir, file_name_base)
        return raw

    def load_raw_data(self, file_path):
        """
        Load EEG data based on the file extension (.set, .fif, or .edf).

        Args:
            file_path (str): Path to the file to load.

        Returns:
            raw (mne.io.Raw): Loaded RAW object.
        """
        # Controlla l'estensione del file
        # Ottieni il nome del paziente (es. 'PD036') dalla cartella
        patient_name = os.path.basename(file_path.rstrip('/'))

        # Costruisci i percorsi possibili per diversi formati
        fif_file = os.path.join(file_path, f"{patient_name}_raw.fif")
        print('fif_file:', fif_file)
        set_file = os.path.join(file_path, f"{patient_name}.set")

        # Prova a caricare il file FIF
        if os.path.exists(fif_file):
            print(f"Caricamento del file FIF per il paziente: {patient_name}")
            raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

        # Prova a caricare il file SET
        elif os.path.exists(set_file):
            print(f"Caricamento del file SET per il paziente: {patient_name}")
            raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

            # Rinomina i canali se necessario
            if all(ch_name.startswith("EEG") and ch_name.split()[-1].isdigit() for ch_name in raw.ch_names):
                mapping = {ch_name: f"E{int(ch_name.split()[-1])}" for ch_name in raw.ch_names}
                raw.rename_channels(mapping)
                print("Nomi dei canali aggiornati al formato 'E{numero}'.")

        else:
            raise FileNotFoundError(
                f"Non è stato trovato alcun file FIF o SET per il paziente: {patient_name} nel percorso: {file_path}")

        # Stampa alcune informazioni sul file caricato
        print(f"Numero di punti totali per canale: {raw.n_times}")
        print(f"Nomi dei canali: {raw.ch_names}")

        # Create a directory to save plots
        plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(plot_dir, exist_ok=True)

        print("Saving raw EEG data plot with subplots...")

        # Define channels for visualization and the time range
        channels_to_plot = ['E27', 'E224', 'E59', 'E55', 'E76', 'E116']
        start_time = 60*60+60  # seconds
        duration = 60  # seconds
        fs = int(raw.info['sfreq'])  # Sampling frequency

        # Calculate start and end indices
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)

        # Extract data for selected channels and time range
        data, times = raw.copy().pick_channels(channels_to_plot).get_data(return_times=True)
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

    import os

    def add_bad_epochs(self, raw, output_dir, file_name_base):
        """Aggiungi epoche cattive basate sui file .npy nella cartella di errori."""
        # Imposta la cartella di errori in base al paziente e al gruppo
        errors_dir = os.path.join(
            self.save_path, "PLOT", self.only_class, self.only_patient, "ICA_individual_epochs", "errors"
        )

        if not os.path.exists(errors_dir):
            print(f"Non sono stati trovati file di errore in {errors_dir}.")
            return

        # Lista delle epoche cattive
        bad_epochs = []

        # Itera attraverso i file .npy e prendi il numero dell'epoca
        for file in os.listdir(errors_dir):
            if file.endswith(".npy"):
                # Estrai l'epoca dal nome del file, ad esempio 'epoch_360_data.npy' -> 360
                epoch_number = int(file.split("_")[1].split(".")[0])
                bad_epochs.append(epoch_number)

        if bad_epochs:
            print(f"Epoche cattive trovate: {bad_epochs}")

            # Crea un campo privato '_bad_epochs' per memorizzare le epoche cattive
            bad_epochs_info = sorted(set(bad_epochs))  # Ordina e rimuovi duplicati
            raw._bad_epochs = bad_epochs_info
            print(f"Epoche cattive aggiunte a raw._bad_epochs (privato): {raw._bad_epochs}")

        else:
            print("Nessuna epoca cattiva trovata.")

        # Salva il raw con il campo privato _bad_epochs
        self.export_data(raw, output_dir, file_name_base)  # Passa il file_name_base qui

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
