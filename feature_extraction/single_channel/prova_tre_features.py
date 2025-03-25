import numpy as np
from scipy.signal import welch
import antropy as ant
from utilities import EEGRegionsDivider
import EntropyHub as EH
from pyentrp import entropy as ent
import nolds
import time
import mne

class SingleChannelFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg, save_path=None, only_stage=None, only_patient=None):
        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg
        self.win_sec = 6  # Window length in seconds

        self.feats = [
            # **Power Spectral Density (PSD) Features**
            '0. Absolute Low Delta Power', '1. Absolute Low Delta1 Power', '2. Absolute High Delta Power',
            '3. Absolute Total Delta Power', '4. Absolute Lower Delta Power', '5. Absolute General Delta Power',
            '6. Absolute Low Theta Power', '7. Absolute High Theta Power', '8. Absolute Total Theta Power',
            '9. Absolute Alpha Power', '10. Absolute Sigma Power', '11. Absolute Low Beta Power',
            '12. Absolute High Beta Power', '13. Absolute Total Beta Power', '14. Absolute Gamma Power',

            '15. Relative Low Delta Power', '16. Relative Low Delta1 Power', '17. Relative High Delta Power',
            '18. Relative Total Delta Power', '19. Relative Lower Delta Power', '20. Relative General Delta Power',
            '21. Relative Low Theta Power', '22. Relative High Theta Power', '23. Relative Total Theta Power',
            '24. Relative Alpha Power', '25. Relative Sigma Power', '26. Relative Low Beta Power',
            '27. Relative High Beta Power', '28. Relative Total Beta Power', '29. Relative Gamma Power',

            '30. Mean Power',

            # **Fractal & Complexity Features**
            '37. Higuchi FD',

            '38. ShanEn',
            '39. RenyiEn',

            # **Multiscale Entropy Features**
            '40. AppEn1', '41. AppEn2', '42. AppEn3', '43. AppEnCI',
            '44. SampEn1', '45. SampEn2', '46. SampEn3', '47. SampEnCI',
            '48. FuzEn1', '49. FuzEn2', '50. FuzEn3', '51. FuzEnCI',
            '52. PermEn1', '53. PermEn2', '54. PermEn3', '55. PermEnCI',
        ]

        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_patient = only_patient
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        print(self.regions)
        self.idx_chs = self.divider.get_index_channels()
        # Channel names and reference setup
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        print(self.names)
        self.all_channels_features_matrix = []

    def extract_features(self, average_channels=False, specific_channels=None):
        """
        Extracts features from EEG epochs for specific channels or brain regions.

        Parameters:
        - average_channels: If True, calculates features averaged over brain regions.
                            If False, calculates features for individual channels.
        - specific_channels: List of specific channel names to extract features from.
                             If None, all channels in the dataset are used.

        Returns:
        - features_matrix: A matrix of extracted features (averaged if average_channels=True).
        - feats: List of feature names.
        - all_channels_features_matrix: A matrix of features for all individual channels.
        - channel_names: List of all channel names used for feature extraction.
        """

        def _compute_psd_features(psd, f):
            bands = {
                "Low Delta": (1.5, 2.0), "Low Delta1": (0.5, 2.0), "High Delta": (2.0, 4.0), "Total Delta": (1.5, 4.0),
                "Lower Delta": (0.5, 1.5), "General Delta": (0.5, 4.0), "Low Theta": (4.0, 6.0),
                "High Theta": (6.0, 8.0), "Total Theta": (4.0, 8.0), "Alpha": (8.0, 12.0),
                "Sigma": (12.0, 15.0), "Low Beta": (15.0, 22.0), "High Beta": (22.0, 30.0),
                "Total Beta": (15.0, 30.0), "Gamma": (30.0, 40.0)
            }
            abs_powers = [np.sum(psd[:, (f >= low) & (f <= high)], axis=-1) for _, (low, high) in bands.items()]
            total_power = np.sum(psd[:, (f >= 0.5) & (f <= 40.0)], axis=-1)
            rel_powers = [abs_power / total_power for abs_power in abs_powers]
            mean_power = np.mean(psd[:, (f >= 0.5) & (f <= 40.0)], axis=-1)

            return np.array(abs_powers + rel_powers + [mean_power])

        def _compute_complexity_features(x_sig_30):
            """Calcola le feature di complessità per ogni epoca."""
            return np.array([
                [ant.higuchi_fd(epoch.astype(np.float64)) for epoch in x_sig_30],  # Higuchi Fractal Dimension
                [ent.shannon_entropy(epoch) for epoch in x_sig_30],  # Shannon Entropy
                self.renyi_entropy(x_sig_30, nperseg=int(self.win_sec * self.fs))  # Renyi Entropy
            ])

        def _compute_multiscale_entropy(x_sig_30):
            """Calcola le feature di entropia multiscala."""
            ms_entropy_methods = [
                ('ApEn', {'m': 3, 'tau': 1, 'r': None}),
                ('SampEn', {'m': 3, 'tau': 1}),
                ('FuzzEn', {'m': 3, 'tau': 1, 'r': (0.2, 2.0)}),
                ('PermEn', {'m': 3, 'tau': 1, 'Norm': True}),
            ]

            entropy_results = []
            for entropy_name, params in ms_entropy_methods:
                Mobj = EH.MSobject(EnType=entropy_name, **params)
                entropy_values, CI_values = [], []

                for epoch in x_sig_30:
                    entropy_per_scale, CI = EH.MSEn(epoch, Mobj, Scales=3, Methodx='coarse')
                    entropy_values.append(entropy_per_scale[:3])  # Scala 1, 2, 3
                    CI_values.append(CI)  # Complexity Index

                entropy_values = np.array(entropy_values).T  # 3 Scale Features
                entropy_results.append(entropy_values)
                entropy_results.append(np.array(CI_values))  # Complexity Index

                for i in range(len(entropy_results)):
                    if entropy_results[i].ndim == 1:
                        entropy_results[i] = entropy_results[i].reshape(1, -1)

            print("\nDEBUG: Controllo delle forme prima di concatenare:\n")
            for i, arr in enumerate(entropy_results):
                print(f"Array {i}: shape = {arr.shape}, ndim = {arr.ndim}")

            return np.concatenate(entropy_results)

        # Inizializzazione

        # Se vogliamo la media per regioni, selezioniamo solo i canali delle regioni
        if average_channels:
            selected_channels = set()  # Per evitare duplicati

            for region in self.regions:
                region_channels = region.split('=')[1].strip().split(', ')
                valid_channels = [ch for ch in region_channels if ch in self.names]  # Solo canali validi

                if valid_channels:
                    selected_channels.update(valid_channels)  # Aggiungi canali alla lista

            selected_channels = list(selected_channels)  # Converti in lista unica
            print(f"Canali selezionati per regioni: {selected_channels}")

        else:
            selected_channels = specific_channels if specific_channels else self.names

        # Estrai solo i canali selezionati
        epochs_chs = self.epochs.copy().pick(picks=selected_channels, verbose=False)
        features_matrix = np.zeros([len(selected_channels), len(self.feats), len(self.epochs)])
        channel_names = selected_channels

        all_channels_features_matrix = []

        # Applicare i filtri per ciascuna sottobanda
        for band, (low, high) in {
            "Low Delta": (1.5, 2.0), "Low Delta1": (0.5, 2.0), "High Delta": (2.0, 4.0), "Total Delta": (1.5, 4.0),
            "Lower Delta": (0.5, 1.5), "General Delta": (0.5, 4.0), "Low Theta": (4.0, 6.0),
            "High Theta": (6.0, 8.0), "Total Theta": (4.0, 8.0), "Alpha": (8.0, 12.0),
            "Sigma": (12.0, 15.0), "Low Beta": (15.0, 22.0), "High Beta": (22.0, 30.0),
            "Total Beta": (15.0, 30.0), "Gamma": (30.0, 40.0)
        }.items():
            # Filtra il segnale per ciascun canale
            filtered_epochs = mne.filter.filter_data(epochs_chs.get_data(), sfreq=self.fs, l_freq=low, h_freq=high,
                                                     verbose=False)
            print(f"Dimensioni del segnale filtrato per la banda {band}: {filtered_epochs.shape}")

            # Per ogni canale, calcola la PSD e le caratteristiche
            for nch, ch in enumerate(selected_channels):
                x_sig_30 = filtered_epochs[:, nch, :]  # Seleziona i dati del canale nch per tutte le epoche
                print(x_sig_30.shape)

                # Calcolo della PSD con 'welch'
                f, psd = welch(x_sig_30, fs=self.fs, window='hamming',
                               nperseg=int(self.win_sec * self.fs),
                               noverlap=4 * self.fs, average='median', axis=-1)

                # Calcolare le caratteristiche PSD per il canale 'nch'
                features_matrix[nch, :31, :] = _compute_psd_features(psd, f)
            '''
            # Misura il tempo di esecuzione per _compute_complexity_features
            start_time = time.time()
            features_matrix[nch, 37:40, :] = _compute_complexity_features(x_sig_30)
            complexity_time = time.time() - start_time
            print(f"Tempo _compute_complexity_features per il canale {ch}: {complexity_time:.2f} secondi")

            # Misura il tempo di esecuzione per _compute_multiscale_entropy
            start_time = time.time()
            features_matrix[nch, 40:, :] = _compute_multiscale_entropy(x_sig_30)
            entropy_time = time.time() - start_time
            print(f"Tempo _compute_multiscale_entropy per il canale {ch}: {entropy_time:.2f} secondi")
            '''
            all_channels_features_matrix.append(features_matrix[nch, :, :])

        all_channels_features_matrix = np.stack(all_channels_features_matrix, axis=0)

        # Se vogliamo fare la media per regioni
        if average_channels:
            region_matrices = []

            for region in self.regions:
                region_channels = region.split('=')[1].strip().split(', ')
                print("Region channels: ", region_channels)
                valid_channels = [ch for ch in region_channels if ch in channel_names]
                print("Valid channels: ", valid_channels)

                if valid_channels:
                    idxs = [channel_names.index(ch) for ch in valid_channels]
                    print("Indexes: ", idxs)
                    region_matrix = np.mean(features_matrix[idxs, :, :], axis=0)
                    print("Region matrix: ", region_matrix)
                    region_matrices.append(region_matrix)

            if region_matrices:
                features_matrix = np.stack(region_matrices, axis=0)
                print(f"Features mediate per regioni: {features_matrix.shape}")

        return features_matrix, self.feats, all_channels_features_matrix, channel_names

    def renyi_entropy(self, x, nperseg, normalize=True):
        """Calcola la Renyi Entropy."""
        _, psd = welch(x, self.fs, nperseg=nperseg, axis=-1)
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        renyi = -np.log(np.sum(psd_norm ** 2, axis=-1)) / np.log(2)
        return renyi / np.log2(psd_norm.shape[-1]) if normalize else renyi


