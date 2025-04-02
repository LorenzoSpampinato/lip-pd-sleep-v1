import numpy as np
from mne.time_frequency import psd_array_multitaper
import mne
import antropy as ant
import time

from utilities import EEGRegionsDivider


class SingleChannelFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg, save_path=None, only_stage=None, only_patient=None):
        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg
        self.win_sec = 6

        self.feats = [
            # PSD Features
            'Absolute Low Delta Power', 'Absolute Low Delta1 Power', 'Absolute High Delta Power',
            'Absolute Total Delta Power', 'Absolute Lower Delta Power', 'Absolute General Delta Power',
            'Absolute Low Theta Power', 'Absolute High Theta Power', 'Absolute Total Theta Power',
            'Absolute Alpha Power', 'Absolute Sigma Power', 'Absolute Low Beta Power',
            'Absolute High Beta Power', 'Absolute Total Beta Power', 'Absolute Gamma Power',
            'Relative Low Delta Power', 'Relative Low Delta1 Power', 'Relative High Delta Power',
            'Relative Total Delta Power', 'Relative Lower Delta Power', 'Relative General Delta Power',
            'Relative Low Theta Power', 'Relative High Theta Power', 'Relative Total Theta Power',
            'Relative Alpha Power', 'Relative Sigma Power', 'Relative Low Beta Power',
            'Relative High Beta Power', 'Relative Total Beta Power', 'Relative Gamma Power',
            'Mean Power',

            # Complexity / Entropy Features (calcolate con antropy)
            'Higuchi FD',
            'Sample Entropy',
            'Approximate Entropy',
            'Spectral Entropy',
            'Permutation Entropy',
        ]

        # Inizializza la matrice delle feature (dimensioni: n_canali x n_feature x n_epoche)
        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_patient = only_patient
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.all_channels_features_matrix = []

    def extract_features(self, average_channels=False, specific_channels=None):

        def _compute_psd_features(psds, freqs):
            bands = {
                "Low Delta": (1.5, 2.0), "Low Delta1": (0.5, 2.0), "High Delta": (2.0, 4.0), "Total Delta": (1.5, 4.0),
                "Lower Delta": (0.5, 1.5), "General Delta": (0.5, 4.0), "Low Theta": (4.0, 6.0),
                "High Theta": (6.0, 8.0), "Total Theta": (4.0, 8.0), "Alpha": (8.0, 12.0),
                "Sigma": (12.0, 15.0), "Low Beta": (15.0, 22.0), "High Beta": (22.0, 30.0),
                "Total Beta": (15.0, 30.0), "Gamma": (30.0, 40.0)
            }
            # Calcola le potenze assolute per ciascuna banda
            abs_powers = [np.sum(psds[:, (freqs >= low) & (freqs <= high)], axis=-1)
                          for _, (low, high) in bands.items()]
            # Potenza totale nell'intervallo 0.5-40 Hz
            total_power = np.sum(psds[:, (freqs >= 0.5) & (freqs <= 40.0)], axis=-1)
            # Calcola le potenze relative
            rel_powers = [abs_power / total_power for abs_power in abs_powers]
            # Calcola la potenza media nell'intervallo 0.5-40 Hz
            mean_power = np.mean(psds[:, (freqs >= 0.5) & (freqs <= 40.0)], axis=-1)
            # Restituisce un array di forma (31, n_epoche)
            return np.array(abs_powers + rel_powers + [mean_power])

        def _compute_complexity_features(x_sig_30):
            """
            Calcola le feature di complessità e entropia per ogni epoca usando antropy:
              - Higuchi Fractal Dimension
              - Sample Entropy
              - Approximate Entropy
              - Spectral Entropy (con metodo 'welch' e normalizzazione)
              - Permutation Entropy (con normalizzazione)
            """
            return np.array([
                [ant.higuchi_fd(epoch) for epoch in x_sig_30],
                [ant.sample_entropy(epoch) for epoch in x_sig_30],
                [ant.app_entropy(epoch) for epoch in x_sig_30],
                [ant.spectral_entropy(epoch, sf=self.fs, method='welch',
                                      nperseg=int(6 * self.fs), normalize=True)
                 for epoch in x_sig_30],
                [ant.perm_entropy(epoch, normalize=True) for epoch in x_sig_30],
            ])

        def _prepare_channels():
            if average_channels:
                selected_channels = set()
                for region in self.regions:
                    region_channels = region.split('=')[1].strip().split(', ')
                    valid_channels = [ch for ch in region_channels if ch in self.names]
                    selected_channels.update(valid_channels)
                return list(selected_channels)
            else:
                return specific_channels if specific_channels else self.names

        selected_channels = _prepare_channels()
        # Seleziona solo i canali desiderati e poi taglia alle prime 3 epoche
        epochs_chs = self.epochs.copy().pick(picks=selected_channels, verbose=False)

        # Aggiorna la dimensione della matrice delle feature in base alle epoche selezionate
        features_matrix = np.zeros([len(selected_channels), len(self.feats), len(epochs_chs)])
        channel_names = selected_channels
        all_channels_features_matrix = []

        # Ciclo su ciascun canale
        for nch, ch in enumerate(selected_channels):
            # Estrae i dati per il canale: shape (n_epoche, n_campionamenti)
            x_sig_30 = epochs_chs.get_data()[:, nch, :]

            # Calcola la PSD per ciascuna epoca usando multitaper
            psds_list = []
            for epoch in x_sig_30:
                psd, freqs = psd_array_multitaper(
                    epoch, sfreq=self.fs, fmin=0.5, fmax=40, adaptive=True,
                    normalization='full', verbose=False
                )
                psds_list.append(psd)
            # psds: shape (n_epoche, n_freqs)
            psds = np.array(psds_list)

            # Calcola le feature PSD (31 feature: 15 assolute, 15 relative, 1 media)
            features_matrix[nch, :31, :] = _compute_psd_features(psds, freqs)

            # Calcola le feature di complessità/entropia con antropy
            start_time = time.time()
            features_matrix[nch, 31:, :] = _compute_complexity_features(x_sig_30)
            print(f"Tempo estrazione entropie per il canale {ch}: {time.time() - start_time:.2f} sec")

            all_channels_features_matrix.append(features_matrix[nch, :, :])

        all_channels_features_matrix = np.stack(all_channels_features_matrix, axis=0)

        # Se richiesto, media le feature per regione
        if average_channels:
            region_matrices = []
            for region in self.regions:
                region_channels = region.split('=')[1].strip().split(', ')
                valid_channels = [ch for ch in region_channels if ch in channel_names]
                if valid_channels:
                    idxs = [channel_names.index(ch) for ch in valid_channels]
                    region_matrix = np.mean(features_matrix[idxs, :, :], axis=0)
                    region_matrices.append(region_matrix)
            if region_matrices:
                features_matrix = np.stack(region_matrices, axis=0)

        return features_matrix, self.feats, all_channels_features_matrix, channel_names
