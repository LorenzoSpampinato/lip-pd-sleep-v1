import numpy as np
from mne_connectivity import spectral_connectivity_time


class ChannelConnectivityFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg):
        """
        Initializes the feature extractor with the necessary parameters.

        Parameters:
        - epochs: mne.Epochs object (contains EEG/PSG data)
        - fs: Sampling frequency, in Hz.
        - ch_reg: Dictionary or set with channel names divided by brain region.
        """
        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg
        self.feats = sorted(['46 Phase Locking Value in Delta', '47 Weighted Phase Lag Index in Delta',
                             '48 Phase Locking Value in Theta', '49 Weighted Phase Lag Index in Theta',
                             '50 Phase Locking Value in Alpha', '51 Weighted Phase Lag Index in Alpha',
                             '52 Phase Locking Value in Sigma', '53 Weighted Phase Lag Index in Sigma',
                             '54 Phase Locking Value in Beta', '55 Weighted Phase Lag Index in Beta',
                             '56 Phase Locking Value in Gamma', '57 Weighted Phase Lag Index in Gamma'])
        self.bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 15], [15, 30], [30, 40]])
        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])

    def extract_features(self):
        """
        Extracts channel connectivity features for each epoch and brain region.

        Returns:
        - features_matrix: A matrix of extracted features.
        - feats: List of feature names.
        """
        # Iterate over brain regions
        for nr, region in enumerate(self.ch_reg):
            chs = [ch.strip() for ch in region.split('=')[1].split(',')]
            epochs_chs = self.epochs.copy().pick(picks=chs, verbose=False)

            # Iterate over epochs
            for t_epc in range(len(epochs_chs)):
                x_sig_30 = epochs_chs.get_data()[t_epc, :, :][np.newaxis, :, :]

                # Extract connectivity features for each frequency band
                for nb, (fl, fu) in enumerate(self.bands):
                    self.extract_band_features(nr, nb, t_epc, x_sig_30, fl, fu)

        return self.features_matrix, self.feats

    def extract_band_features(self, nr, nb, t_epc, x_sig_30, fl, fu):
        """
        Extracts features for a specific frequency band for an epoch.

        Parameters:
        - nr: Brain region index.
        - nb: Band index.
        - t_epc: Epoch index.
        - x_sig_30: EEG signal data for the epoch.
        - fl, fu: Frequency band limits (lower and upper).
        """
        # 1. Phase Locking Value (PLV)
        plv = spectral_connectivity_time(x_sig_30, method='plv', sfreq=self.fs, freqs=[fl, fu],
                                         faverage=True, n_cycles=3, verbose=False).get_data()[0, :, :]
        plv_m = plv.reshape(np.size(x_sig_30, 1), np.size(x_sig_30, 1))
        plv_m_no_rep = np.tril(plv_m, k=-1)
        self.features_matrix[nr, int(nb * 2), t_epc] = np.mean(plv_m_no_rep[plv_m_no_rep != 0].flatten())

        # 2. Weighted Phase Lag Index (wPLI)
        wpli = spectral_connectivity_time(x_sig_30, method='wpli', sfreq=self.fs, freqs=[fl, fu],
                                          faverage=True, n_cycles=3, verbose=False).get_data()[0, :, :]
        wpli_m = wpli.reshape(np.size(x_sig_30, 1), np.size(x_sig_30, 1))
        wpli_m_no_rep = np.tril(wpli_m, k=-1)
        self.features_matrix[nr, int(nb * 2) + 1, t_epc] = np.mean(wpli_m_no_rep[wpli_m_no_rep != 0].flatten())
