import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import antropy as ant
import os
import matplotlib.pyplot as plt
import mne

class SingleChannelFeatureExtractor:
    def __init__(self, epochs, fs, ch_reg, win_sec=3, save_path=None, only_stage=None, only_patient=None):
        """
        Initializes the feature extractor with the necessary parameters.

        Parameters:
        - epochs: mne.Epochs object (contains EEG/PSG data)
        - fs: Sampling frequency in Hz
        - ch_reg: Dictionary of channel names divided by brain region
        - win_sec: Window length in seconds (default = 5)
        """

        self.epochs = epochs
        self.fs = fs
        self.ch_reg = ch_reg
        self.win_sec = win_sec
        self.feats = sorted(['1 Spectral energy', '2 Relative delta power band', '3 Relative theta power band',
                             '4 Relative alpha power band', '5 Relative alpha1 power band',
                             '6 Relative alpha2 power band',
                             '7 Relative alpha3 power band', '8 Relative sigma power band',
                             '9 Relative beta power band',
                             '10 Relative beta1 power band', '11 Relative beta2 power band',
                             '12 Relative gamma power band',
                             '13 theta-delta power ratio', '14 theta-beta power ratio', '15 alpha-delta power ratio',
                             '16 alpha-theta power ratio', '17 alpha-beta power ratio', '18 alpha3-alpha2 power ratio',
                             '19 Spectral mean', '20 Spectral variance', '21 Spectral skewness', '22 Spectral kurtosis',
                             '23 Spectral centroid', '24 Spectral crest factor', '25 Spectral flatness',
                             '26 Spectral rolloff',
                             '27 Spectral spread', '28 Mean', '29 Variance', '30 Skewness', '31 Kurtosis',
                             '32 Zero-crossings',
                             '33 Hjorth mobility', '34 Hjorth complexity', '35 Spectral entropy', '36 Renyi entropy',
                             '37 Approximate entropy', '38 Sample entropy', '39 Singular value decomposition entropy',
                             '40 Permutation entropy', '41 De-trended fluctuation analysis exponent',
                             '42 Lempel–Ziv complexity',
                             '43 Katz fractal dimension', '44 Higuchi fractal dimension',
                             '45 Petrosian fractal dimension'])
        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_patient = only_patient

    def extract_features(self):
        """
        Extracts features from the EEG epochs for each brain region and channel.

        Returns:
        - features_matrix: A matrix of extracted features.
        - feats: List of feature names.
        """

        # Iterate over brain regions
        for nr, region in enumerate(self.ch_reg):
            chs = [ch.strip() for ch in region.split('=')[1].split(',')]
            print("Channel picks:", chs)  # Debugging
            epochs_chs = self.epochs.copy().pick(picks=chs, verbose=False)
            features_matrix_reg = np.zeros([len(chs), len(self.feats), len(self.epochs)])
            print('epochs_chs:', epochs_chs)  # Debugging
            # Iterate over channels within the region
            for nch, ch in enumerate(chs):
                x_sig_30 = epochs_chs.get_data()[:, nch, :]
                print("nch: ", nch)
                # PSD calculation using Welch method
                f, psd = welch(x=x_sig_30,
                               fs=self.fs, window='hamming', nperseg=int(self.win_sec * self.fs),
                               noverlap= int((self.win_sec * self.fs)/2),
                               average='median', axis=-1)

                # Average PSD across epochs
                avg_psd = np.mean(psd, axis=0)

                # Define channel positions
                pos = mne.find_layout(self.epochs.info).pos

                # Plot topomap
                fig, ax = plt.subplots()
                mne.viz.plot_topomap(
                    avg_psd, pos, ch_type='eeg', cmap='viridis', show=False,
                    contours=0, outlines='head', axes=ax
                )
                # Save the figure
                plot_dir = os.path.join(self.save_path, "PSD_topomap", self.only_stage, self.only_patient)
                os.makedirs(plot_dir, exist_ok=True)
                fig.savefig(os.path.join(plot_dir, f"topomap_psd_{ch}.png"))
                plt.close(fig)


                abs_psd = np.abs(psd)
                mean_psd = np.mean(abs_psd, axis=-1)


                # Frequency-domain feature extraction
                features_matrix_reg[nch,0:27,:] = self.extract_frequency_features(f, abs_psd, mean_psd)

                # Time-domain feature extraction
                features_matrix_reg[nch,27:45,:] = self.extract_time_features(x_sig_30, features_matrix_reg[nch])

            # Averaging across channels for each region
            self.features_matrix[nr, :, :] = np.mean(features_matrix_reg, axis=0)

        return self.features_matrix, self.feats

    def extract_frequency_features(self, f, abs_psd, mean_psd):
        """
        Extracts frequency-domain features.

        Parameters:
        - f: Frequencies.
        - abs_psd: Absolute Power Spectral Density.
        - mean_psd: Mean Power Spectral Density.

        Returns:
        - features_vector: A vector of frequency-domain features for a single channel.
        """
        features_frequency_vector = np.zeros([27, len(self.epochs)])

        # Spectral energy, Relative power bands, and Power ratios
        spec_en, p_rel, p_rat = self.spectral(f, abs_psd)
        features_frequency_vector[0,:] = spec_en
        features_frequency_vector[1:12,:] = p_rel
        features_frequency_vector[12:18,:] = p_rat

        # Spectral mean, variance, skewness, kurtosis
        features_frequency_vector[18,:] = mean_psd
        features_frequency_vector[19,:] = np.var(abs_psd, axis=-1)
        features_frequency_vector[20,:] = skew(abs_psd, axis=-1)
        features_frequency_vector[21,:] = kurtosis(abs_psd, axis=-1)

        # Spectral centroid, crest factor, flatness, rolloff, spread
        features_frequency_vector[22,:] = np.sum(f * abs_psd, axis=-1) / spec_en
        features_frequency_vector[23,:] = np.max(abs_psd, axis=-1) / mean_psd
        features_frequency_vector[24,:] = mstats.gmean(abs_psd, axis=-1) / mean_psd
        features_frequency_vector[25,:] = self.rolloff(f, abs_psd)
        features_frequency_vector[26,:] = np.sum(((f - features_frequency_vector[22, :, np.newaxis]) ** 2)* abs_psd, axis=-1) / spec_en

        return features_frequency_vector

    def extract_time_features(self, x_sig_30, features_vector):
        """
        Extracts time-domain features.

        Parameters:
        - x_sig_30: Signal data for 30-second epochs.
        - features_vector: Current vector of features to be populated.

        Returns:
        - features_vector: Updated vector of time-domain features.
        """
        features_time_vector = np.zeros([18, len(self.epochs)])
        # Mean, variance, skewness, kurtosis
        features_time_vector[0,:] = np.mean(x_sig_30, axis=-1)
        features_time_vector[1,:] = np.var(x_sig_30, axis=-1)
        features_time_vector[2,:] = skew(x_sig_30, axis=-1)
        features_time_vector[3,:] = kurtosis(x_sig_30, axis=-1)

        # Zero-crossings and Hjorth parameters
        features_time_vector[4,:] = ant.num_zerocross(x_sig_30, axis=-1)
        features_time_vector[5,:], features_vector[6,:] = ant.hjorth_params(x_sig_30, axis=-1)

        # Entropy measures
        features_time_vector[7,:] = ant.spectral_entropy(x_sig_30, sf=self.fs, method='welch',
                                                   nperseg=int(self.win_sec * self.fs), normalize=True, axis=-1)
        features_time_vector[8,:] = self.renyi_entropy(x_sig_30,  nperseg=int(self.win_sec * self.fs))
        features_time_vector[9,:] = [ant.app_entropy(x) for x in x_sig_30]
        features_time_vector[10,:] = [ant.sample_entropy(x) for x in x_sig_30]
        features_time_vector[11,:] = [ant.svd_entropy(x, normalize=True) for x in x_sig_30]
        features_time_vector[12,:] = [ant.perm_entropy(x, normalize=True) for x in x_sig_30]
        #mobj = eh.MSobject('FuzzEn', m=1, tau=1, Fx="default", r=(0.15 * np.std(x_sig_30[0]), 3))
        #features_vector[40] = [msen for msen, _ in (eh.MSEn(x, Mbjx=mobj, Scales=3, Methodx='coarse') for x in x_sig_30)]

        # Complexity and fractal dimension measures
        features_time_vector[13,:] = [ant.detrended_fluctuation(x) for x in x_sig_30]
        features_time_vector[14,:] = [ant.lziv_complexity(''.join(map(str, np.where(x > np.median(x), 1, 0))),
                                                   normalize=True) for x in x_sig_30]
        features_time_vector[15,:] = ant.katz_fd(x_sig_30, axis=-1)
        features_time_vector[16,:] = [ant.higuchi_fd(x.astype(np.float64)) for x in x_sig_30]
        features_time_vector[17,:] = ant.petrosian_fd(x_sig_30, axis=-1)

        return features_time_vector

    def renyi_entropy(self, x, nperseg, normalize=True):
        _, psd = welch(x, self.fs, nperseg=nperseg, axis=-1)
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        renyi = -np.log(np.sum(psd_norm ** 2, axis=-1)) / np.log(2)
        if normalize:
            renyi /= np.log2(psd_norm.shape[-1])
        return renyi

    def spectral(self, f, abs_psd):
        bands = np.array([[0.5, 4], [4, 8], [8, 12], [8, 9], [9, 11], [11, 12],
                          [12, 15], [15, 30], [15, 25], [25, 30], [30, 40]])
        spec_en = np.sum(abs_psd, axis=-1)
        p_rel = np.array([np.sum(abs_psd[:, (f > low) & (f <= high)], axis=-1) / spec_en for low, high in bands])
        p_rat = np.array([
            p_rel[1, :] / p_rel[0, :],   # Theta/Delta
            p_rel[1, :] / p_rel[7, :],   # Theta/Beta
            p_rel[2, :] / p_rel[0, :],   # Alpha/Delta
            p_rel[2, :] / p_rel[1, :],   # Alpha/Theta
            p_rel[2, :] / p_rel[7, :],   # Alpha/Beta
            p_rel[5, :] / p_rel[4, :]    # Alpha3/Alpha2
        ])
        return spec_en, p_rel, p_rat

    def rolloff(self, frequency, abs_psd, thresh=.85):
        cumulative_sum = np.cumsum(abs_psd, axis=-1)
        threshold = thresh * cumulative_sum[:, -1]
        ro_ind = np.argmax(cumulative_sum > threshold[:, np.newaxis], axis=-1)
        return frequency[ro_ind]
