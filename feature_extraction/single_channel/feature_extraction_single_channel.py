import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.stats import mstats
import antropy as ant
from feature_extraction.single_channel.spectral_analysis_functions.renyi_entropy import renyi_entropy_func
from feature_extraction.single_channel.spectral_analysis_functions.roll_off import rolloff
from feature_extraction.single_channel.spectral_analysis_functions.spectral import spectral


def feature_extraction_single_channel(epochs, fs, ch_reg, win_sec=5):
    # epochs: instance of mne.Epochs
    # fs: sampling frequency, [Hz]
    # ch_reg: set with channel names divided per brain region
    # win_sec: window length, [s]
    #          Note: win_sec should be at least double of the inverse of the lowest frequency of interest
    #                having f_min = 0.5 Hz, win_sec is set to 2*(1/0.5) = 4 s (5 s to be conservative)

    # Ordered list of all the extracted features
    feats = {'1 Spectral energy', '2 Relative delta power band', '3 Relative theta power band',
             '4 Relative alpha power band', '5 Relative alpha1 power band', '6 Relative alpha2 power band',
             '7 Relative alpha3 power band', '8 Relative sigma power band', '9 Relative beta power band',
             '10 Relative beta1 power band', '11 Relative beta2 power band', '12 Relative gamma power band',
             '13 theta-delta power ratio', '14 theta-beta power ratio', '15 alpha-delta power ratio',
             '16 alpha-theta power ratio', '17 alpha-beta power ratio', '18 alpha3-alpha2 power ratio',
             '19 Spectral mean', '20 Spectral variance', '21 Spectral skewness', '22 Spectral kurtosis',
             '23 Spectral centroid', '24 Spectral crest factor', '25 Spectral flatness', '26 Spectral rolloff',
             '27 Spectral spread', '28 Mean', '29 Variance', '30 Skewness', '31 Kurtosis', '32 Zero-crossings',
             '33 Hjorth mobility', '34 Hjorth complexity', '35 Spectral entropy', '36 Renyi entropy',
             '37 Approximate entropy', '38 Sample entropy', '39 Singular value decomposition entropy',
             '40 Permutation entropy', '41 De-trended fluctuation analysis exponent', '42 Lempel–Ziv complexity',
             '43 Katz fractal dimension', '44 Higuchi fractal dimension', '45 Petrosian fractal dimension'}

    features_matrix = np.zeros([len(ch_reg), len(feats), len(epochs)])
    # Cycle on brain region
    for nr, r in enumerate(ch_reg):
        epochs_chs = epochs.copy()
        chs = [ch.strip() for ch in r.split('=')[1].split(',')]
        epochs_chs.pick(picks=chs, verbose=False)
        features_matrix_reg = np.zeros([len(chs), len(feats), len(epochs)])
        # Cycle on channels
        for nch, ch in enumerate(chs):
            # Isolating each epoch i.e., 30-second samples of signals
            x_sig_30 = epochs_chs.get_data()[:, nch, :]

            # PSD using Welch method
            f, psd = welch(x=(x_sig_30 - np.mean(x_sig_30, axis=-1)[:, np.newaxis]), fs=fs, window='hamming',
                           nperseg=int(win_sec * fs), average='median', axis=-1)
            # Absolute value of PSD
            abs_psd = np.abs(psd)
            # Mean value of PSD
            mean_psd = np.mean(abs_psd, axis=-1)

            ########################################################################################
            # ----------------------------  Frequency-domain features  -----------------------------
            ########################################################################################

            # Spectral energy, Relative spectral powers, and Spectral power ratios
            spec_en, p_rel, p_rat = spectral(f, abs_psd)
            features_matrix_reg[nch, 0, :] = spec_en
            features_matrix_reg[nch, 1:12, :] = p_rel
            features_matrix_reg[nch, 12:18, :] = p_rat

            # Spectral mean
            features_matrix_reg[nch, 18, :] = mean_psd
            # Spectral variance
            features_matrix_reg[nch, 19, :] = np.var(abs_psd, axis=-1)
            # Spectral skewness
            features_matrix_reg[nch, 20, :] = skew(abs_psd, axis=-1)
            # Spectral kurtosis
            features_matrix_reg[nch, 21, :] = kurtosis(abs_psd, axis=-1)

            # Spectral centroid
            features_matrix_reg[nch, 22, :] = np.sum(f * abs_psd, axis=-1) / spec_en
            # Spectral crest factor
            features_matrix_reg[nch, 23, :] = np.max(abs_psd, axis=-1) / mean_psd
            # Spectral flatness
            features_matrix_reg[nch, 24, :] = mstats.gmean(abs_psd, axis=-1) / mean_psd
            # Spectral roll-off
            features_matrix_reg[nch, 25, :] = rolloff(f, abs_psd)
            # Spectral spread
            features_matrix_reg[nch, 26, :] = np.sum(((f - features_matrix_reg[nch, 22, :, np.newaxis]) ** 2)
                                                     * abs_psd, axis=-1) / spec_en

            ########################################################################################
            # -------------------------------  Time-domain features  -------------------------------
            ########################################################################################

            # Mean
            features_matrix_reg[nch, 27, :] = np.mean(x_sig_30, axis=-1)
            # Variance
            features_matrix_reg[nch, 28, :] = np.var(x_sig_30, axis=-1)
            # Skewness
            features_matrix_reg[nch, 29, :] = skew(x_sig_30, axis=-1)
            # Kurtosis
            features_matrix_reg[nch, 30, :] = kurtosis(x_sig_30, axis=-1)
            # Number of zero-crossings
            features_matrix_reg[nch, 31, :] = ant.num_zerocross(x_sig_30, axis=-1)
            # Hjorth parameters i.e., mobility, complexity
            features_matrix_reg[nch, 32, :], features_matrix_reg[nch, 33, :] = ant.hjorth_params(x_sig_30, axis=-1)

            # Spectral/Shannon entropy
            features_matrix_reg[nch, 34, :] = ant.spectral_entropy(x_sig_30, sf=fs, method='welch',
                                                                   nperseg=int(win_sec * fs), normalize=True, axis=-1)
            # Renyi's entropy
            features_matrix_reg[nch, 35, :] = renyi_entropy_func(x_sig_30, fs=fs, nperseg=int(win_sec * fs))
            # Approximate entropy
            features_matrix_reg[nch, 36, :] = [ant.app_entropy(x) for x in x_sig_30]
            # Sample entropy
            features_matrix_reg[nch, 37, :] = [ant.sample_entropy(x) for x in x_sig_30]
            # Singular Value Decomposition entropy
            features_matrix_reg[nch, 38, :] = [ant.svd_entropy(x, normalize=True) for x in x_sig_30]
            # Permutation entropy
            features_matrix_reg[nch, 39, :] = [ant.perm_entropy(x, normalize=True) for x in x_sig_30]

            # De-trended fluctuation analysis exponent
            features_matrix_reg[nch, 40, :] = [ant.detrended_fluctuation(x) for x in x_sig_30]

            # Lempel-Ziv complexity coefficient
            features_matrix_reg[nch, 41, :] = [ant.lziv_complexity(''.join(map(str, np.where(x > np.median(x), 1, 0))),
                                                                   normalize=True) for x in x_sig_30]

            # Fractal dimensions i.e., Katz, Higuchi, Petrosian FDs
            features_matrix_reg[nch, 42, :] = ant.katz_fd(x_sig_30, axis=-1)
            features_matrix_reg[nch, 43, :] = [ant.higuchi_fd(x.astype(np.float64)) for x in x_sig_30]
            features_matrix_reg[nch, 44, :] = ant.petrosian_fd(x_sig_30, axis=-1)

        features_matrix[nr, :, :] = np.mean(features_matrix_reg, axis=0)

    return features_matrix, sorted(feats)