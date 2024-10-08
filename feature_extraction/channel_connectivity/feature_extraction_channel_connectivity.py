import numpy as np
from mne_connectivity import spectral_connectivity_time
from utilities import adjacency_matrix


def feature_extraction_channel_connectivity(epochs, fs, ch_reg):
    # epochs: instance of mne.Epochs
    # fs: sampling frequency, [Hz]
    # ch_reg: set with channel names divided per brain region

    feats = {'46 Phase Locking Value in Delta', '47 Weighted Phase Lag Index in Delta',
             '48 Phase Locking Value in Theta', '49 Weighted Phase Lag Index in Theta',
             '50 Phase Locking Value in Alpha', '51 Weighted Phase Lag Index in Alpha',
             '52 Phase Locking Value in Sigma', '53 Weighted Phase Lag Index in Sigma',
             '54 Phase Locking Value in Beta', '55 Weighted Phase Lag Index in Beta',
             '56 Phase Locking Value in Gamma', '57 Weighted Phase Lag Index in Gamma'}

    # Order of frequency bands: Delta, Theta, Alpha, Sigma, Beta, Gamma
    bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 15], [15, 30], [30, 40]])

    features_matrix = np.zeros([len(ch_reg), len(feats), len(epochs)])
    # Cycle on brain region
    for nr, r in enumerate(ch_reg):
        epochs_chs = epochs.copy()
        chs = [ch.strip() for ch in r.split('=')[1].split(',')]
        epochs_chs.pick(picks=chs, verbose=False)
        # Cycle on epochs
        for t_epc in range(len(epochs_chs)):
            # Isolating each epoch i.e., 30-second samples of signals
            x_sig_30 = epochs_chs.get_data()[t_epc, :, :][np.newaxis, :, :]

            # Isolating each frequency band
            for nb, (fl, fu) in enumerate(bands):

                # 1. Phase Locking Value
                plv = spectral_connectivity_time(x_sig_30, method='plv', sfreq=fs, freqs=[fl, fu],
                                                 faverage=True, n_cycles=3, verbose=False).get_data()[0, :, :]
                plv_m = plv.reshape(np.size(x_sig_30, 1), np.size(x_sig_30, 1))
                plv_m_no_rep = np.tril(plv_m, k=-1)
                features_matrix[nr, int(nb * 2), t_epc] = np.mean(plv_m_no_rep[plv_m_no_rep != 0].flatten())

                # 2. Weighted Phase Lag Index
                wpli = spectral_connectivity_time(x_sig_30, method='wpli', sfreq=fs, freqs=[fl, fu],
                                                  faverage=True, n_cycles=3, verbose=False).get_data()[0, :, :]
                wpli_m = wpli.reshape(np.size(x_sig_30, 1), np.size(x_sig_30, 1))
                wpli_m_no_rep = np.tril(wpli_m, k=-1)
                features_matrix[nr, int(nb * 2) + 1, t_epc] = np.mean(wpli_m_no_rep[wpli_m_no_rep != 0].flatten())

    return features_matrix, sorted(feats)