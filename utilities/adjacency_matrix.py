import numpy as np

def adjacency_matrix(x_sig_30, p_th=0.8):
    # Function to compute the dimensional adjacency matrix, based on Pearson correlation between signals at different
    # pairs of EEG channels
    # x_sig_30: one epoch of HD-EEG data with dimensions [N_channels X N_samples]
    # p_th: threshold to compute the adjacent matrix

    x_sig_30_mean = x_sig_30 - np.mean(x_sig_30, axis=1, keepdims=True)
    n = np.dot(x_sig_30_mean, x_sig_30_mean.T)
    d = np.sqrt(np.outer(np.sum(x_sig_30_mean ** 2, axis=1), np.sum(x_sig_30_mean ** 2, axis=1)))
    ad_m = n / d

    ad_m_bin = np.where(ad_m < p_th, 0, 1)
    np.fill_diagonal(ad_m_bin, 0)

    return ad_m_bin