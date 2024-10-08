import numpy as np
from scipy.signal import welch

def renyi_entropy_func(x, fs, nperseg, normalize=True):
    # Function to compute Renyi's entropy

    _, psd = welch(x, fs, nperseg=nperseg, axis=-1)
    psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
    renyi = -np.log(np.sum(psd_norm ** 2, axis=-1)) / np.log(2)

    if normalize: renyi /= np.log2(psd_norm.shape[-1])

    return renyi