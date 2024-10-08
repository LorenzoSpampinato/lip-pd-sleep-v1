import numpy as np

def rolloff(frequency, abs_psd, thresh=.85):
    # Function to compute spectral roll-off i.e., the frequency below which there is the 85% of the spectrum's energy

    cumulative_sum = np.cumsum(abs_psd, axis=-1)
    threshold = thresh * cumulative_sum[:, -1]
    ro_ind = np.argmax(cumulative_sum > threshold[:, np.newaxis], axis=-1)

    return frequency[ro_ind]