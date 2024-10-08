import numpy as np

def spectral(f, abs_psd):
    bands = np.array([[0.5, 4], [4, 8], [8, 12], [8, 9], [9, 11], [11, 12],
                      [12, 15], [15, 30], [15, 25], [25, 30], [30, 40]])

    spec_en = np.sum(abs_psd, axis=-1)
    p_rel = np.array([np.sum(abs_psd[:, (f > low) & (f <= high)], axis=-1) / spec_en for low, high in bands])

    p_rat = np.array([p_rel[1, :] / p_rel[0, :], p_rel[1, :] / p_rel[7, :], p_rel[2, :] / p_rel[0, :],
                      p_rel[2, :] / p_rel[1, :], p_rel[2, :] / p_rel[7, :], p_rel[5, :] / p_rel[4, :]])

    return spec_en, p_rel, p_rat
