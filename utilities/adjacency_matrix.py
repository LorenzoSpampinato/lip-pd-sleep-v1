import numpy as np

class EEGAdjacencyMatrix:
    def __init__(self, p_th=0.8):
        """
        Initializes the class to compute the adjacency matrix.

        Parameters:
        - p_th: threshold for binarizing the adjacency matrix (default=0.8)
        """
        self.p_th = p_th

    def compute_adjacency_matrix(self, x_sig_30):
        """
        Computes the adjacency matrix based on Pearson correlation between EEG signals.

        Parameters:
        - x_sig_30: One epoch of HD-EEG data with dimensions [N_channels X N_samples]

        Returns:
        - ad_m: non-binarized adjacency matrix
        """
        # Remove the mean from the signal
        x_sig_30_mean = x_sig_30 - np.mean(x_sig_30, axis=1, keepdims=True)

        # Calculate Pearson correlation between channels
        numerator = np.dot(x_sig_30_mean, x_sig_30_mean.T)
        denominator = np.sqrt(np.outer(np.sum(x_sig_30_mean ** 2, axis=1), np.sum(x_sig_30_mean ** 2, axis=1)))

        # Adjacency matrix
        ad_m = numerator / denominator

        return ad_m

    def binarize_adjacency_matrix(self, ad_m):
        """
        Binarizes the adjacency matrix based on a threshold.

        Parameters:
        - ad_m: adjacency matrix

        Returns:
        - ad_m_bin: binarized adjacency matrix
        """
        # Binarize the matrix using the defined threshold (self.p_th)
        ad_m_bin = np.where(ad_m < self.p_th, 0, 1)

        # Set the diagonal to 0 to avoid self-loops
        np.fill_diagonal(ad_m_bin, 0)

        return ad_m_bin

    def process(self, x_sig_30):
        """
        Computes and binarizes the adjacency matrix for a single epoch of EEG data.

        Parameters:
        - x_sig_30: One epoch of HD-EEG data with dimensions [N_channels X N_samples]

        Returns:
        - ad_m_bin: binarized adjacency matrix
        """
        # Compute the adjacency matrix
        ad_m = self.compute_adjacency_matrix(x_sig_30)

        # Binarize the adjacency matrix
        ad_m_bin = self.binarize_adjacency_matrix(ad_m)

        return ad_m_bin

