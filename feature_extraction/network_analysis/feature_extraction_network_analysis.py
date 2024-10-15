import numpy as np
import networkx as nx
from utilities.adjacency_matrix import EEGAdjacencyMatrix


class NetworkFeatureExtractor:
    def __init__(self, epochs, ch_reg):
        """
        Initializes the feature extractor with the necessary parameters.

        Parameters:
        - epochs: mne.Epochs object (contains EEG/PSG data)
        - ch_reg: Dictionary of channel names divided by brain region
        """
        self.epochs = epochs
        self.ch_reg = ch_reg
        self.feats = sorted(['58 Degree Centrality', '59 Betweenness Centrality',
                             '60 Closeness Centrality', '61 Clustering Coefficient'])
        self.features_matrix = np.zeros([len(self.ch_reg), len(self.feats), len(self.epochs)])

    def extract_features(self):
        """
        Extracts network features from the EEG epochs for each brain region.

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
                x_sig_30 = epochs_chs.get_data()[t_epc, :, :]
                adjacency_matrix_extractor = EEGAdjacencyMatrix()
                ad_m_bin = adjacency_matrix_extractor.process(x_sig_30)
                graph_m = nx.from_numpy_array(ad_m_bin)

                # Extract network-based features
                self.features_matrix[nr, :, t_epc] = self.extract_network_features(graph_m)

        return self.features_matrix, self.feats

    def extract_network_features(self, graph_m):
        """
        Extracts network-related features from the graph generated for a given epoch.

        Parameters:
        - graph_m: Graph created from adjacency matrix of an epoch.

        Returns:
        - features_vector: A vector of network-related features.
        """
        features_vector = np.zeros(len(self.feats))

        # Degree Centrality
        features_vector[0] = np.mean(list(nx.degree_centrality(graph_m).values()))

        # Betweenness Centrality
        features_vector[1] = np.mean(list(nx.betweenness_centrality(graph_m).values()))

        # Closeness Centrality
        features_vector[2] = np.mean(list(nx.closeness_centrality(graph_m).values()))

        # Clustering Coefficient
        features_vector[3] = np.mean(list(nx.clustering(graph_m).values()))

        return features_vector
