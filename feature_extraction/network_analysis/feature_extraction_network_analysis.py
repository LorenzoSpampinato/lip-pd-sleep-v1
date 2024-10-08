import numpy as np
import networkx as nx
import mne
from utilities.adjacency_matrix import adjacency_matrix

def feature_extraction_network_analysis(epochs, ch_reg):
    # epochs: instance of mne.Epochs
    # ch_reg: set with channel names divided per brain region

    feats = {'58 Degree Centrality', '59 Betweenness Centrality',
             '60 Closeness Centrality', '61 Clustering Coefficient'}

    features_matrix = np.zeros([len(ch_reg), len(feats), len(epochs)])
    # Cycle on brain region
    for nr, r in enumerate(ch_reg):
        epochs_chs = epochs.copy()
        chs = [ch.strip() for ch in r.split('=')[1].split(',')]
        epochs_chs.pick(picks=chs, verbose=False)
        # Cycle on epochs
        for t_epc in range(len(epochs_chs)):
            # Isolating each epoch i.e., 30-second samples of signals
            x_sig_30 = epochs_chs.get_data()[t_epc, :, :]
            ad_m_bin = adjacency_matrix(x_sig_30)
            graph_m = nx.from_numpy_array(ad_m_bin)

            # Degree Centrality
            features_matrix[nr, 0, t_epc] = np.mean(np.array(list(nx.degree_centrality(graph_m).values())))
            # Betweenness Centrality
            features_matrix[nr, 1, t_epc] = np.mean(np.array(list(nx.betweenness_centrality(graph_m).values())))
            # Closeness Centrality
            features_matrix[nr, 2, t_epc] = np.mean(np.array(list(nx.closeness_centrality(graph_m).values())))
            # Clustering Coefficient
            features_matrix[nr, 3, t_epc] = np.mean(np.array(list(nx.clustering(graph_m).values())))

    return features_matrix, sorted(feats)