import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_processing.filt_ds import filt_ds


def dimensionality_reduction(save_path, run_pca, explained_variance, only_stage, only_brain_region):
    # save_path: absolute path where the results are stored
    # run_pca: flag to run dimensionality reduction
    # explained_variance: percentage of data variance to explain (number of PCA components taken consequently)
    # only_stage: sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM
    # only_brain_region: brain region to be considered for the analysis i.e., Fp, F, C, T, P, O

    if not run_pca: return

    # Loading data
    # 1st column (idx = 0): Patient Group; 2nd column (idx = 1): Subject ID; 3rd column (idx = 2): Brain region;
    # 4th column (idx = 3): Sleep stage --> Actual features values are between the 5th and 65th columns (idx = 4 - 64)
    ds_save_path = os.path.join(save_path, 'Sets')
    train_df = pd.read_csv(os.path.join(ds_save_path, 'train_set.csv'), header=0)
    val_df = pd.read_csv(os.path.join(ds_save_path, 'val_set.csv'), header=0)
    test_df = pd.read_csv(os.path.join(ds_save_path, 'test_set.csv'), header=0)

    # Flag for saving the filename
    if only_stage and only_brain_region: ss, br = ['_' + only_stage, '_' + only_brain_region]
    elif only_stage and not only_brain_region: ss, br = ['_' + only_stage, '']
    elif not only_stage and only_brain_region: ss, br = ['', '_' + only_brain_region]
    else: ss, br = ['', '']

    # Filtering per sleep stage and/or per brain region if needed
    train_filt, val_filt, test_filt = filt_ds(train_df, val_df, test_df, only_stage=only_stage,
                                              only_brain_region=only_brain_region)

    # Extracting db info
    train_info = train_filt.iloc[:, :4]
    val_info = val_filt.iloc[:, :4]
    test_info = test_filt.iloc[:, :4]
    # Extracting db features
    train_feats = train_filt.iloc[:, 4:].values
    val_feats = val_filt.iloc[:, 4:].values
    test_feats = test_filt.iloc[:, 4:].values

    # Z-score normalization (standardization):
    z_score = StandardScaler()
    z_score.fit(train_feats)
    train_norm = z_score.transform(train_feats)
    val_norm = z_score.transform(val_feats)
    test_norm = z_score.transform(test_feats)

    # Principal Component Analysis - PCA:
    pca = PCA(n_components=explained_variance, svd_solver='full', random_state=1)
    pca.fit(train_norm)
    train_reduced = pca.transform(train_norm)
    val_reduced = pca.transform(val_norm)
    test_reduced = pca.transform(test_norm)

    # Reattaching the information of the first 4 columns i.e., Patient Group, Subject ID, Brain region, and Sleep stage
    # and defining new labels for columns of reduced sets
    pca_labels = ['PCA component ' + str(n + 1) for n in range(np.size(train_reduced, 1))]

    train_reduced_full_info = pd.concat([train_info, pd.DataFrame(train_reduced, columns=pca_labels)], axis=1)
    val_reduced_full_info = pd.concat([val_info, pd.DataFrame(val_reduced, columns=pca_labels)], axis=1)
    test_reduced_full = pd.concat([test_info, pd.DataFrame(test_reduced, columns=pca_labels)], axis=1)

    train_reduced_full_info.to_csv(os.path.join(ds_save_path, 'reduced_train_set' + ss + br + '.csv'),
                                   index=False, header=True)
    val_reduced_full_info.to_csv(os.path.join(ds_save_path, 'reduced_val_set' + ss + br + '.csv'),
                                 index=False, header=True)
    test_reduced_full.to_csv(os.path.join(ds_save_path, 'reduced_test_set' + ss + br + '.csv'),
                             index=False, header=True)
    ########################################################################################################
    # Note: when loading the dataframe use:
    # --> "pd.read_csv(os.path.join(ds_save_path, 'reduced_train_set' + ss + br + '.csv'), header=0)"