import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_processing.filt_ds import DataFrameFilter


class DimensionalityReducer:
    def __init__(self, save_path, run_pca, explained_variance, only_stage=None, only_brain_region=None):
        """
        Initializes the dimensionality reducer for EEG datasets.

        Parameters:
        - save_path: Absolute path where results are saved.
        - run_pca: Flag to run dimensionality reduction using PCA.
        - explained_variance: Percentage of variance to explain (used for the number of PCA components).
        - only_stage: The sleep stage to consider for the analysis, e.g., W, N1, N2, N3, REM (optional).
        - only_brain_region: The brain region to consider for the analysis, e.g., Fp, F, C, T, P, O (optional).
        """
        self.save_path = save_path
        self.run_pca = run_pca
        self.explained_variance = explained_variance
        self.only_stage = only_stage
        self.only_brain_region = only_brain_region

    def run(self):
        """Executes the dimensionality reduction process if the `run_pca` flag is True."""
        if not self.run_pca:
            return

        train_df, val_df, test_df = self._load_data()
        train_filt, val_filt, test_filt = DataFrameFilter(train_df, val_df, test_df).filter_dfs(
            only_stage=self.only_stage, only_brain_region=self.only_brain_region)
        self._apply_pca_and_save(train_filt, val_filt, test_filt)

    def _load_data(self):
        """Loads the training, validation, and test data from CSV files."""
        ds_save_path = os.path.join(self.save_path, 'Sets')
        train_df = pd.read_csv(os.path.join(ds_save_path, 'train_set.csv'), header=0)
        val_df = pd.read_csv(os.path.join(ds_save_path, 'val_set.csv'), header=0)
        test_df = pd.read_csv(os.path.join(ds_save_path, 'test_set.csv'), header=0)
        return train_df, val_df, test_df

    def _apply_pca_and_save(self, train_filt, val_filt, test_filt):
        """Applies PCA to the filtered data and saves the results."""
        train_info, train_feats = self._split_info_and_feats(train_filt)
        val_info, val_feats = self._split_info_and_feats(val_filt)
        test_info, test_feats = self._split_info_and_feats(test_filt)

        train_norm, val_norm, test_norm = self._normalize_features(train_feats, val_feats, test_feats)

        train_reduced, val_reduced, test_reduced = self._perform_pca(train_norm, val_norm, test_norm)

        self._save_reduced_data(train_info, train_reduced, val_info, val_reduced, test_info, test_reduced)

    def _split_info_and_feats(self, df):
        """Splits the DataFrame into information (first 4 columns) and features."""
        info = df.iloc[:, :4]
        feats = df.iloc[:, 4:].values
        return info, feats

    def _normalize_features(self, train_feats, val_feats, test_feats):
        """Normalizes the features using Z-score standardization."""
        z_score = StandardScaler()
        z_score.fit(train_feats)
        train_norm = z_score.transform(train_feats)
        val_norm = z_score.transform(val_feats)
        test_norm = z_score.transform(test_feats)
        return train_norm, val_norm, test_norm

    def _perform_pca(self, train_norm, val_norm, test_norm):
        """Performs PCA on the normalized data and returns the reduced sets."""
        pca = PCA(n_components=self.explained_variance, svd_solver='full', random_state=1)
        pca.fit(train_norm)
        train_reduced = pca.transform(train_norm)
        val_reduced = pca.transform(val_norm)
        test_reduced = pca.transform(test_norm)
        return train_reduced, val_reduced, test_reduced

    def _save_reduced_data(self, train_info, train_reduced, val_info, val_reduced, test_info, test_reduced):
        """Saves the reduced datasets (train, val, test) along with their information."""
        ss, br = self._get_file_suffixes()
        ds_save_path = os.path.join(self.save_path, 'Sets')

        pca_labels = ['PCA component ' + str(n + 1) for n in range(np.size(train_reduced, 1))]

        train_reduced_full = pd.concat([train_info, pd.DataFrame(train_reduced, columns=pca_labels)], axis=1)
        val_reduced_full = pd.concat([val_info, pd.DataFrame(val_reduced, columns=pca_labels)], axis=1)
        test_reduced_full = pd.concat([test_info, pd.DataFrame(test_reduced, columns=pca_labels)], axis=1)

        train_reduced_full.to_csv(os.path.join(ds_save_path, 'reduced_train_set' + ss + br + '.csv'),
                                  index=False, header=True)
        val_reduced_full.to_csv(os.path.join(ds_save_path, 'reduced_val_set' + ss + br + '.csv'),
                                index=False, header=True)
        test_reduced_full.to_csv(os.path.join(ds_save_path, 'reduced_test_set' + ss + br + '.csv'),
                                 index=False, header=True)

    def _get_file_suffixes(self):
        """Determines the suffixes to append to the file names based on selected stages and brain regions."""
        ss = '_' + self.only_stage if self.only_stage else ''
        br = '_' + self.only_brain_region if self.only_brain_region else ''
        return ss, br

