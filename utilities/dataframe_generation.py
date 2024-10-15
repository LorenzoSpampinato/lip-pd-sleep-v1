import os
import glob
import re
import numpy as np
import pandas as pd

class EEGDataFrameGenerator:
    def __init__(self, label_path, save_path, aggregate_labels):

        """
        Initializes the DataFrame generator for EEG data.

        Parameters:
        - label_path: Absolute path to the hypnograms (labels of sleep stages).
        - save_path: Absolute path where the EEG feature results are saved.
        - aggregate_labels: Flag to include or exclude sleep stages in the DataFrame.
        """
        self.label_path = label_path
        self.save_path = save_path
        self.aggregate_labels = aggregate_labels

        # Pattern to extract the group PD or other categories
        self.pattern_group = r'(' + '|'.join(os.listdir(label_path)) + ')'
        # Pattern to extract the subject ID (e.g., PD001)
        self.pattern_subject = r'(PD\d{3})'

    def generate_dataframe(self, run_aggregation=True):
        """
        Generates DataFrames for each subject based on EEG feature files and saves them as CSV.

        Parameters:
        - run_aggregation: Flag to start the aggregation process.
        """
        if not run_aggregation:
            return

        feature_files = glob.glob(os.path.join(self.save_path, 'Features') + '/*/*/*.npz')

        for feat_file in feature_files:
            self._process_feature_file(feat_file)

    def _process_feature_file(self, feat_file):
        """
        Processes each EEG feature file and creates the corresponding DataFrame.

        Parameters:
        - feat_file: Path of the EEG feature file in `.npz` format.
        """
        # Load the EEG feature data
        features_data = np.load(feat_file)['data'].squeeze()

        # Aggregate features from different brain regions
        feature_matrix = np.concatenate(features_data, axis=1)

        # Extract groups, subjects, brain regions, and stages (if present)
        group = self._extract_group(feat_file)
        subject_id = self._extract_subject_id(feat_file)
        brain_regions = self._extract_brain_regions(feat_file)

        # If sleep stages should be included
        stages = self._extract_sleep_stages(feat_file) if self.aggregate_labels else ['/'] * np.size(features_data, 2)

        # Create the MultiIndex for the DataFrame rows
        row_labels = pd.MultiIndex.from_product([group, subject_id, brain_regions, stages],
                                                names=['Group', 'Subject', 'Brain region', 'Stage'])

        # Extract feature names
        feature_names = self._extract_feature_names(feat_file)

        # Create the DataFrame
        df = pd.DataFrame(feature_matrix.T, index=row_labels, columns=feature_names)

        # Save the DataFrame as CSV
        self._save_dataframe(feat_file, df)

    def _extract_group(self, feat_file):
        """
        Extracts the group from the filename (e.g., DNV, ADV, CTL).

        Parameters:
        - feat_file: Path of the EEG feature file.

        Returns:
        - List containing the group.
        """
        return [re.search(self.pattern_group, feat_file).group(0)]

    def _extract_subject_id(self, feat_file):
        """
        Extracts the subject ID from the filename (e.g., PD001).

        Parameters:
        - feat_file: Path of the EEG feature file.

        Returns:
        - List containing the subject ID.
        """
        return [re.search(self.pattern_subject, feat_file).group(0)]

    def _extract_brain_regions(self, feat_file):
        """
        Extracts the names of brain regions from the feature file.

        Parameters:
        - feat_file: Path of the EEG feature file.

        Returns:
        - List of brain regions.
        """
        return np.load(feat_file)['regions'].tolist()

    def _extract_feature_names(self, feat_file):
        """
        Extracts the feature names from the feature file.

        Parameters:
        - feat_file: Path of the EEG feature file.

        Returns:
        - List of feature names.
        """
        return np.load(feat_file)['feats'].tolist()

    def _extract_sleep_stages(self, feat_file):
        """
        Extracts sleep stage labels from the corresponding file.

        Parameters:
        - feat_file: Path of the EEG feature file.

        Returns:
        - List of sleep stages.
        """
        directory, _ = os.path.split(feat_file.replace(os.path.join(self.save_path, 'Features'), self.label_path))
        sleep_stages = np.load(directory + '.npy').squeeze().astype(str).tolist()

        # Map numeric stages to sleep stage names
        return list(map(lambda x: x.replace('0', 'W').replace('1', 'N1').replace('2', 'N2')
                        .replace('3', 'N3').replace('4', 'REM'), sleep_stages))

    def _save_dataframe(self, feat_file, dataframe):
        """
        Saves the DataFrame as a CSV file.

        Parameters:
        - feat_file: Path of the EEG feature file.
        - dataframe: DataFrame to be saved.
        """
        csv_file_path = os.path.splitext(feat_file)[0] + '.csv'
        dataframe.to_csv(csv_file_path, index=True, header=True, na_rep='NaN')


