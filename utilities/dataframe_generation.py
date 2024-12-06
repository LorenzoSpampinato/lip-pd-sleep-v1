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
        Generates a single DataFrame containing data from all subjects and saves it as a CSV.

        Parameters:
        - run_aggregation: Flag to start the aggregation process.
        """
        if not run_aggregation:
            return

        # Collect all DataFrames for each subject
        feature_files = glob.glob(os.path.join(self.save_path, 'Features') + '/*/*/*.npz')
        all_dataframes = []

        for feat_file in feature_files:
            df = self._process_feature_file(feat_file)
            if df is not None:
                all_dataframes.append(df)

        # Concatenate all individual DataFrames into a single DataFrame
        final_dataframe = pd.concat(all_dataframes)
        final_csv_path = os.path.join(self.save_path, 'all_subjects_features.csv')
        final_dataframe.to_csv(final_csv_path, index=True, header=True, na_rep='NaN')
        print(f"DataFrame for all subjects saved at: {final_csv_path}")

    def _process_feature_file(self, feat_file):
        """
        Processes each EEG feature file and creates the corresponding DataFrame.

        Parameters:
        - feat_file: Path of the EEG feature file in `.npz` format.

        Returns:
        - DataFrame for the single subject.
        """
        print("start loading")
        features_data = np.load(feat_file)['data'].squeeze()
        print("end loading")

        # Aggregate features from different brain regions
        feature_matrix = np.concatenate(features_data, axis=1)
        print("end aggregation")

        # Extract groups, subjects, brain regions, and stages (if present)
        group = self._extract_group(feat_file)
        print('group', group)
        subject_id = self._extract_subject_id(feat_file)
        print('subject_id', subject_id)
        brain_regions = self._extract_brain_regions(feat_file)
        print('brain_regions', brain_regions)
        stages = self._extract_sleep_stages(feat_file) if self.aggregate_labels else ['/'] * feature_matrix.shape[1]
        print('stages', stages)

        # Create the MultiIndex for the DataFrame rows
        row_labels = pd.MultiIndex.from_product([group, subject_id, brain_regions, stages],
                                                names=['Group', 'Subject', 'Brain region', 'Stage'])
        print("end MultiIndex")

        # Extract feature names
        feature_names = self._extract_feature_names(feat_file)
        print("start Dataframe")

        # Create the DataFrame for this subject
        df = pd.DataFrame(feature_matrix.T, index=row_labels, columns=feature_names)
        print("end Dataframe")

        return df

    def _extract_group(self, feat_file):
        return [re.search(self.pattern_group, feat_file).group(0)]

    def _extract_subject_id(self, feat_file):
        return [re.search(self.pattern_subject, feat_file).group(0)]

    def _extract_brain_regions(self, feat_file):
        return np.load(feat_file)['regions'].tolist()

    def _extract_feature_names(self, feat_file):
        return np.load(feat_file)['feats'].tolist()

    def _extract_sleep_stages(self, feat_file):
        """
        Estrae le etichette del sonno per ciascun paziente, considerando l'ID del soggetto.
        Cerca i file delle etichette del sonno in cui il nome del file inizia con l'ID del soggetto.
        """
        # Estrai il gruppo e l'ID del soggetto dal percorso del file
        group_directory = self._extract_group(feat_file)[0]
        subject_id = self._extract_subject_id(feat_file)[0]

        # Cerca il file delle etichette del sonno specifico per il soggetto
        sleep_stage_files = glob.glob(
            os.path.join(self.label_path, group_directory, f'{subject_id}*_preprocessed.npy'))

        if not sleep_stage_files:
            print(
                f"Warning: No '_preprocessed.npy' file found for subject {subject_id} in group {group_directory}. Returning empty stages.")
            return []  # Restituisci un array vuoto o un valore di default

        # Carica le etichette del sonno dal primo file trovato
        sleep_stages = np.load(sleep_stage_files[0]).squeeze().astype(str).tolist()
        print(f"Loaded sleep stages for {subject_id}: {sleep_stages}")

        # Mappa i numeri a etichette comprensibili (W, N1, N2, N3, REM)
        return list(map(lambda x: x.replace('0', 'W').replace('1', 'N1').replace('2', 'N2')
                        .replace('3', 'N3').replace('4', 'REM'), sleep_stages))




