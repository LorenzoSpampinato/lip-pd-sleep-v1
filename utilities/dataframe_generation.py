import os
import glob
import re
import numpy as np
import pandas as pd


class EEGDataFrameGenerator:
    def __init__(self, label_path, save_path, aggregate_labels, class_name, sub_fold):
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
        self.only_class = class_name
        self.only_patient = sub_fold
        annotations_path = os.path.join(label_path, 'clinical_scorings')
        self.pattern_group = r'(' + '|'.join(os.listdir(annotations_path)) + ')'
        print('pattern_group', self.pattern_group)
        # Pattern to extract the subject ID (e.g., PD001)
        self.pattern_subject = r'(PD\d{3})'
        print('pattern_subject', self.pattern_subject)

    def generate_dataframe(self, run_aggregation=True):
        """
        Generates a single DataFrame containing data from a specific subject and saves it as a CSV.

        Parameters:
        - run_aggregation: Flag to start the aggregation process.
        """
        if not run_aggregation:
            return

        # Define the path to the specific feature file (this is hardcoded here based on the patient and class)
        # The path includes the subject's patient ID, class (e.g., 'sleep stages'), and feature file name.
        feature_file_path = os.path.join(
            self.save_path, 'Features', self.only_class, self.only_patient,
            f'{self.only_patient}_no_mean_N2N3FILTERS_specific_channels_150.npz'
        )
        print(f"Processing feature file: {feature_file_path}")

        # Process the single feature file
        df = self._process_feature_file_MEAN(feature_file_path)

        if df is not None:
            # If the DataFrame is processed successfully, we define the path where the resulting CSV file will be saved.

            # Create the 'feature_csv' directory if it doesn't exist
            feature_csv_dir = os.path.join(self.save_path, 'feature_csv')
            os.makedirs(feature_csv_dir, exist_ok=True)

            # Define the path where the final CSV file will be saved, including the subject's patient ID.
            final_csv_path = os.path.join(feature_csv_dir, f'{self.only_patient}_no_mean_N2N3FILTERS_specific_channels_150.csv')

            # Save the DataFrame directly to a CSV file
            df.to_csv(final_csv_path, index=True, header=True, na_rep='NaN')
            print(f"DataFrame for the subject saved at: {final_csv_path}")
        else:
            print(f"Failed to process the file: {feature_file_path}")
        '''
        # Collect all DataFrames for each subject
        #feature_files = glob.glob(os.path.join(self.save_path, 'Features') + '/*/*/*.npz')
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
        '''

    def _process_feature_file_MEAN(self, feat_file):
        """
        Processes each EEG feature file and creates the corresponding DataFrame.

        Parameters:
        - feat_file: Path of the EEG feature file in `.npz` format.

        Returns:
        - DataFrame for the single subject.
        """
        print("Start loading features data")
        npz_data = np.load(feat_file)

        features_data = npz_data['data'].squeeze()  # Shape: (n_regions, n_features, n_samples)
        print(f"Shape of features_data: {features_data.shape}")

        n_regions, n_features, n_samples = features_data.shape
        print(f"Number of regions: {n_regions}, features: {n_features}, samples: {n_samples}")

        for i, region_data in enumerate(features_data):
            print(f"Region {i}: shape {region_data.shape}")

        # **Reshape delle feature**
        reshaped_features = [region_data.reshape(-1, region_data.shape[-1]) for region_data in features_data]
        feature_matrix = np.concatenate(reshaped_features, axis=-1)
        print(f"Feature matrix shape: {feature_matrix.shape}")

        # **Estrarre metadati**
        group = self._extract_group(feat_file)
        subject_id = self._extract_subject_id(feat_file)
        brain_regions = self._extract_brain_regions(feat_file)
        indices = self._extract_selected_indices(feat_file)
        epoch_labels = self._extract_epoch_labels(feat_file)  # **Nuovo metadato**

        print(f"Group: {group}")
        print(f"Subject ID: {subject_id}")
        print(f"Brain regions: {brain_regions}")
        print(f"Selected indices (epochs): {indices}")
        print(f"Epoch labels (sleep stages): {epoch_labels}")

        # **Determinare il numero di ripetizioni per epoca**
        if 'channels' in npz_data:
            channels = self._extract_channels(feat_file)
            print(f"Channels: {channels}")
            num_repeats = len(channels)  # Numero di ripetizioni per ogni epoca
        else:
            num_repeats = len(brain_regions)  # Numero di ripetizioni per ogni epoca

        # **Ripetere gli indici delle epoche e le etichette**
        expanded_indices = np.repeat(indices, num_repeats)
        expanded_epoch_labels = np.repeat(epoch_labels, num_repeats)

        print(f"Expanded epoch indices shape: {expanded_indices.shape}")
        print(f"Expanded epoch labels shape: {expanded_epoch_labels.shape}")

        # **Costruire MultiIndex con la logica ORIGINALE**
        if 'channels' in npz_data:
            row_labels = pd.MultiIndex.from_product(
                [group, subject_id, indices, channels],
                names=['Group', 'Subject', 'Epochs', 'Channel']
            )
        else:
            row_labels = pd.MultiIndex.from_product(
                [group, subject_id, indices, brain_regions],
                names=['Group', 'Subject', 'Epochs', 'Brain region']
            )

        # **Estrarre nomi delle feature**
        feature_names = self._extract_feature_names(feat_file)
        print(f"Feature names: {feature_names}")

        # **Creare DataFrame**
        print("Creating DataFrame")
        # Creare DataFrame
        print("Creating DataFrame")
        # Aggiungere epoch_labels come colonna separata, prima delle features
        df = pd.DataFrame(feature_matrix.T, index=row_labels, columns=feature_names)

        # Aggiungere la colonna "Epoch Stage" prima delle features
        df["Stage"] = expanded_epoch_labels

        # Riordinare le colonne in modo che "Epoch Stage" sia prima delle features
        cols = ["Stage"] + [col for col in df.columns if col != "Epoch Stage"]
        df = df[cols]

        print(f"DataFrame created with shape: {df.shape}")

        return df

    def _extract_group(self, feat_file):
        result = [re.search(self.pattern_group, feat_file).group(0)]
        print(f"Extracted group: {result}")
        return result

    def _extract_subject_id(self, feat_file):
        result = [re.search(self.pattern_subject, feat_file).group(0)]
        print(f"Extracted subject ID: {result}")
        return result

    def _extract_feature_names(self, feat_file):
        result = np.load(feat_file)['feats'].tolist()
        print(f"Extracted feature names: {result}")
        return result

    def _extract_brain_regions(self, feat_file):
        """Extracts the brain regions from the .npz file, if available."""
        try:
            return np.load(feat_file)['regions'].tolist()
        except KeyError:
            print(f"KeyError: 'regions' not found in {feat_file}. Returning an empty list.")
            return []  # Return an empty list if 'regions' is missing

    def _extract_channels(self, feat_file):
        """Extracts the channels (electrodes) from the .npz file, if available."""
        try:
            return np.load(feat_file)['channels'].tolist()
        except KeyError:
            print(f"KeyError: 'channels' not found in {feat_file}. Returning an empty list.")
            return []  # Return an empty list if 'channels' is missing

    def _extract_selected_indices(self, feat_file):
        """Extracts the selected indices (epochs) from the .npz file."""
        result = np.load(feat_file)['selected_indices'].tolist()
        print(f"Extracted selected indices: {result}")
        return result

    def _extract_epoch_labels(self, feat_file):
        """Extracts the sleep stage labels (N2/N3) from the .npz file."""
        labels = np.load(feat_file)['epoch_labels'].tolist()
        print(f"Extracted epoch labels: {labels}")
        return labels


    '''
    
    def _process_feature_file_ORIGINAL(self, feat_file):
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


    def _extract_sleep_stages_perchannel_non_va(self, feat_file):
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

    def _process_feature_file(self, feat_file):
        """
        Processes each EEG feature file and creates the corresponding DataFrame.

        Parameters:
        - feat_file: Path of the EEG feature file in `.npz` format.

        Returns:
        - DataFrame for the single subject.
        """
        print("start loading")
        npz_data = np.load(feat_file)
        features_data = npz_data['data'].squeeze()  # EEG features (shape: epochs, regions, electrodes, features)
        print("end loading")
        print("Available keys in npz file:", np.load(feat_file).keys())
        print("Shape of features_data:", features_data.shape)
        print("Processing file:", feat_file)

        # Extract groups, subjects, brain regions, and electrodes
        group = self._extract_group(feat_file)
        subject_id = self._extract_subject_id(feat_file)
        print('subject_id', subject_id)
        brain_regions = self._extract_brain_regions(feat_file)
        channels = self._extract_channels(feat_file)  # Electrodes (channels)
        feature_names = self._extract_feature_names(feat_file)
        indices = npz_data['selected_indices']  # Epoch indices

        row_labels = pd.MultiIndex.from_product([group, subject_id, indices, brain_regions, channels],
                                                names=['Group', 'Subject', 'Epoch', 'Brain region', 'Channel'])
        df = pd.DataFrame(features_data.T, index=row_labels, columns=feature_names)
        print("end Dataframe")

        return df
'''





