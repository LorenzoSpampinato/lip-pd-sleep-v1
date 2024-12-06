import os
import glob
import re
import numpy as np
import pandas as pd
import random


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

        # Pattern to extract the group PD or other categories from the filename
        self.pattern_group = r'(ADV|CTL|DNV|DYS)'  # Modify this pattern as per your requirements
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
        if all_dataframes:
            final_dataframe = pd.concat(all_dataframes)
            final_csv_path = os.path.join(self.save_path, 'all_subjects_features.csv')
            final_dataframe.to_csv(final_csv_path, index=True, header=True, na_rep='NaN')
            print(f"DataFrame for all subjects saved at: {final_csv_path}")

            # Preview the first few rows of the final dataframe
            print("Preview of the final dataframe:")
            print(final_dataframe.head())
        else:
            print("No dataframes were generated. Please check your input files.")

    def _process_feature_file(self, feat_file):
        print(f"Loading file: {feat_file}")
        features_data = np.load(feat_file)['data'].squeeze()
        print(f"Loaded data shape: {features_data.shape}")

        # Verifica delle dimensioni
        num_regions, num_features, num_epochs = features_data.shape

        # Aggregazione delle caratteristiche per epoca (concatenando le regioni)
        feature_matrix = features_data.reshape(num_epochs * num_regions,
                                               num_features)  # (epoche * regioni, caratteristiche)
        print(f"Feature matrix shape after reshaping: {feature_matrix.shape}")

        # Estrazione del gruppo, soggetto e regioni cerebrali
        group = self._extract_group(feat_file)
        subject_id = self._extract_subject_id(feat_file)
        brain_regions = self._extract_brain_regions(feat_file)

        # Creazione dell'indice MultiIndex per le righe
        row_labels = []
        for epoch in range(num_epochs):
            for region in brain_regions:
                row_labels.append((group[0], subject_id[0], region))

        # Creazione dell'indice MultiIndex
        row_labels = pd.MultiIndex.from_tuples(row_labels[:num_epochs * num_regions],
                                               names=['Group', 'Subject', 'Brain region'])
        print("MultiIndex for DataFrame created.")

        # Estrazione dei nomi delle feature
        feature_names = self._extract_feature_names(feat_file)
        print("Feature names extracted.")

        # Estrazione degli stadi del sonno
        sleep_stages = self._extract_sleep_stages(feat_file)  # Aggiungi gli stadi
        print("Sleep stages extracted.")

        # Aggiungere una colonna per gli stadi del sonno
        sleep_stage_column = []

        for epoch in range(num_epochs):
            # Gli stadi del sonno devono essere assegnati a ogni regione per ogni epoca
            for region in brain_regions:
                sleep_stage_column.append(sleep_stages[epoch])  # Aggiungi lo stadio dell'epoca per ogni regione

        # Creazione del DataFrame per questo soggetto
        df = pd.DataFrame(feature_matrix, index=row_labels, columns=feature_names)
        print("DataFrame for the subject created.")

        # Aggiungi la colonna degli stadi del sonno
        df['Sleep Stage'] = sleep_stage_column  # Aggiungi la colonna con gli stadi

        # Riordina le colonne per posizionare 'Sleep Stage' prima delle feature
        df = df[['Sleep Stage'] + [col for col in df.columns if col != 'Sleep Stage']]

        # Riordinare il DataFrame per regioni cerebrali
        df = df.sort_index(level='Brain region', sort_remaining=True)
        print("DataFrame sorted by brain regions.")

        return df

    def _extract_group(self, feat_file):
        """
        Extract the group information from the filename.
        """
        match = re.search(self.pattern_group, feat_file)
        if match:
            return [match.group(0)]
        else:
            print(f"Group not found in the file: {feat_file}. Assigning 'Unknown'.")
            return ["Unknown"]  # Default group if not found

    def _extract_subject_id(self, feat_file):
        """
        Extract the subject ID from the filename.
        """
        return [re.search(self.pattern_subject, feat_file).group(0)]

    def _extract_brain_regions(self, feat_file):
        """
        Extract the brain regions from the feature file.
        """
        return np.load(feat_file)['regions'].tolist()

    def _extract_feature_names(self, feat_file):
        """
        Extract feature names from the feature file.
        """
        return np.load(feat_file)['features'].tolist()

    import numpy as np
    import random

    def _extract_sleep_stages(self, feat_file):
        """
        Extracts the sleep stages from the corresponding feature file, but randomizes them.
        The stages are consistent for each epoch across brain regions.

        Parameters:
        - feat_file: Path to the EEG feature file in `.npz` format.

        Returns:
        - List of sleep stages, randomized per epoch but consistent within each epoch.
        """
        # Definizione degli stadi del sonno
        sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']

        # Determina il numero di epoche dal file delle caratteristiche
        num_epochs = np.load(feat_file)['data'].shape[2]  # Assumiamo che le epoche siano la terza dimensione dei dati

        # Genera gli stadi del sonno in modo casuale, ma uno per epoca
        sleep_stages = [random.choice(sleep_stages) for _ in range(num_epochs)]

        # Ogni epoca ha un solo stadio, ma deve essere ripetuto per ogni regione cerebrale
        return sleep_stages


# Percorso dei file delle feature EEG
save_path = r"D:\TESI\lid-data-samples\lid-data-samples\npz files"

# Crea il generatore del DataFrame EEG
generator = EEGDataFrameGenerator(label_path=None, save_path=save_path, aggregate_labels=True)

# Genera il DataFrame, aggrega i dati e salva il file CSV
generator.generate_dataframe(run_aggregation=True)
