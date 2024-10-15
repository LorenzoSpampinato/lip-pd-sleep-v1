import os
import glob
import random
import pandas as pd
from data_processing.stratify_split import StratifiedSplitter

class DatasetSplitter:
    def __init__(self, save_path: str, training_percentage: float, validation_percentage: float, test_percentage: float):
        """
        Initializes the DatasetSplitter with the provided parameters.

        Parameters:
        - save_path: absolute path where the results are stored
        - training_percentage: percentage of data for the Training set
        - validation_percentage: percentage of data for the Validation set
        - test_percentage: percentage of data for the Test set
        """
        self.save_path = save_path
        self.training_percentage = training_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.feat_save_path = os.path.join(save_path, 'Features')
        self.ds_save_path = os.path.join(save_path, 'Sets')

        # Create the directory for sets if it doesn't exist
        if not os.path.exists(self.ds_save_path):
            os.makedirs(self.ds_save_path)

    def train_val_test_split(self, run_dataset_split: bool):
        """
        Splits the dataset into training, validation, and test sets.

        Parameters:
        - run_dataset_split: flag to run Training, Validation, and Test Set definition
        """
        if not run_dataset_split:
            return

        # Create an instance of StratifiedSplitter
        splitter = StratifiedSplitter(self.feat_save_path, self.training_percentage, self.validation_percentage, self.test_percentage)

        # Splitting subjects names into Training, Validation, and Test sets
        x_train, x_val, x_test = splitter.stratify_split()

        # To allow for reproducibility
        random.seed(2)

        # Shuffling elements
        random.shuffle(x_train)
        random.shuffle(x_val)
        random.shuffle(x_test)

        name = ['train_set', 'val_set', 'test_set']
        for n_ds, ds in enumerate([x_train, x_val, x_test]):
            # Cycle on subjects referring to the same set
            x_full = pd.DataFrame()  # Initialize empty DataFrame for concatenation
            for sub_id in ds:
                # Loading features
                feat_files = glob.glob(os.path.join(self.feat_save_path, '*/*', sub_id + '*.csv'))
                if feat_files:  # Check if there are any files matching the pattern
                    feats = pd.read_csv(feat_files[0], header=0)
                    # Concatenating feature sets to form Training, Validation, and Test sets
                    x_full = pd.concat([x_full, feats], axis=0, ignore_index=True)

            # Save the concatenated DataFrame to a CSV file
            x_full.to_csv(os.path.join(self.ds_save_path, f"{name[n_ds]}.csv"), index=False, header=True)

# Example usage
# dataset_splitter = DatasetSplitter(save_path='path/to/save', training_percentage=0.7, validation_percentage=0.2, test_percentage=0.1)
# dataset_splitter.train_val_test_split(run_dataset_split=True)
