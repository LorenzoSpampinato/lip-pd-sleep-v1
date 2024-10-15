import os
import glob
import random
import numpy as np
from operator import itemgetter


class StratifiedSplitter:
    def __init__(self, data_path: str, training_percentage: float, validation_percentage: float,
                 test_percentage: float):
        """
        Initializes the StratifiedSplitter with the provided parameters.

        Parameters:
        - data_path: absolute path to define disease stage groups and patients' names
        - training_percentage: percentage of data for the Training set
        - validation_percentage: percentage of data for the Validation set
        - test_percentage: percentage of data for the Test set
        """
        self.data_path = data_path
        self.training_percentage = training_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage

        # Check if the percentages sum to 1
        if not np.isclose(training_percentage + validation_percentage + test_percentage, 1.0):
            raise ValueError("Training, validation, and test percentages must sum to 1.")

    def stratify_split(self):
        """
        Performs stratified splitting of data into training, validation, and test sets.

        Returns:
        - A tuple of lists (x_train, x_val, x_test).
        """
        x_train = []
        x_val = []
        x_test = []

        # To allow for reproducibility
        random.seed(1)

        # Cycle through classes (disease stage groups)
        for c_path in glob.glob(os.path.join(self.data_path, '*')):
            subs = os.listdir(c_path)

            n_subs = np.arange(len(subs)).tolist()
            rnd_train = random.sample(n_subs, int(np.round(self.training_percentage * len(subs))))
            subs_train = list(itemgetter(*rnd_train)(subs))
            x_train.extend(subs_train)

            new_n_subs = list(set(n_subs) - set(rnd_train))
            rnd_val = random.sample(new_n_subs, int(np.round(self.validation_percentage * len(subs))))
            subs_val = list(itemgetter(*rnd_val)(subs))
            x_val.extend(subs_val)

            rnd_test = list(set(new_n_subs) - set(rnd_val))
            subs_test = list(itemgetter(*rnd_test)(subs))
            x_test.extend(subs_test)

        # If the Validation set is greater than the Test set, they are exchanged
        if len(x_val) > len(x_test):
            x_val, x_test = x_test, x_val

        return x_train, x_val, x_test

# Example usage
# splitter = StratifiedSplitter(data_path='path/to/data', training_percentage=0.7, validation_percentage=0.2, test_percentage=0.1)
# train_set, val_set, test_set = splitter.stratify_split()
