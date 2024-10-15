import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

class NaNChecker:
    def __init__(self, save_path, run_nan_check):
        """
        Initializes the class to check for NaN values in feature DataFrames.
        :param save_path: Absolute path where the results are saved.
        :param run_nan_check: Flag to execute the NaN check.
        """
        self.save_path = save_path
        self.run_nan_check = run_nan_check
        # Pattern to extract the subject ID
        self.pattern = r'(PD\d{3})'

    def check_nan_values(self):
        """Main method to check for NaN values in feature DataFrames."""
        if not self.run_nan_check:
            return

        # Loop through all .csv feature files
        for feat_m in glob.glob(os.path.join(self.save_path, 'Features') + '/*/*/*.csv'):
            self._analyze_nan(feat_m)

    def _analyze_nan(self, file_path):
        """Analyzes NaN values in a DataFrame and creates a visualization matrix."""
        # Load the feature DataFrame
        fm2d = pd.read_csv(file_path, header=0).iloc[:, 4:]
        regs = pd.read_csv(file_path, header=0).iloc[:, 2].values
        n_epochs = len(np.where(regs == np.unique(regs)[0])[0])

        # Check if there are NaN values
        if np.isnan(fm2d.values).any():
            self._plot_nan_matrix(file_path, fm2d, n_epochs)

    def _plot_nan_matrix(self, file_path, fm2d, n_epochs):
        """
        Creates a plot of the NaN matrix and saves it as an image.
        :param file_path: Original path of the feature file.
        :param fm2d: Feature DataFrame.
        :param n_epochs: Number of epochs in the DataFrame.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(re.search(self.pattern, file_path).group(0), fontsize=20)
        msno.matrix(fm2d, ax=ax)
        fig.tight_layout()

        # Add lines to separate epochs
        for i in range((len(fm2d) // n_epochs) + 1):
            ax.axhline(i * n_epochs, color='red', linestyle='--', linewidth=2)

        # Save the figure as an image
        fig.savefig(os.path.splitext(file_path)[0] + '.jpg', dpi=300)
        plt.close()

