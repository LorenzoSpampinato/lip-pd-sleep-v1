import numpy as np
import os
from scipy.io import loadmat
import glob


class HypnogramProcessor:
    def __init__(self, label_path, run_hypnogram):
        """
        Initializes the class to process hypnograms.
        :param label_path: Absolute path to the hypnograms (folders for different disease stages).
        :param run_hypnogram: Flag to execute the hypnogram definition.
        """
        self.label_path = label_path
        self.run_hypnogram = run_hypnogram
        # Mapping of sleep stages:
        # [-3, -2, -1, 0, 1] --> [Awake, N1, N2, N3, REM] i.e., [0, 1, 2, 3, 4]
        self.hypnogram_conversion = {0: 0, 1: 4, -1: 1, -2: 2, -3: 3}

    def process_hypnograms(self):
        """Main method to process all hypnogram files."""
        if not self.run_hypnogram:
            return
        print("Processing hypnograms...")
        # Loop through all .mat hypnogram files
        for stages_sub in glob.glob(self.label_path + '/*/*.mat', recursive=True):
            hyp = self._load_hypnogram(stages_sub)
            hyp_30s = self._define_30s_epochs(hyp)
            hyp_converted = self._convert_hypnogram(hyp_30s)
            self._save_hypnogram(stages_sub, hyp_converted)


    def _load_hypnogram(self, file_path):
        """Loads hypnogram data from a .mat file."""
        hyp_data = loadmat(file_path)['sesscoringinfoss'].squeeze()
        return hyp_data

    def _define_30s_epochs(self, hyp):
        """
        Defines the hypnogram based on 30-second epochs by aggregating 6-second epochs.
        :param hyp: Raw hypnogram data.
        :return: Hypnogram reduced to 30-second epochs.
        """
        hyp_file_1 = hyp[:, 2]  # Extracts sleep stages
        hyp_file_2 = hyp[:, 3]  # Indices for aggregating 6-second epochs into 30-second epochs

        # Initialize the hypnogram for 30-second epochs
        hyp_30s = np.zeros(hyp_file_2[-1])
        for ind_epoch_30 in np.unique(hyp_file_2):
            same_epoch = np.where(hyp_file_2 == ind_epoch_30)[0]
            assert np.all(hyp_file_1[same_epoch] == hyp_file_1[same_epoch][0]), \
                "Not all 6-second epochs show the same sleep stage"
            hyp_30s[ind_epoch_30 - 1] = hyp_file_1[same_epoch][0]

        # If the last 30-second epoch does not fully cover the 30 seconds, the hypnogram is shortened
        if len(np.where(hyp_file_2 == np.unique(hyp_file_2)[-1])[0]) != 5:
            hyp_30s = hyp_30s[:-1]

        return hyp_30s

    def _convert_hypnogram(self, hyp_30s):
        """
        Converts sleep stages based on the provided mapping.
        :param hyp_30s: Hypnogram with 30-second epochs.
        :return: Converted hypnogram.
        """
        return np.vectorize(self.hypnogram_conversion.get)(hyp_30s)

    def _save_hypnogram(self, file_path, hypnogram):
        """
        Saves the converted hypnogram as a .npy file.
        :param file_path: Original path of the .mat file.
        :param hypnogram: Converted hypnogram to save.
        """
        np.save(os.path.splitext(file_path)[0] + '.npy', hypnogram)
