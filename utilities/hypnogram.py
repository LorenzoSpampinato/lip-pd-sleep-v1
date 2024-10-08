import numpy as np
import os
from scipy.io import loadmat
import glob

def hypnogram_definition(label_path, run_hypnogram):
    # label_path: absolute path to the hypnograms i.e., folders for the different disease stages
    # run_hypnogram: flag to run hypnogram definition

    if not run_hypnogram: return

    # Sleep stages: [-3, -2, -1, 0, 1] --> conversion into [0, 1, 2, 3, 4], which stand for
    # [Awake, N1, N2, N3, REM], respectively
    hypnogram_conversion = {0: 0, 1: 4, -1: 1, -2: 2, -3: 3}

    # Cycle on hypnogram files
    for stages_sub in glob.glob(label_path + '/*/*.mat', recursive=True):
        hyp = loadmat(stages_sub)['sesscoringinfoss'].squeeze()
        # Extracting sleep stages
        hyp_file_1 = hyp[:, 2]
        # Extracting indexes for aggregating 6-s epochs into 30-s ones
        hyp_file_2 = hyp[:, 3]

        # Defining the hypnogram based on 30-s epochs
        hyp = np.zeros(hyp_file_2[-1])
        for ind_epoch_30 in np.unique(hyp_file_2):
            same_epoch = np.where(hyp_file_2 == ind_epoch_30)[0]
            assert np.all(hyp_file_1[same_epoch] == hyp_file_1[same_epoch][0]), \
                "Not all 6-s epochs show the same sleep stage"
            hyp[ind_epoch_30 - 1] = hyp_file_1[same_epoch][0]

        # If the last 30-s epoch is related to less than 30-s signal, then the hypnogram gets trimmed
        if len(np.where(hyp_file_2 == np.unique(hyp_file_2)[-1])[0]) != 5: hyp = hyp[:-1]

        hyp = np.vectorize(hypnogram_conversion.get)(hyp)
        np.save(os.path.splitext(stages_sub)[0] + '.npy', hyp)