import os
import glob
import random
import numpy as np
from operator import itemgetter


def stratify_split(data_path, training_percentage, validation_percentage, test_percentage):
    # data_path: absolute path to define disease stage groups and patients' names
    # training_percentage: percentage of data for Training set
    # validation_percentage: percentage of data for Validation set
    # test_percentage: percentage of data for Test set

    x_train = []
    x_val = []
    x_test = []

    # To allow for reproducibility
    random.seed(1)
    # Cycle on classes i.e., disease stage groups
    for c_path in glob.glob(data_path + '/*'):
        subs = os.listdir(c_path)

        n_subs = np.arange(0, len(subs), 1).tolist()
        rnd_train = random.sample(n_subs, int(np.round(training_percentage * len(subs))))
        if len(rnd_train) == 1: subs_train = [itemgetter(*rnd_train)(subs)]
        else: subs_train = list(itemgetter(*rnd_train)(subs))
        x_train.extend(subs_train)

        new_n_subs = list(set(n_subs) - set(rnd_train))
        rnd_val = random.sample(new_n_subs, int(np.round(validation_percentage * len(subs))))
        if len(rnd_val) == 1: subs_val = [itemgetter(*rnd_val)(subs)]
        else: subs_val = list(itemgetter(*rnd_val)(subs))
        x_val.extend(subs_val)

        rnd_test = list(set(new_n_subs) - set(rnd_val))
        if len(rnd_test) == 1: subs_test = [itemgetter(*rnd_test)(subs)]
        else: subs_test = list(itemgetter(*rnd_test)(subs))
        x_test.extend(subs_test)

    # If the Validation set is greater than the Test set, they are exchanged
    if len(x_val) > len(x_test):
        x_val, x_test = x_test, x_val

    return x_train, x_val, x_test