import os
import glob
import random
import pandas as pd
from data_processing.stratify_split import stratify_split

def train_val_test_split(save_path, run_dataset_split, training_percentage, validation_percentage, test_percentage):
    # save_path: absolute path where the results are stored
    # training_percentage: percentage of data for Training set
    # validation_percentage: percentage of data for Validation set
    # test_percentage: percentage of data for Test set
    # run_dataset_split: flag to run Training, Validation, and Test Set definition

    if not run_dataset_split: return

    feat_save_path = os.path.join(save_path, 'Features')
    ds_save_path = os.path.join(save_path, 'Sets')
    if not os.path.exists(ds_save_path): os.mkdir(ds_save_path)

    # Splitting subjects names into Training, Validation, and Test sets
    x_train, x_val, x_test = stratify_split(feat_save_path, training_percentage, validation_percentage, test_percentage)

    # To allow for reproducibility
    random.seed(2)

    # Shuffling elements
    random.shuffle(x_train)
    random.shuffle(x_val)
    random.shuffle(x_test)

    name = ['train_set', 'val_set', 'test_set']
    for n_ds, ds in enumerate([x_train, x_val, x_test]):
        # Cycle on subjects referrung to the same set
        for n, sub_id in enumerate(ds):
            # Loading features
            feats = pd.read_csv(glob.glob(os.path.join(feat_save_path, '*/*', sub_id + '*.csv'))[0], header=0)
            # Concatenating feature sets to form Training, Validation, and Test sets
            if n == 0: x_full = feats.copy()
            else: x_full = pd.concat([x_full, feats], axis=0, ignore_index=True)

        x_full.to_csv(os.path.join(ds_save_path, name[n_ds] + '.csv'), index=False, header=True)
        ########################################################################################################
        # Note: when loading the dataframe use:
        # --> "pd.read_csv(os.path.join(ds_save_path, name[n_ds] + '.csv'), header=0)"