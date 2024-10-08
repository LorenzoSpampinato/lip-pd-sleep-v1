import os
import sys
import time
import warnings
from argparse import ArgumentParser
from utilities import hypnogram_definition
from feature_extraction import features_extraction
from utilities import dataframe_generation
from utilities import nan_check
from utilities import dimensionality_reduction
from data_processing import train_val_test_split
from model_training import classification

#from lidpd_utils import (hypnogram_definition, features_extraction, dataframe_generation, nan_check,
#                         train_val_test_split, dimensionality_reduction, classification)
warnings.filterwarnings('ignore')


# -----------------------------------------------  Dataset description  ------------------------------------------------
# - high-density EEG data from 257 channels
# - PNS (peripheral nervous system) data from 6 channels (i.e., Chin, Resp. Pressure, ECG, Resp. Effort,
#   Right and Left leg)


def get_args():
    parser = ArgumentParser(description='Standardize Parkinson data')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data_path", type=str,
                        help='Absolute path to the folder containing PSG data',
                        default=r'D:\TESI\lid-data-samples\lid-data-samples\Dataset')
    parser.add_argument("--label_path", type=str,
                        help='Absolute path to the folder containing the labels',
                        default=r'D:\TESI\lid-data-samples\lid-data-samples/Labels')
    parser.add_argument("--save_path", type=str,
                        help='Absolute path where to save the pre-processed PSG data',
                        default=r'D:\TESI\lid-data-samples\lid-data-samples/Results_prova')
    parser.add_argument("--info_path", type=str,
                        help='Absolute path where to save the logfiles',
                        default=r'D:\TESI\lid-data-samples\lid-data-samples/Results_prova')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_hypnogram", type=bool,
                        help='Flag to run hypnogram definition',
                        default=True)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_features", type=bool,
                        help='Flag to run feature extraction',
                        default=True)
    parser.add_argument("--run_preprocess", type=bool,
                        help='Flag to run pre-processing',
                        default=True)
    parser.add_argument("--run_bad_interpolation", type=bool,
                        help='Flag to run bad channels interpolation',
                        default=False)
    parser.add_argument("--only_class", type=str,
                        help='Disease stage only for which features are extracted (i.e., DNV, ADV, DYS, CTL)',
                        default='DYS')
    parser.add_argument("--only_patient", type=str,
                        help='Patient name only for which features are extracted (i.e., PD002, PD003, ...)',
                        default='PD012')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_aggregation", type=bool,
                        help='Flag to run data aggregation for dataframe definition (for each subject separately)',
                        default=True)
    parser.add_argument("--aggregate_labels", type=bool,
                        help='Flag to whether consider labels during dataframe definition',
                        default=False)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_nan_check", type=bool,
                        help='Flag to run check on NaN values among extracted features',
                        default=True)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_dataset_split", type=bool,
                        help='Flag to run Training, Validation, and Test Set definition',
                        default=True)
    parser.add_argument("--training_percentage", type=float,
                        help='Percentage of the Dataset for Training set',
                        default=.6)
    parser.add_argument("--validation_percentage", type=float,
                        help='Percentage of the Dataset for Validation set',
                        default=.2)
    parser.add_argument("--test_percentage", type=float,
                        help='Percentage of the Dataset for Test set',
                        default=.2)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--only_stage", type=str,
                        help='Sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM. If None, no '
                             'distinction is made',
                        default=None)
    parser.add_argument("--only_brain_region", type=str,
                        help='Brain region to be considered for the analysis i.e., Fp, F, C, T, P, O. If None, no '
                             'distinction is made',
                        default=None)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_pca", type=bool,
                        help='Flag to run dimensionality reduction (PCA)',
                        default=True)
    parser.add_argument("--explained_variance", type=float,
                        help='Percentage of data variance to explain (number of PCA components taken consequently)',
                        default=.95)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_classification_task", type=bool,
                        help='Flag to run classification task',
                        default=True)

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    # Defining the folder where to save the results
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Save all prints in logfile
    sys.stdout = open(os.path.join(args.info_path, f'logfile_{int(time.time())}.txt'), 'w')

    # ------------------------------------  Hypnogram definition (30-s epochs)  ------------------------------------
    hypnogram_definition(args.label_path, args.run_hypnogram)

    # -----------------------------------  Pre-processing + feature extraction  ------------------------------------
    features_extraction(args.data_path, args.label_path, args.save_path, args.run_features,
                        args.run_preprocess, args.run_bad_interpolation, args.only_class, args.only_patient)

    # --------------------------------  Aggregating labels + Dataframe definition  ---------------------------------
    dataframe_generation(args.label_path, args.save_path, args.run_aggregation, args.aggregate_labels)

    # -------------------------------------------  Checking NaN values  --------------------------------------------
    nan_check(args.save_path, args.run_nan_check)

    # ------------------------------  Training, Validation, and Test sets definition  ------------------------------
    train_val_test_split(args.save_path, args.run_dataset_split, args.training_percentage,
                         args.validation_percentage, args.test_percentage)

    # ----------------------------------------  Dimensionality reduction  ------------------------------------------
    dimensionality_reduction(args.save_path, args.run_pca, args.explained_variance,
                             args.only_stage, args.only_brain_region)

    # -------------------------------------------  Classification task  --------------------------------------------
    classification(args.save_path, args.run_class_task, args.run_pca, args.only_stage, args.only_brain_region)
