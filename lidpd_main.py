import os
import sys
import time
import warnings
from utilities import HypnogramProcessor
from feature_extraction import EEGFeatureExtractor
from utilities import EEGDataFrameGenerator
from utilities import NaNChecker
from utilities import DimensionalityReducer
from data_processing import DatasetSplitter
from model_training import Classifier
from input_script import get_args

warnings.filterwarnings('ignore')

# -----------------------------------------------  Dataset description  ------------------------------------------------
# - high-density EEG data from 257 channels
# - PNS (peripheral nervous system) data from 6 channels (i.e., Chin, Resp. Pressure, ECG, Resp. Effort,
#   Right and Left leg)


if __name__ == "__main__":
    args = get_args()  # Call get_args() from the imported file

    # Defining the folder where to save the results
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Save all prints in logfile
    sys.stdout = open(os.path.join(args.info_path, f'logfile_{int(time.time())}.txt'), 'w')

    # ------------------------------------  Hypnogram definition (30-s epochs)  ------------------------------------
    processor = HypnogramProcessor(args.label_path, args.run_hypnogram)
    processor.process_hypnograms()

    # -----------------------------------  Pre-processing + feature extraction  ------------------------------------
    extractor = EEGFeatureExtractor(args.data_path, args.label_path, args.save_path, args.run_preprocess, args.run_bad_interpolation)
    extractor.extract_features_for_all_subjects(args.only_class, args.only_patient, args.run_features)

    # --------------------------------  Aggregating labels + Dataframe definition  ---------------------------------
    generator = EEGDataFrameGenerator(args.label_path, args.save_path, args.aggregate_labels)
    generator.generate_dataframe(args.run_aggregation)

    # -------------------------------------------  Checking NaN values  --------------------------------------------
    checker = NaNChecker(args.save_path, args.run_nan_check)
    checker.check_nan_values()

    # ------------------------------  Training, Validation, and Test sets definition  ------------------------------
    splitter = DatasetSplitter(args.save_path, args.training_percentage,
                               args.validation_percentage, args.test_percentage)
    splitter.train_val_test_split(args.run_dataset_split)

    # ----------------------------------------  Dimensionality reduction  ------------------------------------------
    reducer = DimensionalityReducer(args.save_path, args.run_pca, args.explained_variance,
                                    args.only_stage, args.only_brain_region)
    reducer.run()

    # -------------------------------------------  Classification task  --------------------------------------------
    classifier = Classifier(args.save_path, args.only_stage, args.only_brain_region)
    classifier.classification(args.run_classification_task, args.run_pca)

