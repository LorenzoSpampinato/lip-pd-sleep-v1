import os
import sys
import time
import warnings
from data_loading import EEGDataLoader
from preprocess import EEGPreprocessor
from utilities import HypnogramProcessor
from feature_extraction import EEGFeatureExtractor111
from utilities import EEGDataFrameGenerator
from utilities import NaNChecker
from utilities import DimensionalityReducer
from data_processing import DatasetSplitter
from model_training import Classifier
from input_script import get_args

warnings.filterwarnings('ignore')
args = get_args()  # Call get_args() from the imported file
# -----------------------------------------------  Dataset description  ------------------------------------------------
# - high-density EEG data from 257 channels
# - PNS (peripheral nervous system) data from 6 channels (i.e., Chin, Resp. Pressure, ECG, Resp. Effort,
#   Right and Left leg)


if __name__ == "__main__":

    # Defining the folder where to save the results
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Save all prints in logfile
    log_dir = os.path.join(args.info_path, 'lorenzo_logfiles')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'logfile_{int(time.time())}.txt')
    sys.stdout = open(log_file_path, 'w')
    print("Logfile created")

    # ------------------------------------  Hypnogram definition (30-s epochs)  ------------------------------------
    processor = HypnogramProcessor(args.label_path, args.run_hypnogram)
    processor.process_hypnograms()
    print("Hypnograms processed")

    # Extract subject folders based on class and patient filters
    if args.only_class and args.only_patient:
        sub_folds = [os.path.join(args.data_path, args.only_class, args.only_patient)]
    else:
        sub_folds = [folder for folder in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, folder))]

    # Process Multiple Subjects
    for sub_fold in sub_folds:
        '''
        try:
            print(f"Start processing: {sub_fold}")

            # Define full path for the subject folder
            subject_path = os.path.join(args.data_path, sub_fold)

            # Check if the folder already ends with '.mff' and define mff_path accordingly
            mff_path = subject_path if subject_path.endswith('.mff') else subject_path + '.mff'

            # Rename to .mff if necessary
            if not subject_path.endswith('.mff') and os.path.exists(subject_path):
                print(f"Renaming {subject_path} to {mff_path}...")
                os.rename(subject_path, mff_path)

            # --------------------------------------- Load the raw data --------------------------------------------------
            data_loader = EEGDataLoader(args.data_path, args.label_path)
            raw = data_loader.load_and_prepare_data(mff_path)  # Load using mff_path

            # --------------------------------------- Preprocessing -----------------------------------------------------
            if args.run_preprocess or args.run_bad_interpolation:
                preprocessor = EEGPreprocessor(
                    raw, args.data_path, args.save_path, args.run_preprocess, args.run_bad_interpolation
                )
                processed_raw = preprocessor.preprocess()
            else:
                processed_raw = raw  # Use raw data if no preprocessing is specified

            # ------------------------- Segment into epochs and extract features -----------------------------------------
            if args.run_features:
                feature_extractor = EEGFeatureExtractor111(args.data_path, args.save_path, args.regions)
                feature_extractor.process_and_save_features(processed_raw, mff_path)

            print(f"Finished processing: {sub_fold}")

        except Exception as e:
            print(f"Error encountered during processing of {sub_fold}: {e}")

        finally:
            # Restore original folder name (remove '.mff') if it was renamed
            if os.path.exists(mff_path):
                print(f"Renaming {mff_path} back to {sub_fold}...")
                os.rename(mff_path, subject_path)
                print(f"End renaming back: {mff_path} to {sub_fold}")

    print("Data processing completed.")
    '''
    print(f"Start processing: {sub_fold}")

    # Define full path for the subject folder
    subject_path = os.path.join(args.data_path, sub_fold)

    # Check if the folder already ends with '.mff' and define mff_path accordingly
    mff_path = subject_path if subject_path.endswith('.mff') else subject_path + '.mff'

    # Rename to .mff if necessary
    if not subject_path.endswith('.mff') and os.path.exists(subject_path):
        print(f"Renaming {subject_path} to {mff_path}...")
        os.rename(subject_path, mff_path)

        # --------------------------------------- Load the raw data --------------------------------------------------
        data_loader = EEGDataLoader(args.data_path, args.label_path)
        raw = data_loader.load_and_prepare_data(mff_path)  # Load using mff_path

        # --------------------------------------- Preprocessing -----------------------------------------------------
        if args.run_preprocess or args.run_bad_interpolation:
            preprocessor = EEGPreprocessor(
                raw, args.data_path, args.save_path, args.run_preprocess, args.run_bad_interpolation
            )
            processed_raw = preprocessor.preprocess()
        else:
            processed_raw = raw  # Use raw data if no preprocessing is specified

        # ------------------------- Segment into epochs and extract features -----------------------------------------
        if args.run_features:
            feature_extractor = EEGFeatureExtractor111(args.data_path, args.save_path)
            feature_extractor.process_and_save_features(processed_raw, mff_path)

        print(f"Finished processing: {sub_fold}")

        # Restore original folder name (remove '.mff') if it was renamed
        if os.path.exists(mff_path):
            print(f"Renaming {mff_path} back to {sub_fold}...")
            os.rename(mff_path, subject_path)
            print(f"End renaming back: {mff_path} to {sub_fold}")



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