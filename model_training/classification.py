import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from model_training.training_tuning_evaluation import training_tuning_evaluation
from data_processing.filt_ds import filt_ds

def classification(save_path, run_class_task, run_pca, only_stage, only_brain_region):
    # save_path: absolute path where the results are stored
    # run_class_task: flag to run classification task
    # run_pca: flag to run dimensionality reduction
    # only_stage: sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM
    # only_brain_region: brain region to be considered for the analysis i.e., Fp, F, C, T, P, O

    if not run_class_task: return

    ds_save_path = os.path.join(save_path, 'Sets')

    # Flag for loading/saving files
    if only_stage and only_brain_region: ss, br = ['_' + only_stage, '_' + only_brain_region]
    elif only_stage and not only_brain_region: ss, br = ['_' + only_stage, '']
    elif not only_stage and only_brain_region: ss, br = ['', '_' + only_brain_region]
    else: ss, br = ['', '']

    # For loading data
    # 1st column (idx = 0): Patient Group; 2nd column (idx = 1): Subject ID; 3rd column (idx = 2): Brain region;
    # 4th column (idx = 3): Sleep stage
    # --> Actual features values come after the 5th column

    # If dimensionality reduction has not been applied
    if not run_pca:
        train_df = pd.read_csv(os.path.join(ds_save_path, 'train_set.csv'), header=0)
        val_df = pd.read_csv(os.path.join(ds_save_path, 'val_set.csv'), header=0)
        test_df = pd.read_csv(os.path.join(ds_save_path, 'test_set.csv'), header=0)

        # Filtering per sleep stage and/or per brain region if needed
        train_filt, val_filt, test_filt = filt_ds(train_df, val_df, test_df, only_stage=only_stage,
                                                  only_brain_region=only_brain_region)

    # If dimensionality reduction has been applied, loading reduced sets
    else:
        train_filt = pd.read_csv(os.path.join(ds_save_path, 'reduced_train_set' + ss + br + '.csv'), header=0)
        val_filt = pd.read_csv(os.path.join(ds_save_path, 'reduced_val_set' + ss + br + '.csv'), header=0)
        test_filt = pd.read_csv(os.path.join(ds_save_path, 'reduced_test_set' + ss + br + '.csv'), header=0)

    # For each set, separating features values from labels for classification task
    x_train = train_filt.iloc[:, 4:].values
    y_train = train_filt.iloc[:, 0].values
    x_val = val_filt.iloc[:, 4:].values
    y_val = val_filt.iloc[:, 0].values
    x_test = test_filt.iloc[:, 4:].values
    y_test = test_filt.iloc[:, 0].values

    # --------------------------------------------  Random forest (RF)  --------------------------------------------
    base_rf = RandomForestClassifier(random_state=2)
    param_grid_rf = {'n_estimators': [50, 100, 150],
                     'max_features': ['sqrt', 'log2'],
                     'max_depth': [3, 6, 9]}
    training_tuning_evaluation(base_rf, param_grid_rf, x_train, y_train, x_val, y_val, x_test, y_test, save_path)

    # ---------------------------------------  Support Vector Machine (SVM)  ---------------------------------------
    base_svm = SVC(random_state=3)
    param_grid_svm = {'gamma': [0.1, 0.01, 0.001],
                      'kernel': ['linear', 'rbf'],
                      'C': [.1, 1, 10]}
    training_tuning_evaluation(base_svm, param_grid_svm, x_train, y_train, x_val, y_val, x_test, y_test, save_path)

    # ---------------------------------------  Multi-Layer Perceptron (MLP)  ---------------------------------------
    base_mlp = MLPClassifier(random_state=4)
    param_grid_mlp = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs', 'sgd', 'adam'],
                      'hidden_layer_sizes': [100, 200, 300, 400, 500]}
    training_tuning_evaluation(base_mlp, param_grid_mlp, x_train, y_train, x_val, y_val, x_test, y_test, save_path)

