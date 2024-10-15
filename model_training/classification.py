import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from model_training.training_tuning_evaluation import ModelEvaluator
from data_processing.filt_ds import DataFrameFilter

class Classifier:
    def __init__(self, save_path: str, only_stage: str = None, only_brain_region: str = None):
        """
        Initializes the Classifier with the provided parameters.

        Parameters:
        - save_path: absolute path where the results are stored
        - only_stage: sleep stage to be considered for the analysis (e.g., W, N1, N2, N3, REM)
        - only_brain_region: brain region to be considered for the analysis (e.g., Fp, F, C, T, P, O)
        """
        self.save_path = save_path
        self.only_stage = only_stage
        self.only_brain_region = only_brain_region

    def classification(self, run_class_task: bool, run_pca: bool):
        """
        Runs the classification task.

        Parameters:
        - run_class_task: flag to run classification task
        - run_pca: flag to run dimensionality reduction
        """
        if not run_class_task:
            return

        ds_save_path = os.path.join(self.save_path, 'Sets')

        # Define suffixes based on filtering criteria
        ss, br = self._get_suffixes()

        # Load data
        if not run_pca:
            train_df = pd.read_csv(os.path.join(ds_save_path, 'train_set.csv'), header=0)
            val_df = pd.read_csv(os.path.join(ds_save_path, 'val_set.csv'), header=0)
            test_df = pd.read_csv(os.path.join(ds_save_path, 'test_set.csv'), header=0)

            # Filter the data
            filter = DataFrameFilter(train_df, val_df, test_df)
            train_filt, val_filt, test_filt = filter.filter_dfs(only_stage=self.only_stage,
                                                      only_brain_region=self.only_brain_region)
        else:
            train_filt = pd.read_csv(os.path.join(ds_save_path, f'reduced_train_set{ss}{br}.csv'), header=0)
            val_filt = pd.read_csv(os.path.join(ds_save_path, f'reduced_val_set{ss}{br}.csv'), header=0)
            test_filt = pd.read_csv(os.path.join(ds_save_path, f'reduced_test_set{ss}{br}.csv'), header=0)

        # Separate features and labels
        x_train, y_train = train_filt.iloc[:, 4:].values, train_filt.iloc[:, 0].values
        x_val, y_val = val_filt.iloc[:, 4:].values, val_filt.iloc[:, 0].values
        x_test, y_test = test_filt.iloc[:, 4:].values, test_filt.iloc[:, 0].values

        # Train and evaluate Random Forest
        self._train_evaluate_rf(x_train, y_train, x_val, y_val, x_test, y_test)

        # Train and evaluate Support Vector Machine
        self._train_evaluate_svm(x_train, y_train, x_val, y_val, x_test, y_test)

        # Train and evaluate Multi-Layer Perceptron
        self._train_evaluate_mlp(x_train, y_train, x_val, y_val, x_test, y_test)

    def _get_suffixes(self):
        """
        Determine the suffixes based on filtering criteria.

        Returns:
        - Tuple containing suffixes for stage and brain region
        """
        if self.only_stage and self.only_brain_region:
            return f'_{self.only_stage}', f'_{self.only_brain_region}'
        elif self.only_stage:
            return f'_{self.only_stage}', ''
        elif self.only_brain_region:
            return '', f'_{self.only_brain_region}'
        else:
            return '', ''

    def _train_evaluate_rf(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Train and evaluate the Random Forest model."""
        base_rf = RandomForestClassifier(random_state=2)
        param_grid_rf = {'n_estimators': [50, 100, 150],
                         'max_features': ['sqrt', 'log2'],
                         'max_depth': [3, 6, 9]}
        evaluator =ModelEvaluator(base_rf, param_grid_rf, self.save_path)
        evaluator.fit_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
    def _train_evaluate_svm(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Train and evaluate the Support Vector Machine model."""
        base_svm = SVC(random_state=3)
        param_grid_svm = {'gamma': [0.1, 0.01, 0.001],
                          'kernel': ['linear', 'rbf'],
                          'C': [.1, 1, 10]}
        evaluator =ModelEvaluator(base_svm, param_grid_svm, self.save_path)
        evaluator.fit_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
    def _train_evaluate_mlp(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """Train and evaluate the Multi-Layer Perceptron model."""
        base_mlp = MLPClassifier(random_state=4)
        param_grid_mlp = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'solver': ['lbfgs', 'sgd', 'adam'],
                          'hidden_layer_sizes': [100, 200, 300, 400, 500]}
        evaluator =ModelEvaluator(base_mlp, param_grid_mlp,self.save_path)
        evaluator.fit_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
# Example usage
# classifier = Classifier(save_path='path/to/save', only_stage='N2', only_brain_region='F')
# classifier.classification(run_class_task=True, run_pca=False)


