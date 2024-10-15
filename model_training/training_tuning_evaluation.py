import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, base_model, param_grid, save_path):
        """
        Initializes the ModelEvaluator with the given parameters.

        Parameters:
        - base_model: the model chosen for classification tasks
        - param_grid: dictionary with parameters to optimize during fine-tuning
        - save_path: absolute path where to save results
        """
        self.base_model = base_model
        self.param_grid = param_grid
        self.save_path = save_path

    def fit_and_evaluate(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """
        Trains, tunes, and evaluates the model.

        Parameters:
        - x_train: features values for Training Set
        - y_train: labels for Training Set
        - x_val: features values for Validation Set
        - y_val: labels for Validation Set
        - x_test: features values for Test Set
        - y_test: labels for Test Set
        """
        # 1. Training on Training set
        self.base_model.fit(x_train, y_train)
        params_base = self.base_model.get_params()

        # 2. Fine-tuning on Validation set
        grid_search = GridSearchCV(self.base_model.__class__(**params_base), param_grid=self.param_grid, cv=5)
        grid_search.fit(x_val, y_val)
        best_model = grid_search.best_estimator_

        # 3. Evaluation on Test set
        y_test_predicted = best_model.predict(x_test)
        self._evaluate_model(y_test, y_test_predicted)

    def _evaluate_model(self, y_test, y_test_predicted):
        """
        Evaluates the model and plots the results.

        Parameters:
        - y_test: true labels for the Test Set
        - y_test_predicted: predicted labels for the Test Set
        """
        classes = sorted(set(y_test))
        metrics = classification_report(y_test, y_test_predicted, output_dict=True)
        cm = confusion_matrix(y_test, y_test_predicted, labels=classes)

        # Plotting final results
        fig, ax = plt.subplots(figsize=(14, 7), ncols=2)

        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax[0],
                    xticklabels=classes, yticklabels=classes, cbar=False,
                    annot_kws={"size": 14})
        ax[0].set_xlabel('Predicted', fontsize=16)
        ax[0].set_ylabel('True', fontsize=16)

        # Metrics table
        metrics_df = pd.DataFrame(metrics).transpose().iloc[:4, :3].round(2)
        table = ax[1].table(cellText=metrics_df.values,
                            colLabels=[col.capitalize() for col in metrics_df.columns],
                            rowLabels=metrics_df.index,
                            loc='center',
                            rowLoc='center',
                            cellLoc='center',
                            bbox=[0, 0, 1, 1])

        for key, cell in table.get_celld().items():
            if key[0] == 0 or key[1] == -1:
                cell.set_text_props(fontweight='bold', color='white', fontsize=16)
                cell.set_facecolor('royalblue')
            else:
                cell.set_facecolor((65/255, 105/255, 225/255, 100/255))

        table.auto_set_column_width(col=list(range(len(metrics_df.columns) + 1)))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 4)
        ax[1].axis('off')

        fig.suptitle(str(self.base_model.__class__.__name__), fontsize=18)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, f"{self.base_model.__class__.__name__}.jpg"), dpi=300)
        plt.close()

# Example usage
# from sklearn.ensemble import RandomForestClassifier
# param_grid_rf = {'n_estimators': [50, 100, 150], 'max_features': ['sqrt', 'log2'], 'max_depth': [3, 6, 9]}
# evaluator = ModelEvaluator(RandomForestClassifier(random_state=2), param_grid_rf, save_path='path/to/save')
# evaluator.fit_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
