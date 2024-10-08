import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix



def training_tuning_evaluation(base_model, param_grid, x_train, y_train, x_val, y_val, x_test, y_test, save_path):
    # base_model: model chosen for classification task
    # param_grid: dictionary with parameters to optimize during fine-tuning
    # x_train: features values for Training Set
    # y_train: labels for Training Set
    # x_val: features values for Validation Set
    # y_val: labels for Validation Set
    # x_test: features values for Test Set
    # y_test: labels for Test Set
    # save_path: absolute path where to save results

    # 1. training on Training set
    base_model.fit(x_train, y_train)
    params_base = base_model.get_params()

    # 2. fine-tuning on Validation set
    grid_search = GridSearchCV(base_model.__class__(**params_base), param_grid=param_grid, cv=5)
    grid_search.fit(x_val, y_val)
    best_model = grid_search.best_estimator_

    # 3. evaluation on Test set
    y_test_predicted = best_model.predict(x_test)

    classes = sorted(set(y_test))
    metrics = classification_report(y_test, y_test_predicted, output_dict=True)
    cm = confusion_matrix(y_test, y_test_predicted, labels=classes)

    # Plotting final results
    fig, ax = plt.subplots(figsize=(14, 7), ncols=2)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax[0], xticklabels=classes, yticklabels=classes, cbar=False,
                annot_kws={"size": 14})
    ax[0].set_xlabel('Predicted', fontsize=16)
    ax[0].set_ylabel('True', fontsize=16)

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
            cell.set_text_props(fontweight='bold', color='white', fontsize=16)
            cell.set_facecolor('royalblue')
        else:
            cell.set_facecolor((65/255, 105/255, 225/255, 100/255))

    table.auto_set_column_width(col=list(range(len(metrics_df.columns) + 1)))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 4)
    ax[1].axis('off')

    fig.suptitle(str(base_model.__class__.__name__), fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, str(base_model.__class__.__name__) + '.jpg'), dpi=300)
    plt.close()