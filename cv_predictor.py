from collections import defaultdict
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, train_test_split, cross_val_score, cross_validate
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

from ensemble_classifier import SimpleAveragingClassifier
from data_preproc import pre_process_data_for_ensembling, PATIENT_NAME_COL, CELL_TYPE_COL, CELL_NAME_COL, PATIENT_STATE_COL

FILEPATH = "raw_data.csv"
FIGURES_DIR = "figures"

TEST_SIZE = 0.2

LEAVE_ONE_OUT = False
DEFAULT_SPLITS_CV = 5
SCALE_DATA = True

# for reproducibility
RANDOM_STATE = 53


class StratifiedCustomCV(BaseCrossValidator):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        # Ensure y is a numpy array
        y = np.asarray(y)
        n = len(y)

        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        # Shuffle indices within each class for randomness
        for indices in class_indices.values():
            np.random.shuffle(indices)

        # Create folds
        folds = [[] for _ in range(self.n_splits)]
        for indices in class_indices.values():
            # Distribute indices across folds
            for i, idx in enumerate(indices):
                folds[i % self.n_splits].append(idx)

        # Yield train-test splits
        for i in range(self.n_splits):
            test_idx = np.array(folds[i], dtype=int)
            train_idx = np.array(
                [idx for fold in folds if fold is not folds[i] for idx in fold], dtype=int
            )
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def main():

    # Read data
    df = pd.read_csv(FILEPATH, delimiter=";")

    df_grouped_patient = [group for _, group in df.groupby(PATIENT_NAME_COL)]
    df_grouped_patient_state = [group[PATIENT_STATE_COL].iloc[0] for group in df_grouped_patient]

    label_encoder_patient_state = LabelEncoder()
    label_encoder_patient_state.fit(np.array(df_grouped_patient_state))
    label_encoder_patient_state.classes_ = ['sain', 'malade'] # force 'sain' to be class 0, and 'malade' to be class 1

    label_encoder_cell_type = LabelEncoder()
    label_encoder_cell_type.fit(df[CELL_TYPE_COL])
    label_encoder_patient_state = LabelEncoder()
    label_encoder_patient_state.fit(df[PATIENT_STATE_COL])
    X, y = pre_process_data_for_ensembling(df, label_encoder_cell_type, label_encoder_patient_state)

    # Custom Cross-Validator
    if LEAVE_ONE_OUT:
        n_splits_cv = len(y)
    else:
        n_splits_cv = DEFAULT_SPLITS_CV

    custom_cv = StratifiedCustomCV(n_splits=n_splits_cv)

    # Train Enesemble logistic regression model with Cross-Validation
    base_model = LogisticRegression(max_iter=1000)
    ensemble_model = SimpleAveragingClassifier(base_model, scale_data=SCALE_DATA)

    #scores = cross_val_score(ensemble_model, X, y, cv=custom_cv, scoring='roc_auc')
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'}
    metrics_for_pot = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'roc_auc': 'ROC AUC'}
    scores = cross_validate(ensemble_model, X, y, cv=custom_cv, scoring=scoring)

    print(f"Custom Cross-Validation accuracy scores: {scores}")
    if isinstance(scores, list):
        print(f"Mean auc score: {np.mean(scores):.2f}")
    elif isinstance(scores, dict):
        for k, v in scores.items():
            print(f"Mean {k} score: {np.mean(v):.2f}")

    # Prepare the data for plotting
    del scores['fit_time']
    del scores['score_time']
    metrics = list(scores.keys())
    values = [scores[metric] for metric in metrics]
    metrics = [metrics_for_pot[m.lstrip('test_')] for m in metrics]

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(values, showmeans=True)
    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels(metrics)
    ax.set_title(f"{n_splits_cv}-fold cross validation results")
    ax.set_ylabel("Values")
    ax.grid(True)

    # Save and close the plot
    save_path = f"{n_splits_cv}fold_ens_results"
    if SCALE_DATA:
        save_path += "_scaled.png"
    else:
        save_path += ".png"

    plt.savefig(os.path.join(FIGURES_DIR, save_path))
    plt.close()


if __name__ == "__main__":
    main()