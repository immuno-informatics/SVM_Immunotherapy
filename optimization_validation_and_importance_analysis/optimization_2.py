"""."""

import random

import _config as cfg
import DataSets_validation as ds
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn import svm  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit  # noqa: E402
from tqdm import tqdm  # noqa: E402

random.seed(1024)
np.random.seed(1024)

# Config

# mutations_vector_len_range = np.linspace(1, 1000, num=1000, dtype=int)
# mutations_vector_len_range = [149, 327, 77, 21]
mutations_vector_len_range = np.linspace(2001, 3000, num=1000, dtype=int)

gs_n_jobs = 60

gs_param_grid = [
    {"C": [1, 5, 10, 50, 100, 1000], "kernel": ["linear"]},
    {
        "C": [1, 5, 10, 50, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf"],
    },
    {
        "C": [1, 5, 10, 50, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["poly"],
        "degree": [2, 3, 4, 5],
        "coef0": [0, 1, 2, 3, 4, 5],
    },
    {
        "C": [1, 5, 10, 50, 100, 1000],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["sigmoid"],
        "coef0": [0, 1, 2, 3, 4, 5],
    },
]

label_key = "plot_label"

#


def average_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)
    probSV = [i[1] for i in y_proba]
    new_pd = pd.DataFrame(probSV)
    probs = new_pd.values.flatten()

    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    rauc = roc_auc_score(y, probs)

    return (prec + rec + f1 + acc + rauc) / 5


def grid_searching(train_data, train_y, test_data, test_y):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)

    clf = svm.SVC(probability=True, random_state=2137)

    X_full = np.concatenate((train_data, test_data))
    y_full = np.concatenate((train_y, test_y))
    train_indices = [-1] * len(train_data)
    validation_indices = [0] * len(test_data)
    test_fold = np.array(train_indices + validation_indices)
    ps = PredefinedSplit(test_fold)

    search = GridSearchCV(
        clf,
        gs_param_grid,
        scoring={
            "mean": average_scorer,
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
        refit="mean",
        cv=ps,
        n_jobs=gs_n_jobs,
        verbose=0,
    ).fit(X_full, y_full)

    return search


best_score = {}
best_vec_len = {}
best_params = {}
best_scores = {}

for config in tqdm(cfg.configurations):
    model_label = config[label_key]
    best_score[model_label] = 0
    best_vec_len[model_label] = None
    best_params[model_label] = None
    best_scores[model_label] = "None"

    for v_len in tqdm(mutations_vector_len_range, leave=False):
        train_data, train_y, test_data, test_y, test_pfs, _ = (
            ds.transforming_Braun_dataset(config, v_len)
        )

        search = grid_searching(train_data, train_y, test_data, test_y)

        b_score = search.best_score_

        if b_score > best_score[model_label]:
            best_score[model_label] = b_score
            best_vec_len[model_label] = v_len
            best_params[model_label] = search.best_params_

            b_index = search.best_index_
            acc = search.cv_results_["mean_test_accuracy"][b_index]
            prec = search.cv_results_["mean_test_precision"][b_index]
            rec = search.cv_results_["mean_test_recall"][b_index]
            f1 = search.cv_results_["mean_test_f1"][b_index]
            rauc = search.cv_results_["mean_test_roc_auc"][b_index]
            best_scores[model_label] = ""
            best_scores[model_label] += f"    Accuracy: {acc:.3f}\n"
            best_scores[model_label] += f"    Precision: {prec:.3f}\n"
            best_scores[model_label] += f"    Recall: {rec:.3f}\n"
            best_scores[model_label] += f"    F1 Score: {f1:.3f}\n"
            best_scores[model_label] += f"    ROC AUC: {rauc:.3f}"

for k in best_score.keys():
    print(
        f"{k}:\n"
        f"  Best score: {best_score[k]:.3f}\n"
        f"  Best vector length: {best_vec_len[k]}\n"
        f"  Best parameters: {best_params[k]}\n"
        f"{best_scores[k]}"
    )
