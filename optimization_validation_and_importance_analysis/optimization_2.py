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
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
# from sklearn.metrics import fbeta_score  # noqa: E402
# from sklearn.metrics import make_scorer  # noqa: E402
# from sklearn.metrics import matthews_corrcoef  # noqa: E402
from sklearn.metrics import (precision_score, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit  # noqa: E402
from tqdm import tqdm  # noqa: E402

random.seed(1024)
np.random.seed(1024)

# Config

mutations_vector_len_range = np.linspace(1, 4000, num=4000, dtype=int)
# mutations_vector_len_range = [887, 243, 375, 881]
# mutations_vector_len_range = np.linspace(3001, 4000, num=1000, dtype=int)

gs_n_jobs = 50

p_list_c = [0.01, 0.1, 1, 5, 10, 50, 100]
p_list_gamma = ["auto", 0.1, 0.01, 0.001, 0.0001]
p_list_coef0 = [0, 1, 2, 3, 4, 5]
p_list_degree = [2, 3, 4, 5]

gs_param_grid = [
    {"C": p_list_c, "kernel": ["linear"]},
    {
        "C": p_list_c,
        "gamma": p_list_gamma,
        "kernel": ["rbf"],
    },
    {
        "C": p_list_c,
        "gamma": p_list_gamma,
        "kernel": ["poly"],
        "degree": p_list_degree,
        "coef0": p_list_coef0,
    },
    {
        "C": p_list_c,
        "gamma": p_list_gamma,
        "kernel": ["sigmoid"],
        "coef0": p_list_coef0,
    },
]

label_key = "plot_label"

#


def average_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)
    probSV = [i[1] for i in y_proba]
    probs = pd.DataFrame(probSV).values.flatten()

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
            "f1": "f1",  # ok
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",  # ok
            # "fbeta": make_scorer(fbeta_score, beta=1),  # ok
            # "average_precision": "average_precision",  # nope
            # "matthews_corrcoef": make_scorer(matthews_corrcoef),  # ok
        },
        refit="balanced_accuracy",
        cv=ps,
        n_jobs=gs_n_jobs,
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
            #
            # ba = search.cv_results_["mean_test_balanced_accuracy"][b_index]
            # best_scores[model_label] += f"\n    balanced_accuracy: {ba:.3f}\n"
            # fb = search.cv_results_["mean_test_fbeta"][b_index]
            # best_scores[model_label] += f"    fbeta: {fb:.3f}\n"
            # ap = search.cv_results_["mean_test_average_precision"][b_index]
            # best_scores[model_label] += f"    average_precision: {ap:.3f}\n"
            # mc = search.cv_results_["mean_test_matthews_corrcoef"][b_index]
            # best_scores[model_label] += f"    matthews_corrcoef: {mc:.3f}"
            #
            m = search.cv_results_["mean_test_mean"][b_index]
            best_scores[model_label] += f"\n      _MEAN_: {m:.3f}"

for k in best_score.keys():
    print(
        f"{k}:\n"
        f"  Best score: {best_score[k]:.3f}\n"
        f"  Best vector length: {best_vec_len[k]}\n"
        f"  Best parameters: {best_params[k]}\n"
        f"{best_scores[k]}"
    )
