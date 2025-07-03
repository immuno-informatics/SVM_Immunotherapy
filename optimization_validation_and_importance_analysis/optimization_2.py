"""."""

import random
import warnings

warnings.filterwarnings("ignore")

import _config as cfg  # noqa: E402
import DataSets_validation as ds  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearnex import patch_sklearn  # noqa: E402

patch_sklearn()

from sklearn import svm  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV  # noqa: E402
from tqdm import tqdm  # noqa: E402

random.seed(1024)
np.random.seed(1024)

# Config

mutations_vector_len_range = np.linspace(1, 1000, num=1000, dtype=int)

gs_cv = None
gs_n_jobs = 50

label_key = "plot_label"

#


def grid_searching(train_data, train_y):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)

    param_grid = [
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

    clf = svm.SVC(probability=True, random_state=2137)

    search = GridSearchCV(
        clf,
        param_grid,
        # scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
        scoring="accuracy",
        # refit="accuracy",
        cv=gs_cv,
        # scoring="accuracy",
        n_jobs=gs_n_jobs,
        verbose=0,
    ).fit(train_data, train_y)

    return search


best_score = {}
best_vec_len = {}
best_params = {}

for config in tqdm(cfg.configurations):
    best_score[config[label_key]] = 0
    best_vec_len[config[label_key]] = None
    best_params[config[label_key]] = None

    for v_len in tqdm(mutations_vector_len_range, leave=False):
        train_data, train_y, test_data, test_y, test_pfs, _ = (
            ds.transforming_Braun_dataset(config, v_len)
        )

        search = grid_searching(train_data, train_y)

        b_clf = search.best_estimator_
        b_par = search.best_params_

        svm_linear_preds = b_clf.predict(test_data)
        probs = b_clf.predict_proba(test_data)
        probSV = [i[1] for i in probs]
        svm_prob = pd.DataFrame(probSV)
        probs = svm_prob.values.flatten()

        prec = precision_score(test_y, svm_linear_preds)
        rec = recall_score(test_y, svm_linear_preds)
        f1 = f1_score(test_y, svm_linear_preds)
        acc = accuracy_score(test_y, svm_linear_preds)
        rauc = roc_auc_score(test_y, probs)

        da_score = (prec + rec + f1 + acc + rauc) / 5
        if da_score > best_score[config[label_key]]:
            best_score[config[label_key]] = da_score
            best_vec_len[config[label_key]] = v_len
            best_params[config[label_key]] = b_par

for k in best_score.keys():
    print(
        f"{k}:\n"
        f"  Best score: {best_score[k]}\n"
        f"  Best vector length: {best_vec_len[k]}\n"
        f"  Best parameters: {best_params[k]}"
    )
