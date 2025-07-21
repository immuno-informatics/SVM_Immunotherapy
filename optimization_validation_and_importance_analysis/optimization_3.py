"""."""

import random

import _config as cfg
import DataSets_validation as ds
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn

patch_sklearn()

import optuna  # noqa: E402
from optuna.distributions import CategoricalDistribution  # noqa: E402
from optuna.distributions import FloatDistribution  # noqa: E402
from optuna.distributions import IntDistribution  # noqa: E402
from optuna.integration import OptunaSearchCV  # noqa: E402
from sklearn import svm  # noqa: E402
# from sklearn.metrics import fbeta_score  # noqa: E402
# from sklearn.metrics import make_scorer  # noqa: E402
# from sklearn.metrics import matthews_corrcoef  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.metrics import precision_score  # noqa: E402
from sklearn.metrics import recall_score  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import PredefinedSplit  # noqa: E402
from tqdm import tqdm  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

yeloh_seed = 2137

random.seed(1024)
np.random.seed(1024)

# Config

mutations_vector_len_range = np.linspace(1, 1000, num=1000, dtype=int)
# mutations_vector_len_range = [887, 243, 375, 881]
# mutations_vector_len_range = np.linspace(3001, 4000, num=1000, dtype=int)

opt_n_trials = 150
opt_n_jobs = 1  # Results are unreproducible if > 1
opt_scoring = "balanced_accuracy"
# opt_scoring = "f1"

param_distributions = {
    "kernel": CategoricalDistribution(["linear", "rbf", "poly", "sigmoid"]),
    "C": FloatDistribution(1e-3, 1e3, step=1e-3),
    "gamma": FloatDistribution(1e-4, 1, step=1e-4),
    "coef0": FloatDistribution(1e-2, 1e2, step=1e-2),
    "degree": IntDistribution(1, 1e1),
    # "C": FloatDistribution(0.01, 100),
    # "gamma": FloatDistribution(0.0001, 0.1),
    # "coef0": FloatDistribution(0, 5),
    # "degree": IntDistribution(2, 5),
}

svc_core_args = {"probability": True, "random_state": yeloh_seed}

x_train_name = "x_train"
y_train_name = "y_train"
x_test_name = "x_test"
y_test_name = "y_test"

label_key = "plot_label"

#


def oversample_x_y(x, y):
    sm = SMOTE(random_state=432)
    new_x, new_y = sm.fit_resample(x, y)
    return new_x, new_y


def optuna_searching(train_data, train_y, test_data, test_y):
    train_data, train_y = oversample_x_y(train_data, train_y)

    clf = svm.SVC(**svc_core_args)

    X_full = np.concatenate((train_data, test_data))
    y_full = np.concatenate((train_y, test_y))
    train_indices = [-1] * len(train_data)
    validation_indices = [0] * len(test_data)
    test_fold = np.array(train_indices + validation_indices)
    ps = PredefinedSplit(test_fold)

    search = OptunaSearchCV(
        clf,
        param_distributions,
        n_trials=opt_n_trials,
        timeout=None,
        cv=ps,
        scoring=opt_scoring,
        refit=False,
        random_state=yeloh_seed,
        n_jobs=opt_n_jobs,
    ).fit(X_full, y_full)

    return search


def svm_train_test(train_data, train_y, test_data, clf_kwargs=None):
    if clf_kwargs is None:
        clf_kwargs = {}
    train_data, train_y = oversample_x_y(train_data, train_y)
    clf = svm.SVC(**svc_core_args, **clf_kwargs)
    clf = clf.fit(train_data, train_y)

    predictions = clf.predict(test_data)
    probs = clf.predict_proba(test_data)
    probSV = [i[1] for i in probs]
    new_pd = pd.DataFrame(probSV).values.flatten()

    return clf, predictions, new_pd


best_score = {}
best_vec_len = {}
best_params = {}
best_data = {}

for config in tqdm(cfg.configurations[1:]):
    model_label = config[label_key]

    best_score[model_label] = 0
    best_vec_len[model_label] = None
    best_params[model_label] = None
    best_data[model_label] = {
        x_train_name: None,
        y_train_name: None,
        x_test_name: None,
        y_test_name: None,
    }

    for v_len in tqdm(mutations_vector_len_range, leave=False):
        train_data, train_y, test_data, test_y, _, _ = ds.transforming_Braun_dataset(
            config, v_len
        )

        search = optuna_searching(train_data, train_y, test_data, test_y)

        b_score = search.best_score_

        if b_score > best_score[model_label]:
            best_score[model_label] = b_score
            best_vec_len[model_label] = v_len
            best_params[model_label] = search.best_params_
            best_data[model_label] = {
                x_train_name: train_data,
                y_train_name: train_y,
                x_test_name: test_data,
                y_test_name: test_y,
            }

    print(
        f"{model_label}:\n"
        f"  Best score: {best_score[model_label]:.3f}\n"
        f"  Best vector length: {best_vec_len[model_label]}\n"
        f"  Best parameters: {best_params[model_label]}"
    )

    _, y_pred, y_proba = svm_train_test(
        best_data[model_label][x_train_name],
        best_data[model_label][y_train_name],
        best_data[model_label][x_test_name],
        clf_kwargs=best_params[model_label],
    )
    test_y = best_data[model_label][y_test_name]

    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred)
    rec = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    rauc = roc_auc_score(test_y, y_proba)
    mean_s = (prec + rec + f1 + acc + rauc) / 5

    print(
        f"    Accuracy: {acc:.3f}\n"
        f"    Precision: {prec:.3f}\n"
        f"    Recall: {rec:.3f}\n"
        f"    F1 Score: {f1:.3f}\n"
        f"    ROC AUC: {rauc:.3f}\n"
        f"      _MEAN_: {mean_s:.3f}"
    )

# for k in best_score.keys():
#     print(
#         f"{k}:\n"
#         f"  Best score: {best_score[k]:.3f}\n"
#         f"  Best vector length: {best_vec_len[k]}\n"
#         f"  Best parameters: {best_params[k]}"
#     )

#     _, y_pred, y_proba = svm_train_test(
#         best_data[k][x_train_name],
#         best_data[k][y_train_name],
#         best_data[k][x_test_name],
#         clf_kwargs=best_params[k],
#     )
#     test_y = best_data[k][y_test_name]

#     acc = accuracy_score(test_y, y_pred)
#     prec = precision_score(test_y, y_pred)
#     rec = recall_score(test_y, y_pred)
#     f1 = f1_score(test_y, y_pred)
#     rauc = roc_auc_score(test_y, y_proba)
#     mean_s = (prec + rec + f1 + acc + rauc) / 5

#     print(
#         f"    Accuracy: {acc:.3f}\n"
#         f"    Precision: {prec:.3f}\n"
#         f"    Recall: {rec:.3f}\n"
#         f"    F1 Score: {f1:.3f}\n"
#         f"    ROC AUC: {rauc:.3f}\n"
#         f"      _MEAN_: {mean_s:.3f}"
#     )
