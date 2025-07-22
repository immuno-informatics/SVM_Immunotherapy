"""."""

import copy
import random

import _config as cfg
import DataSets_validation as ds
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn

patch_sklearn()

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.terminator import BestValueStagnationEvaluator  # noqa: E402
from optuna.terminator import StaticErrorEvaluator  # noqa: E402
from optuna.terminator import Terminator  # noqa: E402
from optuna.terminator.callback import TerminatorCallback  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.metrics import precision_score  # noqa: E402
from sklearn.metrics import recall_score  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import PredefinedSplit  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

yeloh_seed = 2137

random.seed(1024)
np.random.seed(1024)

# Config

opt_n_trials = 10_000
opt_n_jobs = 1  # Results are unreproducible if > 1
opt_max_stagnation_trials = 1_000

cv_n_jobs = 1
cv_scoring = "balanced_accuracy"

svc_core_args = {"probability": True, "random_state": yeloh_seed}

label_key = "plot_label"
weights_key = "weights"
weights_groups = ["PS", "TF", "CF", "BP", "MT", "GE"]

v_len_name = "v_len"

#


def oversample_x_y(x, y):
    sm = SMOTE(random_state=432)
    new_x, new_y = sm.fit_resample(x, y)
    return new_x, new_y


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


def objective(trial, config):
    # SVM parameters to optimize
    optional_clf_params = {}
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    c = trial.suggest_float("C", 0.001, 1_000.0, log=True)
    if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
        optional_clf_params["gamma"] = trial.suggest_float(
            "gamma", 0.0001, 10.0, log=True
        )
        if kernel == "poly" or kernel == "sigmoid":
            optional_clf_params["coef0"] = trial.suggest_float(
                "coef0", -1, 10.0, step=0.01
            )
            if kernel == "poly":
                optional_clf_params["degree"] = trial.suggest_int("degree", 2, 10)

    clf = svm.SVC(kernel=kernel, C=c, **svc_core_args, **optional_clf_params)

    # Mutation vector length optimization
    v_len = trial.suggest_int(v_len_name, 1, 4_000)

    # Weights of input parameter groups optimization
    weights = {wg: trial.suggest_float(wg, 0.0, 1.0, step=0.1) for wg in weights_groups}
    conf_copy = copy.deepcopy(config)
    conf_copy[weights_key] = weights

    train_data, train_y, test_data, test_y, _, _ = ds.transforming_Braun_dataset(
        conf_copy, v_len
    )
    train_data, train_y = oversample_x_y(train_data, train_y)

    X_full = np.concatenate((train_data, test_data))
    y_full = np.concatenate((train_y, test_y))
    train_indices = [-1] * len(train_data)
    validation_indices = [0] * len(test_data)
    split = np.array(train_indices + validation_indices)
    ps = PredefinedSplit(split)

    scores = cross_val_score(
        clf, X_full, y_full, n_jobs=cv_n_jobs, cv=ps, scoring=cv_scoring
    )

    return scores.mean()


if __name__ == "__main__":
    for config in tqdm(cfg.configurations, desc="Models"):
        model_label = config[label_key]

        sampler = TPESampler(seed=yeloh_seed)
        improvement_evaluator = BestValueStagnationEvaluator(
            max_stagnation_trials=opt_max_stagnation_trials
        )
        error_evaluator = StaticErrorEvaluator(0)
        terminator = TerminatorCallback(
            Terminator(
                improvement_evaluator=improvement_evaluator,
                error_evaluator=error_evaluator,
            )
        )

        study = optuna.create_study(
            study_name=f"{model_label}", direction="maximize", sampler=sampler
        )
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=opt_n_trials,
            n_jobs=opt_n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[terminator],
        )

        # Print the best model parameters and results

        best_score = study.best_value
        best_svm_params = study.best_params
        best_v_len = best_svm_params.pop(v_len_name)
        best_weights = {wg: best_svm_params.pop(wg) for wg in weights_groups}

        config[weights_key] = best_weights

        train_data, train_y, test_data, test_y, _, _ = ds.transforming_Braun_dataset(
            config, best_v_len
        )
        _, y_pred, y_proba = svm_train_test(
            train_data, train_y, test_data, clf_kwargs=best_svm_params
        )
        acc = accuracy_score(test_y, y_pred)
        prec = precision_score(test_y, y_pred)
        rec = recall_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)
        rauc = roc_auc_score(test_y, y_proba)
        mean_s = (prec + rec + f1 + acc + rauc) / 5

        print(
            f"\n{model_label}:\n"
            f"  Best score: {best_score:.3f}\n"
            f"  Best vector length: {best_v_len}\n"
            f"  Best SVM parameters: {best_svm_params}\n"
            f"  Best weights: {best_weights}\n"
            f"    Accuracy: {acc:.3f}\n"
            f"    Precision: {prec:.3f}\n"
            f"    Recall: {rec:.3f}\n"
            f"    F1 Score: {f1:.3f}\n"
            f"    ROC AUC: {rauc:.3f}\n"
            f"      _MEAN_: {mean_s:.3f}"
        )
