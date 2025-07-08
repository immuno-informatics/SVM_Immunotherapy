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
from tqdm import tqdm  # noqa: E402

random.seed(1024)
np.random.seed(1024)

# Config

mutations_vector_len_range = np.linspace(2001, 4000, num=2000, dtype=int)

clf_kwargs = {"kernel": "rbf"}

label_key = "plot_label"

#


def svm_train_test(train_data, train_y, test_data, verbose=False, clf_kwargs={}):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)
    clf = svm.SVC(probability=True, random_state=2137, **clf_kwargs)
    clf = clf.fit(train_data, train_y)

    predictions = clf.predict(test_data)
    probs = clf.predict_proba(test_data)
    probSV = [i[1] for i in probs]
    if verbose:
        print(probSV)
    new_pd = pd.DataFrame(probSV)
    return clf, predictions, new_pd


best_score = {}
best_vec_len = {}

for config in tqdm(cfg.configurations):
    best_score[config[label_key]] = 0
    best_vec_len[config[label_key]] = None

    for v_len in tqdm(mutations_vector_len_range, leave=False):
        train_data, train_y, test_data, test_y, test_pfs, _ = (
            ds.transforming_Braun_dataset(config, v_len)
        )

        model, svm_linear_preds, svm_prob = svm_train_test(
            train_data, train_y, test_data, clf_kwargs=clf_kwargs
        )

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

for k in best_score.keys():
    print(
        f"{k}:\n"
        f"  Best score: {best_score[k]}\n"
        f"  Best vector length: {best_vec_len[k]}"
    )
