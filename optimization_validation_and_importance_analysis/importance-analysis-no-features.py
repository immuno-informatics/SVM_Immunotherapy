import pandas as pd
import polars as pl
import numpy as np
import random
import DataSets_validation as ds
from sklearnex import patch_sklearn
patch_sklearn()
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import kaplanmeier as km
import warnings
import pymysql
import sys
import optuna
warnings.filterwarnings("ignore")
random.seed(1024)
np.random.seed(1024)

def svm_train_test(train_data, train_y, test_data, classifier_params={}, verbose=False):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)
    clf = svm.SVC(kernel=classifier_params["kernel"], probability=True)
    clf = clf.fit(train_data, train_y)

    predictions = clf.predict(test_data)
    probs = clf.predict_proba(test_data)
    probSV = [i[1] for i in probs]
    if verbose:
        print(probSV)
    new_pd = pd.DataFrame(probSV)
    return clf, predictions, new_pd

def evaluate(ytest, preds, yprobs):
    evaluations = {}
    evaluations["aroc"] = roc_auc_score(ytest, yprobs)
    evaluations["precision"] = precision_score(ytest, preds, average="macro")
    evaluations["recall"] = recall_score(ytest, preds, average="macro")
    evaluations["f1"] = f1_score(ytest, preds, average="macro")
    evaluations["precision_true"] = precision_score(ytest, preds, average="binary")
    evaluations["recall_true"] = recall_score(ytest, preds, average="binary")
    evaluations["f1_true"] = f1_score(ytest, preds, average="binary")
    evaluations["precision_false"] = precision_score(ytest, preds, pos_label=-1, average="binary")
    evaluations["recall_false"] = recall_score(ytest, preds, pos_label=-1, average="binary")
    evaluations["f1_false"] = f1_score(ytest, preds, pos_label=-1, average="binary")
    evaluations["accuracy"] = accuracy_score(ytest, preds)
    return evaluations

def svm_experiment(additional_info={}):
    train_data, train_y, test_data, test_y, _, excluded_mutation = ds.transforming_Braun_dataset(additional_info)
    possible_svm_models = [{"kernel": "rbf"}]

    local_results_frames = pd.DataFrame()
    index = 0
    for params in possible_svm_models:
        results = {}
        results["METHOD"] = "Predictor:SVM" + str(params)
        results["Info"] = str(additional_info)
        #print("SVM Meta-parameters : " + str(params))

        _, svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, params)

        results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
        if ds.param_check(additional_info, "exclude_mutation"):
            results.update(excluded_mutation.to_dict("records")[0])
        results.update(results1)

        results_frame = pd.DataFrame(results, index=[index])
        #print(results_frame.dtypes)
        if local_results_frames.empty:
            local_results_frames = [results_frame]
        else:
            local_results_frames = [local_results_frames, results_frame]
        local_results_frames = pd.concat(local_results_frames)
        index += 1
        #print(local_results_frames.dtypes)
    return pd.DataFrame(local_results_frames)

weights = {"PS": 0.8, "TF": 0.0, "CF": 0.5, "BP": 0.7, "MT": 0.8, "GE": 0.1}
additional_info = {
    "PRIMARY_TUMOR_ONLY": False,
    "with_mutations": True,
    "random_contigs": False,
    "hotspots": False,
    "weights": weights,
    "contig_file": "data/Michal_combined_set_14_02_2025.tsv",
    "HS_features": []
}

overall_results = pd.DataFrame()
results_frame = svm_experiment(additional_info=additional_info)
overall_results_frames = [overall_results, results_frame]
overall_results = pd.concat(overall_results_frames)

for chr_to_remove in range(0, 567):
    print("excluding mutation: " + str(chr_to_remove))
    additional_info["exclude_mutation"] = chr_to_remove
    results_frame = svm_experiment(additional_info)
    overall_results_frames = [overall_results, results_frame]
    overall_results = pd.concat(overall_results_frames)

overall_results.to_csv("../data/Bruan_MutationExperiments_nofeatures.csv")
