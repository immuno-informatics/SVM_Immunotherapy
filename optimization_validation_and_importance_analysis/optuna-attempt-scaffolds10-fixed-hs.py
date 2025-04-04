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

def svm_train_test(train_data, train_y, test_data, classifier_params ={}, verbose=False):
    sm = SMOTE(random_state= 432)
    train_data, train_y = sm.fit_resample(train_data, train_y)
    clf = svm.SVC(kernel = classifier_params["kernel"],probability=True)
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
    evaluations["precision_false"] = precision_score(ytest, preds, pos_label=-1 ,average="binary")
    evaluations["recall_false"] = recall_score(ytest, preds,pos_label=-1, average="binary")
    evaluations["f1_false"] = f1_score(ytest, preds,pos_label= -1, average="binary")
    evaluations["accuracy"] = accuracy_score(ytest, preds)
    return evaluations

from numpy import isnan


def svm_experiment_and_km_curve(trial):
    PS = trial.suggest_float("PS", 0.0, 1.0, step=0.1)
    TF = trial.suggest_float("TF", 0.0, 1.0, step=0.1)
    CF = trial.suggest_float("CF", 0.0, 1.0, step=0.1)
    BP = trial.suggest_float("BP", 0.0, 1.0, step=0.1)
    MT = trial.suggest_float("MT", 0.0, 1.0, step=0.1)
    GE = trial.suggest_float("GE", 0.0, 1.0, step=0.1)
    n_unique_HLA = trial.suggest_categorical("n_unique_HLA", [True, False])
    unique_peptides = trial.suggest_categorical("unique_peptides",[True, False])
    popcov_but_sqrt = True
    hotspots = True
    kernel = "rbf"
    weights = {"PS":PS,
               "TF":TF,
               "CF":CF,
               "BP":BP,
               "MT":MT,
               "GE":GE
               }
    HS_features = []
    if(n_unique_HLA):
        HS_features.append("n_unique_HLA")
    if(unique_peptides):
        HS_features.append("unique_peptides")
    if(popcov_but_sqrt):
        HS_features.append("popcov_but_sqrt")
    additional_info = {
        "PRIMARY_TUMOR_ONLY":False,
        "with_mutations":True, 
        "random_contigs":False, 
        "hotspots":hotspots, 
        "weights":weights, 
        "contig_file":"../data/Braun_hg38_scaff_gp10.tsv",
        "HS_features":HS_features
        }
    
    train_data, train_y, test_data, test_y, test_pfs , excluded_mutation = ds.transforming_Braun_dataset(additional_info)
    
    results = {}
    results["METHOD"] = "Predictor:SVM"+ str(kernel)
    results["Info"] = str(additional_info)

    
    _,svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, {"kernel":kernel})
    
    results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
    results.update(results1)
    results1 = km.fit(test_pfs,test_y,svm_linear_preds)
    
    if isnan(results1["logrank_P"]):
        results["logP"] = 1
    else:
        results["logP"] = results1["logrank_P"]

    return results["aroc"], results["logP"]

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="scaf10-level-random-fixed-hs", 
        storage="mysql+pymysql://optuna:optunapassword@localhost/optuna"
    )
    study.optimize(svm_experiment_and_km_curve, n_trials=100, catch=(ValueError,))