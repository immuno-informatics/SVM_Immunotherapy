import copy
import random
import sys
import warnings
from pathlib import Path

import _config as cfg
import DataSets_validation as ds
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn import svm  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm  # noqa: E402

# import polars as pl
# import matplotlib.pyplot as plt
# import kaplanmeier as km
# import pymysql
# import sys
# import optuna

warnings.filterwarnings("ignore")
random.seed(1024)
np.random.seed(1024)
yeloh_seed = 2137

# Set `True` if you want to use only age, gender, and mutation data:
cut_input_params = True

int_cols = [
    "Chromosome",
    "Start_position",
    "End_position",
    "Count",
    "Unique_peptides_narrow",
    "unique_peptides",
]

mut_vec_len_label = "mut_vec_len"
clf_params_label = "clf_params"
model_name_label = "plot_label"
contig_file_label = "contig_file"

pep_lvl_name = "Peptide level"
cont_lvl_name = "Contig level"
scaff_lvl_name = "Scaffold level"

results_dir = Path("Results")
results_dir.mkdir(parents=True, exist_ok=True)


def svm_train_test(train_data, train_y, test_data, verbose=False, clf_kwargs={}):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)
    clf = svm.SVC(probability=True, random_state=yeloh_seed, **clf_kwargs)
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
    evaluations["precision_false"] = precision_score(
        ytest, preds, pos_label=-1, average="binary"
    )
    evaluations["recall_false"] = recall_score(
        ytest, preds, pos_label=-1, average="binary"
    )
    evaluations["f1_false"] = f1_score(ytest, preds, pos_label=-1, average="binary")
    evaluations["accuracy"] = accuracy_score(ytest, preds)
    return evaluations


def svm_experiment(config, cut_input_params, deletion_type):
    dimension_of_embedding_vectors = config[mut_vec_len_label]
    clf_params = config[clf_params_label]

    train_data, train_y, test_data, test_y, _, excluded_mutation, mut_num = (
        ds.transforming_Braun_dataset(
            config,
            dimension_of_embedding_vectors=dimension_of_embedding_vectors,
            cut_input_params=cut_input_params,
            deletion_type=deletion_type,
        )
    )

    # OMG
    if train_data is None:
        return None, mut_num
    # OMG

    # possible_svm_models = [{"kernel": "rbf"}]

    # local_results_frames = pd.DataFrame()
    # index = 0
    # for params in possible_svm_models:
    results = {}
    # results["METHOD"] = "Predictor:SVM" + str(clf_params)
    results["Info"] = str(config)
    # print("SVM Meta-parameters : " + str(clf_params))

    _, svm_linear_preds, svm_linear_prob = svm_train_test(
        train_data, train_y, test_data, clf_kwargs=clf_params
    )

    results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
    if "exclude_mutation" in config and excluded_mutation is not None:
        results.update(excluded_mutation.to_dict("records")[0])
    results.update(results1)

    results_frame = pd.DataFrame(results, index=[0])

    return results_frame, mut_num

    # if local_results_frames.empty:
    #     local_results_frames = [results_frame]
    # else:
    #     local_results_frames = [local_results_frames, results_frame]
    # local_results_frames = pd.concat(local_results_frames)
    # # index += 1
    # return pd.DataFrame(local_results_frames), None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("You must specify index of a model")
    m_idx = int(sys.argv[1])

    config = copy.deepcopy(cfg.configurations[m_idx])

    model_name = config[model_name_label]

    if model_name == pep_lvl_name:
        deletion_type = "mutations"
    elif model_name == cont_lvl_name:
        deletion_type = "contigs"
    elif model_name == scaff_lvl_name:
        deletion_type = "scaffolds"
    else:
        deletion_type = None

    overall_results = []

    results_frame, mut_num = svm_experiment(config, cut_input_params, deletion_type)

    overall_results.append(results_frame)

    if model_name == pep_lvl_name:
        iterator = range(5)  # mut_num
    elif model_name == cont_lvl_name or model_name == scaff_lvl_name:
        conts_scaffs_file = config[contig_file_label]
        conts_scaffs = pd.read_csv(conts_scaffs_file, sep="\t", low_memory=False)
        common_cols = ["unique_peptides", "popcov_but_sqrt"]
        if model_name == cont_lvl_name:
            del_col = "contig"
        else:
            del_col = "Id"
        squeeze_cols = [del_col] + common_cols
        conts_scaffs = conts_scaffs.loc[conts_scaffs[del_col].notna()][
            squeeze_cols
        ].value_counts()
        cs_ids = conts_scaffs.index.get_level_values(del_col)
        if len(cs_ids) != len(cs_ids.unique()):
            raise ValueError("There's a problem with IDs")
        conts_scaffs = conts_scaffs.reset_index(name="Count")
        iterator = cs_ids
    else:
        iterator = None

    for stuff_to_remove in tqdm(iterator):
        config["exclude_mutation"] = stuff_to_remove
        results_frame, _ = svm_experiment(config, cut_input_params, deletion_type)
        overall_results.append(results_frame)

    overall_results = pd.concat(overall_results, ignore_index=True)

    for c in int_cols:
        if c in overall_results.columns:
            overall_results[c] = overall_results[c].astype("Int64")

    # overall_results.to_csv(
    #     results_dir / f"mutation-remove-experiments-{model_name.replace(' ', '_')}.csv",
    #     index=False,
    # )
