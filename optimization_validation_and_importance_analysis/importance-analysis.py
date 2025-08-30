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

warnings.filterwarnings("ignore")
random.seed(1024)
np.random.seed(1024)
yeloh_seed = 2137

# Set `True` if you want to use only age, gender, and mutation data:
cut_input_params = True

int_cols = [
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

base_lvl_name = "Baseline"
pep_lvl_name = "Peptide level"
cont_lvl_name = "Contig level"
scaff_lvl_name = "Scaffold level"

scaffold_col = "scaffold"
prot_change_col = "Protein_Change"

exclude_mutation_col = "exclude_mutation"

list_separator = ";"

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

    if train_data is None:
        return None, mut_num

    results = {}

    _, svm_linear_preds, svm_linear_prob = svm_train_test(
        train_data, train_y, test_data, clf_kwargs=clf_params
    )

    results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
    if "exclude_mutation" in config and excluded_mutation is not None:
        results.update(excluded_mutation.to_dict("records")[0])
    results.update(results1)

    results_frame = pd.DataFrame(results, index=[0])

    return results_frame, mut_num


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("You must specify index of a model")
    m_idx = int(sys.argv[1])

    try:
        config = copy.deepcopy(cfg.configurations[m_idx])
    except IndexError:
        raise IndexError(f"Theres no model at index {m_idx}")

    muts = pd.read_csv(config[contig_file_label], sep="\t", low_memory=False)

    model_name = config[model_name_label]

    if model_name in (pep_lvl_name, base_lvl_name):
        deletion_type = "mutations"
    elif model_name == cont_lvl_name:
        deletion_type = "contigs"
    elif model_name == scaff_lvl_name:
        deletion_type = "scaffolds"
    else:
        deletion_type = None

    overall_results = []

    results_frame, mut_num = svm_experiment(config, cut_input_params, deletion_type)
    results_frame.insert(0, exclude_mutation_col, pd.NA)
    overall_results.append(results_frame)

    if model_name in (pep_lvl_name, base_lvl_name):
        iterator = range(mut_num)
    elif model_name in (cont_lvl_name, scaff_lvl_name):
        common_cols = ["unique_peptides", "popcov_but_sqrt"]
        if model_name == cont_lvl_name:
            del_col = "contig"
        else:
            muts = muts.rename(columns={"Id": scaffold_col})
            del_col = scaffold_col
        squeeze_cols = [del_col] + common_cols
        muts_vc = muts.loc[muts[del_col].notna()][squeeze_cols].value_counts()
        cs_ids = muts_vc.index.get_level_values(del_col)
        if len(cs_ids) != len(cs_ids.unique()):
            raise ValueError("There's a problem with IDs")
        muts_vc = muts_vc.reset_index(name="Count")
        iterator = cs_ids
    else:
        iterator = None

    for stuff_to_remove in tqdm(iterator):
        config["exclude_mutation"] = stuff_to_remove
        results_frame, _ = svm_experiment(config, cut_input_params, deletion_type)
        results_frame.insert(0, exclude_mutation_col, stuff_to_remove)
        if model_name in (cont_lvl_name, scaff_lvl_name):
            cs_info = muts_vc.loc[muts_vc[del_col] == stuff_to_remove].reset_index(
                drop=True
            )
            results_frame = pd.concat([results_frame, cs_info], axis=1)
        overall_results.append(results_frame)

    overall_results = pd.concat(overall_results, ignore_index=True)

    for c in int_cols:
        if c in overall_results.columns:
            overall_results[c] = overall_results[c].astype("Int64")
    if model_name in (pep_lvl_name, base_lvl_name):
        overall_results[exclude_mutation_col] = overall_results[
            exclude_mutation_col
        ].astype("Int64")

    # Adding 'Protein_Change' info
    desc = "Adding additional info"
    protein_change = [""]
    if model_name in (pep_lvl_name, base_lvl_name):
        for i in tqdm(range(1, len(overall_results)), desc=desc):
            row = overall_results.iloc[i]
            chr = row["Chromosome"]
            st_p = row["Start_position"]
            en_p = row["End_position"]
            m = muts[
                (muts["Chromosome"] == chr)
                & (muts["Start_position"] == st_p)
                & (muts["End_position"] == en_p)
            ]
            pc = m[prot_change_col].fillna(value="").unique().tolist()
            pc = list_separator.join(pc)
            protein_change.append(pc)
    else:
        for cs in tqdm(overall_results[del_col][1:], desc=desc):
            pc = (
                muts[muts[del_col] == cs][prot_change_col]
                .fillna(value="")
                .unique()
                .tolist()
            )
            pc = list_separator.join(pc)
            protein_change.append(pc)
    protein_change = pd.Series(protein_change, name=prot_change_col)
    overall_results = pd.concat([overall_results, protein_change], axis=1)
    #

    overall_results.to_csv(
        results_dir / f"mutation-remove-experiments-{model_name.replace(' ', '_')}.csv",
        index=False,
    )
