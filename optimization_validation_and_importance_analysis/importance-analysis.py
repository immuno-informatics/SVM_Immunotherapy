"""."""

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
    balanced_accuracy_score,
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

mut_vec_len_label = "mut_vec_len"
clf_params_label = "clf_params"
model_name_label = "plot_label"
contig_file_label = "contig_file"

base_lvl_name = "Baseline"
pep_lvl_name = "Peptide level"
cont_lvl_name = "Contig level"
scaff_lvl_name = "Scaffold level"

scaffold_col = "scaffold"
exclude_mutation_col = "exclude_mutation"

prot_change_col = "Protein_Change"
gene_count_col = "Gene_count"

list_separator = ";"

int_cols = [gene_count_col, "Unique_peptides_narrow", "unique_peptides"]
int_cols_single = ["Start_position", "End_position", exclude_mutation_col]

add_info_cols = [
    "Chromosome",
    "Start_position",
    "End_position",
    "Variant_Classification",
    "Protein_Change",
    "gene_name",
]

results_dir = Path("Results")
results_dir.mkdir(parents=True, exist_ok=True)


def svm_train_test(train_data, train_y, test_data, clf_kwargs={}):
    sm = SMOTE(random_state=432)
    train_data, train_y = sm.fit_resample(train_data, train_y)
    clf = svm.SVC(probability=True, random_state=yeloh_seed, **clf_kwargs)
    clf = clf.fit(train_data, train_y)

    predictions = clf.predict(test_data)
    probs = clf.predict_proba(test_data)
    probSV = [i[1] for i in probs]
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
    evaluations["balanced_accuracy"] = balanced_accuracy_score(ytest, preds)
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

    if "exclude_mutation" in config and excluded_mutation is not None:
        results_frame = results_frame.rename(columns={"Count": gene_count_col})

    return results_frame, mut_num


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("You must specify index of a model")
    m_idx = int(sys.argv[1])

    try:
        config = copy.deepcopy(cfg.configurations[m_idx])
    except IndexError:
        raise IndexError(f"Theres no model at index {m_idx}")

    model_name = config[model_name_label]

    if model_name in (pep_lvl_name, base_lvl_name):
        deletion_type = "mutations"
    elif model_name == cont_lvl_name:
        deletion_type = "contigs"
    elif model_name == scaff_lvl_name:
        deletion_type = "scaffolds"
    else:
        deletion_type = None

    muts = pd.read_csv(config[contig_file_label], sep="\t", low_memory=False)
    # Filter the file like in 'DataSets_validation.py' first:
    if config["hotspots"]:
        muts = muts.loc[muts["contig"].notna()]
    unique_base_cols = ["Chromosome", "Start_position", "End_position"]
    if model_name == base_lvl_name:
        unique_cols = unique_base_cols
    elif model_name == pep_lvl_name:
        unique_cols = unique_base_cols + [
            "Unique_peptides_narrow",
            "Promiscuity_narrow",
        ]
    elif model_name in (cont_lvl_name, scaff_lvl_name):
        unique_cols = unique_base_cols + ["unique_peptides", "popcov_but_sqrt"]
    else:
        unique_cols = ["nope"]
    muts = muts.dropna(
        axis="index", how="any", subset=unique_cols, ignore_index=True
    ).drop_duplicates(unique_cols, ignore_index=True)
    #

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
        g = muts.groupby(by=squeeze_cols, as_index=False)
        g_1 = g[add_info_cols].agg(
            lambda x: list_separator.join(
                ["" if pd.isna(v) or pd.isnull(v) else str(v) for v in x]
            )
        )
        g_2 = g.size().rename(columns={"size": gene_count_col})
        muts_g = g_1.merge(
            g_2[[del_col, gene_count_col]], on=del_col, how="left", validate="1:1"
        ).sort_values(gene_count_col, ascending=False, ignore_index=True)
        cs_ids = muts_g[del_col]
        if len(cs_ids) != len(cs_ids.unique()):
            raise ValueError("There's a problem with IDs")
        iterator = cs_ids
    else:
        iterator = None

    for stuff_to_remove in tqdm(iterator, desc=model_name):
        config["exclude_mutation"] = stuff_to_remove
        results_frame, _ = svm_experiment(config, cut_input_params, deletion_type)
        results_frame.insert(0, exclude_mutation_col, stuff_to_remove)
        if model_name in (cont_lvl_name, scaff_lvl_name):
            cs_info = muts_g.loc[muts_g[del_col] == stuff_to_remove].reset_index(
                drop=True
            )
            results_frame = pd.concat([results_frame, cs_info], axis=1)
        overall_results.append(results_frame)

    overall_results = pd.concat(overall_results, ignore_index=True)

    for c in int_cols:
        if c in overall_results.columns:
            overall_results[c] = overall_results[c].astype("Int64")
    if model_name in (pep_lvl_name, base_lvl_name):
        for c in int_cols_single:
            if c in overall_results.columns:
                overall_results[c] = overall_results[c].astype("Int64")

    # Adding other mutation info to baseline/peptide
    if model_name in (pep_lvl_name, base_lvl_name):
        add_info_single_cols = list(set(add_info_cols) - set(unique_cols))
        add_info = []
        add_info.append(
            pd.DataFrame(
                [[pd.NA] * len(add_info_single_cols)], columns=add_info_single_cols
            )
        )
        for i in tqdm(range(1, len(overall_results)), desc="Adding additional info"):
            row = overall_results.iloc[i]
            row_q = []
            for c in unique_cols:
                if overall_results[c].dtype == "object":
                    s = f"{c} == '{row[c]}'"
                else:
                    s = f"{c} == {row[c]}"
                row_q.append(s)
            row_q = " & ".join(row_q)
            m = muts.query(row_q)
            a_info = m[add_info_single_cols].reset_index(drop=True)
            add_info.append(a_info)
        add_info = pd.concat(add_info, ignore_index=True)
        overall_results = pd.concat([overall_results, add_info], axis=1)
    #

    overall_results.to_csv(
        results_dir / f"mutation-remove-experiments-{model_name.replace(' ', '_')}.csv",
        index=False,
    )
