"""."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

# Config

results_dir = Path("Results")

# Set `True` if you want to use the updated input data schema
new_full_data = True

if new_full_data:
    version_suffix = "new"
else:
    version_suffix = "old"

og_braun_data_file = "/data/teamgdansk/Braun_2020_ALL_UNIQUE_final.csv"
reduced_braun_data_file = (
    f"../data/Braun_2020_ALL_UNIQUE_final_reduced_{version_suffix}_traintest.csv"
)

new_full_data_cols_file = "../data/Common_columns_Braun_to_KATY.csv"

# Set to path if you want to switch `test` with `validation` data to process
validation_data_file = "/data/teamgdansk/asd.csv"
# validation_data_file = None
reduced_validation_data_file = (
    f"../data/Braun_2020_ALL_UNIQUE_final_reduced_{version_suffix}_validation.csv"
)

# PCA options
retrain_pca = False
pca_model_bp_file = results_dir / f"pca-model-bp-{version_suffix}.pkl"
pca_model_ge_file = results_dir / f"pca-model-ge-{version_suffix}.pkl"
# Set `True` if you want PCA to train on both `train` and `test` subsets
reduce_train_on_all = False

#

og_braun_data = pd.read_csv(
    og_braun_data_file, dtype={"TF_Site_of_Metastasis": "object"}
)

if validation_data_file is not None:
    validation_data = pd.read_csv(validation_data_file)
    #! Dropping NaNs (?)
    validation_data = validation_data.dropna()

if new_full_data:
    ok_cols = pd.read_csv(new_full_data_cols_file).iloc[:, 0].to_list()
    og_braun_data = og_braun_data[ok_cols]
    if validation_data_file is not None:
        validation_data = validation_data[ok_cols]

# FILLING nan with the Mode
filler = og_braun_data.mode(axis=0, dropna=True).loc[0].fillna(0)
og_braun_data = og_braun_data.fillna(filler)

# REDUCING `BP_` AND `GE_`
new_data_train = og_braun_data.loc[
    og_braun_data[og_braun_data["TrainTestStatus"] == "Train"].index
]
if validation_data_file is None:
    new_data_test = og_braun_data.loc[
        og_braun_data[og_braun_data["TrainTestStatus"] == "Test"].index
    ]
else:
    new_data_test = validation_data


def reducing_training_and_testing(
    new_data_train,
    new_data_test,
    kind,
    dim,
    pca_model_file,
    train_on_all=False,
    retrain_pca=False,
):
    BP_columns = [x for x in new_data_train.columns if x.startswith(kind)]

    if train_on_all:
        train_data_for_reducer = pd.concat(
            [new_data_train[BP_columns], new_data_test[BP_columns]]
        )
    else:
        train_data_for_reducer = new_data_train[BP_columns]

    if BP_columns:
        if Path(pca_model_file).exists() and not retrain_pca:
            with open(pca_model_file, "rb") as f:
                model_pca_BP = pickle.load(f)
        else:
            pca = PCA(n_components=dim, svd_solver="full")
            model_pca_BP = pca.fit(train_data_for_reducer)
            with open(pca_model_file, "wb") as f:
                pickle.dump(model_pca_BP, f)

        # APPLYING THE REDUCER ON THE TRAINING SET
        bp_red = model_pca_BP.transform(new_data_train[BP_columns])
        df_bp_red = pd.DataFrame(
            bp_red, columns=[kind + "_" + str(i) for i in range(0, dim)]
        )
        new_data_train = new_data_train.reset_index(drop=True)
        new_data_train = pd.concat([new_data_train, df_bp_red], axis=1)
        new_data_train = new_data_train.drop(BP_columns, axis=1)

        # APPLYING THE REDUCER ON THE TESTING SET
        bp_red = model_pca_BP.transform(new_data_test[BP_columns])
        df_bp_red = pd.DataFrame(
            bp_red, columns=[kind + "_" + str(i) for i in range(0, dim)]
        )
        new_data_test = new_data_test.reset_index(drop=True)
        new_data_test = pd.concat([new_data_test, df_bp_red], axis=1)
        new_data_test = new_data_test.drop(BP_columns, axis=1)

        # NORMALIZER
        columns_to_normalize = [kind + "_" + str(i) for i in range(0, dim)]
        norm = Normalizer()
        new_data_train[columns_to_normalize] = norm.fit_transform(
            new_data_train[columns_to_normalize]
        )
        new_data_test[columns_to_normalize] = norm.fit_transform(
            new_data_test[columns_to_normalize]
        )

    return new_data_train, new_data_test


dim = 100
print("Reducing Biological Pathways to " + str(dim) + " dimensions")
new_data_train, new_data_test = reducing_training_and_testing(
    new_data_train,
    new_data_test,
    "BP_",
    dim,
    pca_model_file=pca_model_bp_file,
    train_on_all=reduce_train_on_all,
    retrain_pca=retrain_pca,
)
print("...done")
dim = 200
print("Reducing Gene Expression to " + str(dim) + " dimensions")
new_data_train, new_data_test = reducing_training_and_testing(
    new_data_train,
    new_data_test,
    "GE_",
    dim,
    pca_model_file=pca_model_ge_file,
    train_on_all=reduce_train_on_all,
    retrain_pca=retrain_pca,
)
print("...done")

if validation_data_file is None:
    new_data = pd.concat([new_data_train, new_data_test], ignore_index=True)
    new_data.to_csv(reduced_braun_data_file, index=False)
else:
    new_data_test.to_csv(reduced_validation_data_file, index=False)
