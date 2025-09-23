"""."""

import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from DataSets import reducing_training_and_testing  # noqa: E402

# Config

# Set `True` if you want to use the updated input data schema
new_full_data = False

og_braun_data_file = "/data/teamgdansk/Braun_2020_ALL_UNIQUE_final.csv"
reduced_braun_data_file = (
    "../data/Braun_2020_ALL_UNIQUE_final_reduced_new_traintest.csv"
)

# Set to path if you want to switch `test` with `validation` data to process
# validation_data_file = "../data/Braun_2020_ALL_UNIQUE_final_reduced_new_validation.csv"
validation_data_file = None

# Set `True` if you want PCA to train on both `train` and `test` subsets
reduce_train_on_all = False

#

og_braun_data = pd.read_csv(
    og_braun_data_file, dtype={"TF_Site_of_Metastasis": "object"}
)

if new_full_data:
    # ok_cols = og_braun_data.columns[:-50]
    # og_braun_data = og_braun_data[ok_cols]
    raise NotImplementedError("not yet")

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
    # new_data_test = validation data
    raise NotImplementedError("not yet")

dim = 100
print("Reducing Biological Pathways to " + str(dim) + " dimensions")
new_data_train, new_data_test = reducing_training_and_testing(
    new_data_train, new_data_test, "BP_", dim, train_on_all=reduce_train_on_all
)
print("...done")
dim = 200
print("Reducing Gene Expression to " + str(dim) + " dimensions")
new_data_train, new_data_test = reducing_training_and_testing(
    new_data_train, new_data_test, "GE_", dim, train_on_all=reduce_train_on_all
)
print("...done")

if validation_data_file is None:
    new_data = pd.concat([new_data_train, new_data_test], ignore_index=True)
    # new_data.to_csv(reduced_braun_data_file, index=False)
    print("\nTURN ON FILE SAVING\n")
else:
    # new_data_test.to_csv(validation_data_file, index=False)
    raise NotImplementedError("not yet")

# TESTING

print()

ge_cols = [c for c in new_data.columns if c.startswith("GE_")]
bp_cols = [c for c in new_data.columns if c.startswith("BP_")]
rest_cols = [c for c in new_data.columns if c not in ge_cols and c not in bp_cols]

df = pd.read_csv("../data/Braun_2020_ALL_UNIQUE_final_reduced.csv")
df = df.drop(["Unnamed: 0"], axis=1)

print(all(new_data.columns == df.columns))
print()

round_n = 8

print(df[rest_cols].equals(new_data[rest_cols]))
print()

for cols in (ge_cols, bp_cols):
    print(df[cols].equals(new_data[cols]))
    print(df[cols].round(round_n).equals(new_data[cols].round(round_n)))
    print(df[cols].abs().equals(new_data[cols].abs()))
    print(df[cols].abs().round(round_n).equals(new_data[cols].abs().round(round_n)))
    print((df[cols] * -1).round(round_n).equals(new_data[cols].round(round_n)))

    print(
        (df[cols].round(round_n) == new_data[cols].round(round_n)).all().sum()
        / new_data[cols].shape[1]
        * 100
    )

    print(
        ((df[cols] * -1).round(round_n) == new_data[cols].round(round_n)).all().sum()
        / new_data[cols].shape[1]
        * 100
    )

    print(
        (df[cols].round(round_n) == new_data[cols].round(round_n)).any().sum()
        / new_data[cols].shape[1]
        * 100
    )

    print()
