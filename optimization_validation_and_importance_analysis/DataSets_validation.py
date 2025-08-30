import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=UserWarning)


def mutation_vector_base_per_patient(contig_values_to_experiment_with, dimension_of_embedding_vectors, patients, hotspots, random_contigs, contig_file=False, to_remove=None, deletion_type=None):
    if contig_file:
        data = pd.read_csv(contig_file, sep="\t", low_memory=False)
    else:
        raise ValueError("You can't leave this empty, gimme a pep/con/scaff file")
    if random_contigs:
        raise ValueError("You don't want to do this rn")

    removed = None
    out_shape = None

    if type(to_remove) is str:
        if deletion_type == "contigs":
            del_col = "contig"
        elif deletion_type == "scaffolds":
            del_col = "Id"
        try:
            data = data.loc[data[del_col] != to_remove]
        except KeyError:
            raise KeyError("Did you set a correct value for `deletion_type`?")

    if hotspots:
        data = data.loc[data["contig"].notna()]

    variation_keys = ["Chromosome", "Start_position", "End_position"]
    out = data[variation_keys + contig_values_to_experiment_with].value_counts()

    # TAKE ALL MUTATIONS
    # out = out.loc[out > 3]
    out = out.loc[out > 0]
    # TAKE ALL MUTATIONS

    out = out.reset_index(name="Count")

    if type(to_remove) is int:
        # OMG
        out_shape = out.shape[0]
        try:
            removed = out.iloc[[to_remove]]
        except IndexError:
            return None, None, out_shape
        # OMG
        out = out.drop([to_remove])

    np.random.seed(100)
    embedding_vector_matrices = {}
    if len(contig_values_to_experiment_with) == 0:
        contig_values_to_experiment_with = ["NO_VALUE"]
    for contig_value_to_experiment_with in contig_values_to_experiment_with:
        embedding_vector_matrix = np.empty((len(out), dimension_of_embedding_vectors))
        for i in range(len(out)):
            embedding_vector_matrix[i] = np.random.normal(0, 1. / np.sqrt(dimension_of_embedding_vectors),
                                                          dimension_of_embedding_vectors)
        embedding_vectors = pd.DataFrame(embedding_vector_matrix)
        embedding_vector_matrices[contig_value_to_experiment_with] = embedding_vectors

    zero_matrix = np.zeros((len(out), dimension_of_embedding_vectors))
    embedding_vectors = pd.DataFrame(zero_matrix)
    mut_wh_emb_vs = pd.concat([out, embedding_vectors], axis=1)
    mut_wh_emb_vs = mut_wh_emb_vs.drop('Count', axis=1)


    # INSERTING THE VALUE contig_value_to_experiment_with
    if contig_values_to_experiment_with == ["NO_VALUE"]:
        mut_wh_emb_vs[embedding_vectors.columns] = embedding_vector_matrices["NO_VALUE"][embedding_vectors.columns]
    else:
        scaler = MinMaxScaler()
        mut_wh_emb_vs[contig_values_to_experiment_with] = scaler.fit_transform(mut_wh_emb_vs[contig_values_to_experiment_with]) + 1
        #print("MinMax Scaler done of the HS features")
        for cv in contig_values_to_experiment_with:
            mut_wh_emb_vs[embedding_vectors.columns] = mut_wh_emb_vs[embedding_vectors.columns] + embedding_vector_matrices[cv][embedding_vectors.columns].multiply(np.sqrt(np.sqrt(mut_wh_emb_vs[cv])), axis=0)

    augmented_data = pd.merge(data, mut_wh_emb_vs, on = variation_keys)
    # #print(embedding_vectors.columns)
    augmented_data = augmented_data[["SUBJID"] + [x for x in embedding_vectors.columns]]
    augmented_data = augmented_data.groupby(["SUBJID"]).sum()

    norm = Normalizer()
    augmented_data[embedding_vectors.columns] = norm.fit_transform(augmented_data[embedding_vectors.columns])

    augmented_data[embedding_vectors.columns] = len(contig_values_to_experiment_with) * augmented_data[embedding_vectors.columns]
    augmented_data = pd.merge(patients, augmented_data, on=["SUBJID"])

    new_data = pd.merge(patients, augmented_data, on=["SUBJID"], how="outer")
    new_out = new_data.loc[new_data[0].isnull()]
    new_out = new_out["SUBJID"].to_frame().reset_index()
    missing_patient_matrix = np.empty((len(new_out), dimension_of_embedding_vectors))
    for i in range(len(new_out)):
        missing_patient_matrix[i] = np.random.normal(0, 1. / np.sqrt(dimension_of_embedding_vectors),
                                                     dimension_of_embedding_vectors)

    missing_patient_matrix_df = pd.DataFrame(missing_patient_matrix)
    missing_patient_df = pd.concat([new_out, missing_patient_matrix_df], axis=1)
    missing_patient_df = missing_patient_df.drop("index", axis=1)

    augmented_data = pd.concat([augmented_data, missing_patient_df], axis=0)


    # Trick to avoid minmax normalization of prepare_training
    replac = {i: "MT_" + str(i) + "###" for i in embedding_vectors.columns}

    augmented_data.rename(columns=replac, inplace=True)

    return augmented_data, removed, out_shape


def param_check(params, name):
    if name in params:
        return params[name]
    else:
        return False


def transforming_Braun_dataset(params, dimension_of_embedding_vectors=4000, cut_input_params=False, deletion_type=None):
    if os.path.exists('../data/Braun_2020_ALL_UNIQUE_final_reduced.csv') and not param_check(params, "recompute"):
        ##print("Loading reduced dataset")
        new_data = pd.read_csv('../data/Braun_2020_ALL_UNIQUE_final_reduced.csv')
        ##print("...done")
    else:
        raise NotImplementedError("Something's wrong, check if you have the data actually synced?")

    # Dropping useless features
    new_data = new_data.drop(['ST_CD8_IF_ID', 'ST_MAF_Normal_ID', 'ST_MAF_Tumor_ID', 'ST_RNA_ID', 'ST_CNV_ID'], axis=1)
    new_data = new_data.drop(['Number_of_Prior_Therapies'], axis=1)

    # if param_check(params, "damien_split"):
    #     damien_split = pd.read_csv('data/damien_split.csv')
    #     new_data = new_data.drop(['TrainTestStatus'], axis=1)
    #     new_data = pd.merge(new_data,damien_split, on=["SUBJID"])



    # SETTING TARGET
    new_data["Outcome"] = new_data["PFS"].map(lambda x: 1 if x > 6 else -1)

    new_data["TRAIN"] = new_data["TrainTestStatus"].map(lambda x: "TRAIN" if x == "Train" else "TEST")
    new_data = new_data.drop(['TrainTestStatus'], axis=1)

    if param_check(params, "PRIMARY_TUMOR_ONLY"):
        new_data = new_data.loc[new_data["TF_Tumor_Sample_Primary_or_Metastasis"] == "PRIMARY"]


    new_data = new_data.drop([
        #"TF_Tumor_Sample_Primary_or_Metastasis",
        "TF_Site_of_Metastasis",
        "TF_ImmunoPhenotype",
        "TF_Days_from_TumorSample_Collection_and_Start_of_Trial_Therapy"], axis=1)

    mut_num = None
    if param_check(params, "with_mutations"):
        patients = new_data["SUBJID"].drop_duplicates()
        mutations, removed, mut_num = mutation_vector_base_per_patient(param_check(params, "HS_features"), dimension_of_embedding_vectors, patients, param_check(params, "hotspots"),param_check(params, "random_contigs"), contig_file = param_check(params, "contig_file"),to_remove = param_check(params, "exclude_mutation"), deletion_type=deletion_type)
        # OMG
        if mutations is None:
            return None, None, None, None, None, None, mut_num
        # OMG
        new_data = pd.merge(new_data, mutations, on=["SUBJID"], how="outer")
        # out = new_data.loc[new_data["0###"].isnull()]
    new_data = new_data.drop("SUBJID", axis=1)

    #new_data = new_data.drop([x for x in new_data.columns if x.startswith("TF_")], axis=1)

    if "Unnamed: 0" in new_data:
        new_data = new_data.drop("Unnamed: 0", axis=1)

    new_data.rename(columns={"Received_Prior_Therapy":"CF_Received_Prior_Therapy",
                             "Sex":"CF_Sex",
                             "Age":"CF_Age"
                             }, inplace=True)

    columns_to_remove = [x for x in new_data.columns if not x.startswith("BP_") and not x.startswith("GE_") and not x.startswith("MT_")]
    columns_to_remove.remove("TRAIN")
    columns_to_remove.remove("Outcome")
    columns_to_remove = ['PS_MSKCC', 'PS_IMDC', 'TF_Tumor_Sample_Primary_or_Metastasis', 'CF_Received_Prior_Therapy', 'CF_Sex']
    columns_to_remove = columns_to_remove + ['Arm']
    columns_to_discretize = ['TF_Purity', 'TF_Ploidy', 'TF_TMB_Counts', 'TF_TM_Area', 'TF_TM_CD8', 'TF_TM_CD8_Density',
                             'TF_TC_Area', 'TF_TC_CD8', 'TF_TC_CD8_Density', 'TF_TM_TC_Ratio', 'TF_TM_CD8_PERCENT',
                             'TF_TC_CD8_PERCENT', 'CF_Age']


    discretizer = KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')


    new_data[columns_to_discretize] = discretizer.fit_transform(new_data[columns_to_discretize])

    # >>>>>>>>>>>>>>>>>>>>>>>> VERY STUPID : DROPPING ARM
    #new_data = new_data.drop("Arm", axis=1)
    #### REMEBER TO PUT BACK 'Arm', in columns_to_remove
    # >>>>>>>>>>>>>>>>>>>>>>>> VERY STUPID : DROPPING ARM


    #data_for_revert_exp = new_data[['TRAIN','Outcome','Arm']]
    new_data = pd.get_dummies(new_data, prefix_sep="###", dtype=float, columns=columns_to_remove+columns_to_discretize)

    new_data_train = new_data.loc[new_data["TRAIN"] == "TRAIN"]
    #FOOL: Training on the whole dataset
    ####    new_data_train = new_data
    #FOOL: Training on the whole dataset

    train_classification_labels = new_data_train['Outcome']
    new_data_train = new_data_train.drop(["TRAIN", "Outcome"], axis=1)
    # Dropping target related features
    new_data_train = new_data_train.drop(['Benefit', 'ORR', 'PFS_CNSR', 'OS', 'OS_CNSR','PFS'], axis=1)

    new_data_test = new_data.loc[new_data["TRAIN"] == "TEST"]
    test_classification_labels = new_data_test['Outcome']
    new_data_test = new_data_test.drop(["TRAIN", "Outcome"], axis=1)
    test_pfs = new_data_test['PFS']
    # Dropping target related features
    new_data_test = new_data_test.drop(['Benefit', 'ORR', 'PFS_CNSR', 'OS', 'OS_CNSR','PFS'], axis=1)

    #dim = 100
    ##print("Reducing Mutations to " + str(dim) + " dimensions")
    #new_data_train, new_data_test = reducing_training_and_testing(new_data_train, new_data_test, "MT_", dim)
    ##print("... done")

    TF_columns = [x for x in new_data.columns if x.startswith("TF_")]
    # MIN MAX Scaler
    scaler = MinMaxScaler()
    new_data_train[TF_columns] = scaler.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = scaler.transform(new_data_test[TF_columns])
    #print("MinMax Scaler done")

    norm = Normalizer()
    new_data_train[TF_columns] = norm.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = norm.fit_transform(new_data_test[TF_columns])
    #print("Normalizer done")


    TF_columns = [x for x in new_data.columns if x.startswith("CF_")]
    # MIN MAX Scaler
    scaler = MinMaxScaler()
    new_data_train[TF_columns] = scaler.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = scaler.transform(new_data_test[TF_columns])
    #print("MinMax Scaler done")

    # COLUMNS FILTERING
    if cut_input_params:
        ok_columns = ["CF_Sex", "CF_Age", "MT_"]
        new_data_train = new_data_train[[c for c in new_data_train.columns if c.startswith(tuple(ok_columns))]]
        new_data_test = new_data_test[[c for c in new_data_test.columns if c.startswith(tuple(ok_columns))]]
    # COLUMNS FILTERING

    # Relative weights among groups of features

    if "weights" in params:
        weights = params["weights"]
        TF_columns = [x for x in new_data_train.columns if x.startswith("TF_")]
        GE_columns = [x for x in new_data_train.columns if x.startswith("GE_")]
        BP_columns = [x for x in new_data_train.columns if x.startswith("BP_")]
        MT_columns = [x for x in new_data_train.columns if x.startswith("MT_")]
        PS_columns = [x for x in new_data_train.columns if x.startswith("PS_")]
        CF_columns = [x for x in new_data_train.columns if x.startswith("CF_")]
        # ARM WEIGHTS
        Arm_columns = [x for x in new_data_train.columns if x.startswith("Arm")]
        # ARM WEIGHTS
        if TF_columns:
            new_data_train[TF_columns] = weights["TF"]*scaler.fit_transform(new_data_train[TF_columns])
            new_data_test[TF_columns] = weights["TF"]*scaler.transform(new_data_test[TF_columns])
        if GE_columns:
            new_data_train[GE_columns] = weights["GE"]*scaler.fit_transform(new_data_train[GE_columns])
            new_data_test[GE_columns] = weights["GE"]*scaler.transform(new_data_test[GE_columns])
        if BP_columns:
            new_data_train[BP_columns] = weights["BP"]*scaler.fit_transform(new_data_train[BP_columns])
            new_data_test[BP_columns] = weights["BP"]*scaler.transform(new_data_test[BP_columns])
        # ARM WEIGHTS
        if Arm_columns:
            new_data_train[Arm_columns] = weights["Arm"]*scaler.fit_transform(new_data_train[Arm_columns])
            new_data_test[Arm_columns] = weights["Arm"]*scaler.transform(new_data_test[Arm_columns])
        # ARM WEIGHTS
        if MT_columns:
            new_data_train[MT_columns] = weights["MT"]*scaler.fit_transform(new_data_train[MT_columns])  # \/
            new_data_test[MT_columns] = weights["MT"]*scaler.transform(new_data_test[MT_columns])
        if PS_columns:
            new_data_train[PS_columns] = weights["PS"]*scaler.fit_transform(new_data_train[PS_columns])
            new_data_test[PS_columns] = weights["PS"]*scaler.transform(new_data_test[PS_columns])
        if CF_columns:
            new_data_train[CF_columns] = weights["CF"]*scaler.fit_transform(new_data_train[CF_columns])
            new_data_test[CF_columns] = weights["CF"]*scaler.transform(new_data_test[CF_columns])

    # COLUMNS FILTERING
    # if cut_input_params:
    #     ok_columns = ["CF_Sex", "CF_Age", "MT_"]
    #     new_data_train = new_data_train[[c for c in new_data_train.columns if c.startswith(tuple(ok_columns))]]
    #     new_data_test = new_data_test[[c for c in new_data_test.columns if c.startswith(tuple(ok_columns))]]
    # COLUMNS FILTERING

    norm = Normalizer()
    new_data_train[new_data_train.columns] = norm.fit_transform(new_data_train[new_data_train.columns])
    new_data_test[new_data_test.columns] = norm.fit_transform(new_data_test[new_data_test.columns])
    #print("Normalizer done")

    return new_data_train, train_classification_labels, new_data_test, test_classification_labels, test_pfs, removed if "removed" in locals() else None, mut_num
