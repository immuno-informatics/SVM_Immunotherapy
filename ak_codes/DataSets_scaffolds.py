import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=UserWarning)

import os


def mutation_vector_base_per_patient(scaffold_values_to_experiment_with, dimension_of_embedding_vectors, patients, hotspots, random_scaffolds, scaffold_file=False):
    if scaffold_file:
        data = pd.read_csv(scaffold_file, sep="\t", low_memory=False)
        #data = data[(data["gap_width"] == 10) & (data["Multi_scaffold"] == "Y")] 
    else:
        #data = pd.read_csv("data/Braun_mutations_hg38_with_epitope_scaffolds.tsv", sep="\t", low_memory=False)
        #data = pd.read_csv("data/scaffold_features_reduced2.tsv", sep="\t", low_memory=False)
        data = pd.read_csv("data/Braun_hg38_scaff_gp10.tsv",sep = "\t",low_memory = False)  
    #random_scaffolds = True
    if random_scaffolds:
        ratio = len(data.loc[data["scaffold"].notna()])/len(data)*100
        np.random.seed(800)
        data["scaffold_1"] = [np.random.randint(0,100) for _ in range(len(data))]
        data["scaffold"] = data["scaffold_1"].map(lambda x: "YES" if x<ratio else np.nan)
        new_ratio = len(data.loc[data["scaffold"].notna()])/len(data)*100
        print("Real Contig Ratio  : " + str(ratio) + "\n" +
              "Random Contig Ratio: " + str(new_ratio))

    if hotspots:
        data = data.loc[data["scaffold"].notna()]

    variation_keys = ["Chromosome", "Start_position", "End_position"]
    #scaffold_values_to_experiment_with = ["unique_peptides","spec_count"]
    #scaffold_values_to_experiment_with = ["unique_peptides"]
    #scaffold_values_to_experiment_with = ["spec_count"]
    out = data[variation_keys + scaffold_values_to_experiment_with].value_counts()
    out = out.loc[out > 3] 
    # out = out.to_dict()
    # mutations_ = out.keys()

    #scaler = MinMaxScaler()
    #out[scaffold_values_to_experiment_with] = scaler.fit_transform(out[scaffold_values_to_experiment_with])
    #print("MinMax Scaler done")

    out = out.reset_index(name='Count')
    np.random.seed(100)
    embedding_vector_matrices = {}
    if len(scaffold_values_to_experiment_with) == 0:
        scaffold_values_to_experiment_with = ["NO_VALUE"]
    for scaffold_value_to_experiment_with in scaffold_values_to_experiment_with:
        embedding_vector_matrix = np.empty((len(out), dimension_of_embedding_vectors))
        for i in range(len(out)):
            embedding_vector_matrix[i] = np.random.normal(0, 1. / np.sqrt(dimension_of_embedding_vectors),
                                                          dimension_of_embedding_vectors)

        embedding_vectors = pd.DataFrame(embedding_vector_matrix)
        embedding_vector_matrices[scaffold_value_to_experiment_with] = embedding_vectors

    zero_matrix = np.zeros((len(out), dimension_of_embedding_vectors))
    embedding_vectors = pd.DataFrame(zero_matrix)
    mut_wh_emb_vs = pd.concat([out, embedding_vectors], axis=1)
    mut_wh_emb_vs = mut_wh_emb_vs.drop('Count', axis=1)


    # INSERTING THE VALUE scaffold_value_to_experiment_with
    if scaffold_values_to_experiment_with == ["NO_VALUE"]:
        mut_wh_emb_vs[embedding_vectors.columns] = embedding_vector_matrices["NO_VALUE"][embedding_vectors.columns]
    else:
        scaler = MinMaxScaler()
        mut_wh_emb_vs[scaffold_values_to_experiment_with] = scaler.fit_transform(mut_wh_emb_vs[scaffold_values_to_experiment_with]) + 1
        print("MinMax Scaler done of the HS features")
        for cv in scaffold_values_to_experiment_with:
            mut_wh_emb_vs[embedding_vectors.columns] = mut_wh_emb_vs[embedding_vectors.columns] + embedding_vector_matrices[cv][embedding_vectors.columns].multiply(np.sqrt(np.sqrt(mut_wh_emb_vs[cv])), axis=0)
            #mut_wh_emb_vs[embedding_vectors.columns] = mut_wh_emb_vs[embedding_vectors.columns] + embedding_vector_matrices[cv][embedding_vectors.columns]

#    augmented_data = pd.merge(data, mut_wh_emb_vs, on = variation_keys + scaffold_values_to_experiment_with)
    augmented_data = pd.merge(data, mut_wh_emb_vs, on = variation_keys)
    augmented_data = augmented_data[["SUBJID"] + [x for x in embedding_vectors.columns]]
    augmented_data = augmented_data.groupby(["SUBJID"]).sum()

    norm = Normalizer()
    augmented_data[embedding_vectors.columns] = norm.fit_transform(augmented_data[embedding_vectors.columns])

    #augmented_data[embedding_vectors.columns] = len(scaffold_values_to_experiment_with) * augmented_data[embedding_vectors.columns]
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
    # missing_patient_df = missing_patient_df.set_index("SUBJID")

    augmented_data = pd.concat([augmented_data, missing_patient_df], axis=0) 
     

    # augmented_data.drop("index",axis=1)

    # Trick to avoid minmax normalization of prepare_training
    replac = {i: "MT_" + str(i) + "###" for i in embedding_vectors.columns}

    augmented_data.rename(columns=replac, inplace=True)

    return augmented_data

bp = ["BP_chr5q23", "BP_chr16q24", "BP_chr8q24", "BP_chr13q11", "BP_chr7p21", "BP_chr10q23",
      "BP_chr13q13", "BP_chr10q21", "BP_chr1p13", "BP_chrxp21", "BP_chr4q12", "BP_chr6q13", "BP_chr2p22",
      "BP_chr18q23", "BP_chr8p11", "BP_chr1p11", "BP_chr9p12", "BP_chr20p12", "BP_chr7p13", "BP_chr4p16",
      "BP_chr8q22", "BP_chr14q11", "BP_chryp11", "BP_chr10q25", "BP_chr11q", "BP_chr4p14", "BP_chr15q26",
      "BP_chr12q15", "BP_chr4p12", "BP_chr9q12", "BP_chr2q22", "BP_chr9q33", "BP_chr18p", "BP_chr3q28",
      "BP_chr12q14", "BP_chr2q36", "BP_chr9q22", "BP_chr3p22", "BP_chrxq", "BP_chr6p11", "BP_chr12q24",
      "BP_chr2q32", "BP_chr4q35", "BP_chr3p14", "BP_chr22q", "BP_chr15q24", "BP_chr2p", "BP_chr13q31",
      "BP_chr19p13", "BP_chr12q22", "BP_chr7p", "BP_chr11q22", "BP_chrxq28", "BP_chr16q22",
      "BP_chr11q24", "BP_chr5q12", "BP_chr10p11", "BP_chr5q21", "BP_chr14q21", "BP_chrxq12",
      "BP_chr1p35", "BP_chr3p12", "BP_chr4q21", "BP_chr10p15", "BP_chr18q21", "BP_chr1p33",
      "BP_chr11q12", "BP_chr17q22", "BP_chr1p31", "BP_chr10p13", "BP_chr14q13", "BP_chr2q24",
      "BP_chr22q13", "BP_chr15q11", "BP_chr1q43", "BP_chr19p", "BP_chr3q24", "BP_chr22q11", "BP_chr7p11",
      "BP_chr16q13", "BP_chr1p22", "BP_chr12p", "BP_chr2q11", "BP_chr9p21", "BP_chr4q33", "BP_chr2p11",
      "BP_chr2q34", "BP_chr17q23", "BP_chr14q23", "BP_chr5q33", "BP_chr6q27", "BP_chr3q26",
      "BP_chr10q11", "BP_chr5p12", "BP_chr1q21", "BP_chr4q24", "BP_chr7q31", "BP_chr6p21", "BP_chr17q",
      "BP_chr5q15", "BP_chr2q13", "BP_chr1q23", "BP_chr6q23", "BP_chr11q14", "BP_chr5p14", "BP_chr5q31",
      "BP_chr2p24", "BP_chr9p11", "BP_chr4q26", "BP_chr1q25", "BP_chrxq11", "BP_chr5q14", "BP_chr4q28",
      "BP_chr1q42", "BP_chr6p25", "BP_chr13q34", "BP_chr8q12", "BP_chr4q22", "BP_chr7q22", "BP_chr6p23",
      "BP_chr8p21", "BP_chr5q35", "BP_chr13q32", "BP_chr12q12", "BP_chr3p24", "BP_chr7q33",
      "BP_chr12p12", "BP_chr9p23", "BP_chr7p14", "BP_chr6q21", "BP_chryq11", "BP_chr15q13", "BP_chr3q22",
      "BP_chr6q15", "BP_chr4q31", "BP_chr3q12", "BP_chr13q22", "BP_chr15q15", "BP_chr3p26",
      "BP_chr17q12", "BP_chr11p12", "BP_chr7p12", "BP_chr6q", "BP_chrxq21", "BP_chr19q13", "BP_chr16p12",
      "BP_chr1q32", "BP_chr8p23", "BP_chrxq23", "BP_chr2p15", "BP_chr18q12", "BP_chr17p13", "BP_chr14q",
      "BP_chr11p14", "BP_chr1q", "BP_chr9q32", "BP_chr20q12", "BP_chr14q32", "BP_chr7p22", "BP_chr5q",
      "BP_chr21q21", "BP_chr2p13", "BP_chr16q12", "BP_chr3p", "BP_chr10q22", "BP_chr17p11",
      "BP_chr15q22", "BP_chr10q", "BP_chr1q31", "BP_chr10q24", "BP_chr13q14", "BP_chr11p", "BP_chr14q12",
      "BP_chr4q", "BP_chr1p34", "BP_chr2q23", "BP_chr6q16", "BP_chr16q21", "BP_chr9q13", "BP_chr15q25",
      "BP_chr18q22", "BP_chr2q21", "BP_chr3p25", "BP_chr20p13", "BP_chr8q21", "BP_chr4p15",
      "BP_chr12p11", "BP_chr9p13", "BP_chr2q37", "BP_chr3q29", "BP_chr16p", "BP_chr13q12", "BP_chr3p21",
      "BP_chr1p12", "BP_chr4p11", "BP_chr10q26", "BP_chr13q21", "BP_chr15q23", "BP_chr2p21",
      "BP_chr19q12", "BP_chr20p11", "BP_chr5q22", "BP_chr4p13", "BP_chr8q23", "BP_chr19p12",
      "BP_chr21q22", "BP_chrxq25", "BP_chr4q34", "BP_chr11q23", "BP_chr10p14", "BP_chr9q34", "BP_chr15q",
      "BP_chr7q", "BP_chr2q33", "BP_chr8q", "BP_chr22q12", "BP_chr6p12", "BP_chr7q21", "BP_chr16q23",
      "BP_chr11q25", "BP_chr5q11", "BP_chr21q11", "BP_chr11q13", "BP_chr9p", "BP_chr17q25", "BP_chr6q25",
      "BP_chr10p12", "BP_chr7q36", "BP_chrxp11", "BP_chr9q21", "BP_chrxp22", "BP_chr1p32", "BP_chr12q21",
      "BP_chr8p12", "BP_chr11q11", "BP_chr12q23", "BP_chrxq13", "BP_chr1p36", "BP_chr2p23", "BP_chr2q31",
      "BP_chr3p13", "BP_chr2q", "BP_chrxq27", "BP_chr3p11", "BP_chr17q21", "BP_chr13q33", "BP_chr11p15",
      "BP_chr1p21", "BP_chr3q25", "BP_chr3q13", "BP_chr5p13", "BP_chr14q22", "BP_chr9p22", "BP_chr4q27",
      "BP_chr2p25", "BP_chr6p24", "BP_chr1p", "BP_chr2q12", "BP_chr8q13", "BP_chr2p16", "BP_chr12q",
      "BP_chr4q25", "BP_chr20q13", "BP_chr5q34", "BP_chr17q24", "BP_chr2q14", "BP_chr7q35", "BP_chr5q13",
      "BP_chr19q", "BP_chr5q32", "BP_chr20q11", "BP_chr6q24", "BP_chr5p15", "BP_chr1q22", "BP_chr6q26",
      "BP_chr6p22", "BP_chr11q21", "BP_chr1q41", "BP_chr15q14", "BP_chr8q11", "BP_chr1q24", "BP_chr4q23",
      "BP_chrxq26", "BP_chr3q23", "BP_chr3p23", "BP_chr12q13", "BP_chr9q31", "BP_chr2p12", "BP_chr1q44",
      "BP_chr7q34", "BP_chr2q35", "BP_chr6q14", "BP_chr16p13", "BP_chr18q11", "BP_chr3q27", "BP_chr7q11",
      "BP_chr3q21", "BP_chr16p11", "BP_chr12p13", "BP_chr7p15", "BP_chr3q11", "BP_chr6q22", "BP_chr7q32",
      "BP_chr12q11", "BP_chr18p11", "BP_chr17q11", "BP_chrxq24", "BP_chr6q12", "BP_chr3q", "BP_chr6p",
      "BP_chrxq22", "BP_chr9p24", "BP_chr11p11", "BP_chr4q13", "BP_chr16q11", "BP_chr8p22",
      "BP_chr14q24", "BP_chr1q12", "BP_chr15q21", "BP_chr11p13", "BP_chr17p12", "BP_chr4q32",
      "BP_chr4q11", "BP_chr14q31", "BP_chr2p14"]


def param_check(params, type):
    if type in params:
        return params[type]
    else:
        return False


def reading_katy_data_2(params):
    dataset = pd.read_csv('data/Braun_2020_ALL_UNIQUE_revert_62.csv')
    dataset = dataset.drop("Outcome",axis=1)
    to_drop = [x for x in dataset.columns if x.startswith("Arm###")]
    dataset = dataset.drop(to_drop,axis=1)
    return dataset


def reading_katy_data(params):
    with_mutations = False
    hotspot = False
    bio_path = False
    if "with_mutations" in params: with_mutations = params["with_mutations"]
    if "hotspots" in params: hotspots = params["hotspots"]
    if "bio_path" in params: bio_path = params["bio_path"]
    data = pd.read_csv('data/katy_pazienti_all_features_v2.csv')
    # "Benefit", "ORR",  "PFS_CNSR", "OS", "OS_CNSR", "Sample_Type",
    new_data = data[["Dataset", "MSKCC", "IMDC", "Sample_Type", "SUBJID", "PFS", "Arm",
                     "Tumor_Sample_Primary_or_Metastasis",
                     # "Received_Prior_Therapy", "Number_of_Prior_Therapies",
                     # "Sex", "Age",
                     ] + bp + [
                        "Days_from_TumorSample_Collection_and_Start_of_Trial_Therapy",
                        "Site_of_Metastasis", "ImmunoPhenotype", "Purity", "Ploidy",
                        "TMB_Counts", "TM_Area", "TM_CD8", "TM_CD8_Density", "TC_Area", "TC_CD8", "TC_CD8_Density",
                        "TM_TC_Ratio", "TM_CD8_PERCENT", "TC_CD8_PERCENT",
                        "TrainTestStatus"]].drop_duplicates()

    new_data = new_data.loc[new_data["Sample_Type"].str.startswith("RNA")].drop_duplicates()
    new_data["TARGET"] = new_data["PFS"].map({"Yes": "YES", "No": "NO"})
    new_data = new_data.drop(['PFS', 'Sample_Type'], axis=1)
    new_data["TRAIN"] = new_data["TrainTestStatus"].map(lambda x: "TRAIN" if x == "Train" else "TEST")
    new_data = new_data.drop(['TrainTestStatus'], axis=1)

    new_data = new_data.loc[new_data["Dataset"] == "BRAUN_2020"]
    new_data = new_data.loc[new_data["Tumor_Sample_Primary_or_Metastasis"] == "PRIMARY"]

    BP_columns = [x for x in new_data.columns if x.startswith("BP_")]
    if BP_columns:
        scaler = MinMaxScaler()
        new_data[BP_columns] = scaler.fit_transform(new_data[BP_columns])  # We noramlize them
        norm = Normalizer()
        new_data[BP_columns] = norm.fit_transform(new_data[BP_columns])

    # Trick to avoid minmax normalization of prepare_training
    replac = {}
    for i in BP_columns:
        replac[i] = str(i).replace("BP_", "GE_") + "###"
    new_data.rename(columns=replac, inplace=True)

    if with_mutations:
        patients = new_data["SUBJID"].drop_duplicates()
        mutations = mutation_vector_base_per_patient(400, patients, hotspots)

        new_data = pd.merge(new_data, mutations, on=["SUBJID"], how="outer")
        # out = new_data.loc[new_data["0###"].isnull()]

    if bio_path:
        bio_path_data = pd.read_excel('data/b_p_per_patient.xlsx')
        BP_columns = [x for x in bio_path_data.columns if x.startswith("BP_")]
        scaler = MinMaxScaler()
        bio_path_data[BP_columns] = scaler.fit_transform(bio_path_data[BP_columns])  # We noramlize them
        norm = Normalizer()
        bio_path_data[BP_columns] = norm.fit_transform(bio_path_data[BP_columns])
        replac = {}
        for i in BP_columns:
            replac[i] = str(i) + "###"
        bio_path_data.rename(columns=replac, inplace=True)

        new_data = pd.merge(new_data, bio_path_data, on=["SUBJID"], how="outer")

    new_data = new_data.drop("SUBJID", axis=1)
    return new_data


def reducing_training_and_testing(new_data_train, new_data_test, kind, dim, train_on_all=False):
    BP_columns = [x for x in new_data_train.columns if x.startswith(kind)]
    # MIN MAX Scaler
    #scaler = MinMaxScaler()
    #new_data_train[BP_columns] = scaler.fit_transform(new_data_train[BP_columns])
    #new_data_test[BP_columns] = scaler.transform(new_data_test[BP_columns])
    #print("MinMax Scaler done")

    #train_on_all = True
    if train_on_all:
        train_data_for_reducer = pd.concat([new_data_train[BP_columns],new_data_test[BP_columns]])
    else:
        train_data_for_reducer = new_data_train[BP_columns]
    # REDUCER
    # LEARNING THE REDUCER ON THE TRAINING SET
    if BP_columns:
        pca = PCA(n_components=dim, svd_solver='full')
        model_pca_BP = pca.fit(train_data_for_reducer)
        print("PCA reducer learnt")

        # APPLYING THE REDUCER ON THE TRAINING SET
        bp_red = model_pca_BP.transform(new_data_train[BP_columns])
        df_bp_red = pd.DataFrame(bp_red, columns=[kind + "_" + str(i) for i in range(0, dim)])
        new_data_train = new_data_train.reset_index(drop=True)
        new_data_train = pd.concat([new_data_train, df_bp_red], axis=1)
        new_data_train = new_data_train.drop(BP_columns, axis=1)
        print("PCA reducer applied to training")

        # APPLYING THE REDUCER ON THE TESTING SET
        bp_red = model_pca_BP.transform(new_data_test[BP_columns])
        df_bp_red = pd.DataFrame(bp_red, columns=[kind + "_" + str(i) for i in range(0, dim)])
        new_data_test = new_data_test.reset_index(drop=True)
        new_data_test = pd.concat([new_data_test, df_bp_red], axis=1)
        new_data_test = new_data_test.drop(BP_columns, axis=1)
        print("PCA reducer applied to testing")

        # NORMALIZER
        columns_to_normalize = [kind + "_" + str(i) for i in range(0, dim)]
        norm = Normalizer()
        new_data_train[columns_to_normalize] = norm.fit_transform(new_data_train[columns_to_normalize])
        new_data_test[columns_to_normalize] = norm.fit_transform(new_data_test[columns_to_normalize])
        print("Normalizer done")

    return new_data_train, new_data_test

def random_indexing_training_and_testing(new_data_train, new_data_test, kind, dimension_of_embedding_vectors, train_on_all=False):
    BP_columns = [x for x in new_data_train.columns if x.startswith(kind)]
    if BP_columns:
        np.random.seed(100)
        embedding_vector_matrix = np.empty((len(BP_columns), dimension_of_embedding_vectors))
        for i in range(len(BP_columns)):
            embedding_vector_matrix[i] = np.random.normal(0, 1. / np.sqrt(dimension_of_embedding_vectors),
                                                          dimension_of_embedding_vectors)
        new_data_train_emb = pd.DataFrame(new_data_train[BP_columns].values @ embedding_vector_matrix)
        new_data_test_emb = pd.DataFrame(new_data_test[BP_columns].values @ embedding_vector_matrix)

        # Trick to avoid minmax normalization of prepare_training
        replac = {}
        for i in new_data_train_emb.columns:
            replac[i] = kind + str(i) + "###"
        new_data_train_emb.rename(columns=replac, inplace=True)
        new_data_test_emb.rename(columns=replac, inplace=True)
        new_data_train = pd.concat([new_data_train.reset_index(drop=True), new_data_train_emb], axis=1)
        new_data_test = pd.concat([new_data_test.reset_index(drop=True), new_data_test_emb], axis=1)
        new_data_train = new_data_train.drop(BP_columns, axis=1)
        new_data_test = new_data_test.drop(BP_columns, axis=1)

    return new_data_train, new_data_test



def reducing_training_and_testing_with_MI(new_data_train, new_data_test, kind, dim, train_on_all=False):
    BP_columns = [x for x in new_data_train.columns if x.startswith(kind)]
    # MIN MAX Scaler
    #scaler = MinMaxScaler()
    #new_data_train[BP_columns] = scaler.fit_transform(new_data_train[BP_columns])
    #new_data_test[BP_columns] = scaler.transform(new_data_test[BP_columns])
    #print("MinMax Scaler done")

    #train_on_all = True
    if train_on_all:
        train_data_for_reducer = pd.concat([new_data_train[BP_columns],new_data_test[BP_columns]])
        train_data_for_reducer_y = pd.concat([new_data_train["PFS"].map(lambda x: 1 if x > 6 else 0),new_data_test["PFS"].map(lambda x: 1 if x > 6 else 0)])
    else:
        train_data_for_reducer = new_data_train[BP_columns]
        train_data_for_reducer_y = new_data_train["PFS"].map(lambda x: 1 if x > 6 else 0)

    # REDUCER
    # LEARNING THE REDUCER ON THE TRAINING SET
    if BP_columns:
        threshold = 0.03
        mi = mutual_info_regression(train_data_for_reducer,train_data_for_reducer_y)
        selected_features_to_drop = [feature for i, feature in enumerate(BP_columns) if mi[i] < threshold]
        new_data_train = new_data_train.drop(selected_features_to_drop,axis=1)
        new_data_test = new_data_test.drop(selected_features_to_drop,axis=1)

    return new_data_train, new_data_test


def transforming_Braun_dataset(params):
    if os.path.exists('data/Braun_2020_ALL_UNIQUE_final_reduced.csv') and not param_check(params, "recompute"):
        print("Loading reduced dataset")
        new_data = pd.read_csv('data/Braun_2020_ALL_UNIQUE_final_reduced.csv')
        print("...done")
    else:
        normal = True
        if normal:
            print("Loading dataset")
            new_data = pd.read_csv('data/Braun_2020_ALL_UNIQUE_final.csv')
            print(len(new_data))
            print("Dataset loaded")
            # FILLING nan with the Mode
            print("FILLING nan with the Mode")
            filler = new_data.mode(axis=0, dropna=True).loc[0].fillna(0)
            new_data = new_data.fillna(filler)
            print("nan changed with the mode loaded")
        else:
            print("Loading (toy) dataset")
            new_data = pd.read_csv('data/Braun_2020_ALL_UNIQUE_final_toy.csv')
            print(len(new_data))
            print("Dataset loaded")

        new_data_train = new_data.loc[new_data[new_data["TrainTestStatus"] == "Train"].index]
        new_data_test = new_data.loc[new_data[new_data["TrainTestStatus"] == "Test"].index]
        dim = 100
        print("Reducing Biological Pathways to " + str(dim) + " dimensions")
        new_data_train, new_data_test = reducing_training_and_testing(new_data_train, new_data_test, "BP_", dim, train_on_all=True)
        #dim = 400
        #new_data_train, new_data_test = random_indexing_training_and_testing(new_data_train, new_data_test, "BP_", dim, train_on_all=True)
        #new_data_train, new_data_test = reducing_training_and_testing_with_MI(new_data_train, new_data_test, "BP_", dim, train_on_all=True)
        print("... done")
        dim = 200
        print("Reducing Gene Expression to " + str(dim) + " dimensions")
        new_data_train, new_data_test = reducing_training_and_testing(new_data_train, new_data_test, "GE_", dim, train_on_all=True)

        #dim = 400
        #new_data_train, new_data_test = random_indexing_training_and_testing(new_data_train, new_data_test, "GE_", dim, train_on_all=True)
        #new_data_train, new_data_test = reducing_training_and_testing_with_MI(new_data_train, new_data_test, "GE_", dim, train_on_all=True)

        print("... done")

        new_data = pd.concat([new_data_train, new_data_test])
        new_data.to_csv('data/Braun_2020_ALL_UNIQUE_final_reduced.csv', index="False")

    # Dropping useless features
    new_data = new_data.drop(['ST_CD8_IF_ID', 'ST_MAF_Normal_ID', 'ST_MAF_Tumor_ID', 'ST_RNA_ID', 'ST_CNV_ID'], axis=1)
    new_data = new_data.drop(['Number_of_Prior_Therapies'], axis=1)

    if param_check(params, "damien_split"):
        damien_split = pd.read_csv('data/damien_split.csv')
        new_data = new_data.drop(['TrainTestStatus'], axis=1)
        new_data = pd.merge(new_data,damien_split, on=["SUBJID"])



    # SETTING TARGET
    new_data["Outcome"] = new_data["PFS"].map(lambda x: 1 if x > 6 else -1)
    #new_data = new_data.drop(['PFS'], axis=1)


    new_data["TRAIN"] = new_data["TrainTestStatus"].map(lambda x: "TRAIN" if x == "Train" else "TEST")
    new_data = new_data.drop(['TrainTestStatus'], axis=1)

    if param_check(params, "PRIMARY_TUMOR_ONLY"):
        new_data = new_data.loc[new_data["TF_Tumor_Sample_Primary_or_Metastasis"] == "PRIMARY"]


    new_data = new_data.drop([
        #"TF_Tumor_Sample_Primary_or_Metastasis",
        "TF_Site_of_Metastasis",
        "TF_ImmunoPhenotype",
        "TF_Days_from_TumorSample_Collection_and_Start_of_Trial_Therapy"], axis=1)

    if param_check(params, "with_mutations"):
        patients = new_data["SUBJID"].drop_duplicates()
        mutations = mutation_vector_base_per_patient(param_check(params, "HS_features"), 4000, patients, param_check(params, "hotspots"),param_check(params, "random_scaffolds"), scaffold_file = param_check(params, "scaffold_file"))
        new_data = pd.merge(new_data, mutations, on=["SUBJID"], how="outer")
        # out = new_data.loc[new_data["0###"].isnull()]
    new_data = new_data.drop("SUBJID", axis=1)

    #new_data = new_data.drop([x for x in new_data.columns if x.startswith("TF_")], axis=1)

    if "Unnamed: 0" in new_data: new_data = new_data.drop("Unnamed: 0", axis=1)

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
    #print("Reducing Mutations to " + str(dim) + " dimensions")
    #new_data_train, new_data_test = reducing_training_and_testing(new_data_train, new_data_test, "MT_", dim)
    #print("... done")

    TF_columns = [x for x in new_data.columns if x.startswith("TF_")]
    # MIN MAX Scaler
    scaler = MinMaxScaler()
    new_data_train[TF_columns] = scaler.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = scaler.transform(new_data_test[TF_columns])
    print("MinMax Scaler done")

    norm = Normalizer()
    new_data_train[TF_columns] = norm.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = norm.fit_transform(new_data_test[TF_columns])
    print("Normalizer done")


    TF_columns = [x for x in new_data.columns if x.startswith("CF_")]
    # MIN MAX Scaler
    scaler = MinMaxScaler()
    new_data_train[TF_columns] = scaler.fit_transform(new_data_train[TF_columns])
    new_data_test[TF_columns] = scaler.transform(new_data_test[TF_columns])
    print("MinMax Scaler done")

    ###norm = Normalizer()
    #new_data_train[TF_columns] = norm.fit_transform(new_data_train[TF_columns])
    #new_data_test[TF_columns] = norm.fit_transform(new_data_test[TF_columns])
    #print("Normalizer done")


    # Relative weights among groups of features


    if "weights" in params:
#        weights = [0.6,0.1,0.1,0.1]
        weights = params["weights"]
        TF_columns = [x for x in new_data.columns if x.startswith("TF_")]
        GE_columns = [x for x in new_data.columns if x.startswith("GE_")]
        BP_columns = [x for x in new_data.columns if x.startswith("BP_")]
        MT_columns = [x for x in new_data_train.columns if x.startswith("MT_")]
        PS_columns = [x for x in new_data.columns if x.startswith("PS_")]
        CF_columns = [x for x in new_data.columns if x.startswith("CF_")]
        if TF_columns:
            new_data_train[TF_columns] = weights["TF"]*scaler.fit_transform(new_data_train[TF_columns])
            new_data_test[TF_columns] = weights["TF"]*scaler.transform(new_data_test[TF_columns])
        if GE_columns:
            new_data_train[GE_columns] = weights["GE"]*scaler.fit_transform(new_data_train[GE_columns])
            new_data_test[GE_columns] = weights["GE"]*scaler.transform(new_data_test[GE_columns])
        if BP_columns:
            new_data_train[BP_columns] = weights["BP"]*scaler.fit_transform(new_data_train[BP_columns])
            new_data_test[BP_columns] = weights["BP"]*scaler.transform(new_data_test[BP_columns])
        if MT_columns:
            new_data_train[MT_columns] = weights["MT"]**scaler.fit_transform(new_data_train[MT_columns])
            new_data_test[MT_columns] = weights["MT"]**scaler.transform(new_data_test[MT_columns])
        if PS_columns:
            new_data_train[PS_columns] = weights["PS"]**scaler.fit_transform(new_data_train[PS_columns])
            new_data_test[PS_columns] = weights["PS"]**scaler.transform(new_data_test[PS_columns])
        if CF_columns:
            new_data_train[PS_columns] = weights["CF"]**scaler.fit_transform(new_data_train[PS_columns])
            new_data_test[PS_columns] = weights["CF"]**scaler.transform(new_data_test[PS_columns])

    norm = Normalizer()
    new_data_train[new_data_train.columns] = norm.fit_transform(new_data_train[new_data_train.columns])
    new_data_test[new_data_test.columns] = norm.fit_transform(new_data_test[new_data_test.columns])
    print("Normalizer done")


    #data_for_revert_experiment = pd.concat([new_data_train,new_data_test]).join(data_for_revert_exp)
    #to_drop = [x for x in data_for_revert_experiment.columns if x.startswith("Arm###")]
    #data_for_revert_experiment = data_for_revert_experiment.drop(to_drop,axis=1)
    #data_for_revert_experiment["TARGET"] = new_data["Outcome"].map(lambda x: "YES" if x == 1.0 else "NO")
    #data_for_revert_experiment.drop("Outcome",axis=1)
    #data_for_revert_experiment.to_csv('data/Braun_2020_ALL_UNIQUE_revert.csv', index="False")


    return new_data_train, train_classification_labels, new_data_test, test_classification_labels, test_pfs,removed if "removed" in locals() else None
