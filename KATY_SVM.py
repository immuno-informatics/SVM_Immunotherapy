import pandas as pd
import numpy as np
import random
import DataSets as ds
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import kaplanmeier as km 
import matplotlib.pyplot as plt
import warnings
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

def svm_experiment(additional_info={}):
    train_data, train_y, test_data, test_y, _ = ds.transforming_Braun_dataset(additional_info)
    possible_svm_models = [{"kernel": "linear"},{"kernel":"rbf"},{"kernel":"poly"} ]

    local_results_frames = pd.DataFrame()
    index = 0
    for params in possible_svm_models:
        results = {}
        results["METHOD"] = "Predictor:SVM" + str(params)
        results["Info"] = str(additional_info)
        print("SVM Meta-parameters : " + str(params))

        _,svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, params)

        results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
        results.update(results1)


        results_frame = pd.DataFrame(results,index = [index])
        if local_results_frames.empty:
            local_results_frames = [results_frame]
        else:
            local_results_frames = [local_results_frames,results_frame]
        local_results_frames = pd.concat(local_results_frames)
        index += 1
    return local_results_frames


def draw_KM_curve_svm_experiment(additional_info={}, svm_params={}):
    train_data, train_y, test_data, test_y, test_pfs = ds.transforming_Braun_dataset(additional_info)
    print("SVM Meta-parameters : " + str(svm_params))
    print("Info :                " + str(additional_info))
    _,svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, svm_params)

    results = km.fit(test_pfs,test_y,svm_linear_preds)
    km.plot(results) 
    plt.savefig("comb_newcode_gp$1_$2.pdf",dpi = 300) 
    return None

weights= {"PS":0.1,"TF":0.1,"CF":0.3,"BP":0.1,"MT":0.1,"GE":0.1}
additional_infos = [
    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":False,"hotspots":False, "weights":weights,
     "contig_file":"data/scaffolds_mutation_all.tsv", "recompute":False},
    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True,"hotspots":False, "weights":weights, "contig_file":"data/scaffolds_mutation_all.tsv"},
    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":True , "hotspots":True, "weights":weights, "contig_file":"data/scaffolds_mutation_all.tsv"},
    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":True, "weights":weights, "contig_file":"data/scaffolds_mutation_all.tsv"}

]


overall_results = pd.DataFrame()

for additional_info in additional_infos:
    results_frame = svm_experiment(additional_info=additional_info)
    overall_results_frames = [overall_results,results_frame]
    overall_results = pd.concat(overall_results_frames)


#overall_results.to_excel("./data/Bruan_MutationExperiments_3.xlsx") 
overall_results.to_csv("./data/scaffolds_gp_$1_$2.tsv",sep = "\t",index = 0)
overall_results

draw_KM_curve_svm_experiment(svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/scaffolds_mutation_all.tsv"})
