#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import DataSets_scaffolds as ds
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import kaplanmeier as km
import warnings
warnings.filterwarnings("ignore")
random.seed(1024)
np.random.seed(1024)


# In[2]:


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


# In[3]:


def svm_experiment(additional_info={}):
    train_data, train_y, test_data, test_y, _ , excluded_mutation = ds.transforming_Braun_dataset(additional_info)
#    possible_svm_models = [{"kernel": "linear"},{"kernel":"poly"},{"kernel":"poly"} ]
    possible_svm_models = [{"kernel":"poly"} ]

    local_results_frames = pd.DataFrame()
    index = 0
    for params in possible_svm_models:
        results = {}
        results["METHOD"] = "Predictor:SVM" + str(params)
        results["Info"] = str(additional_info)
        print("SVM Meta-parameters : " + str(params))

        _,svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, params)

        results1 = evaluate(test_y, svm_linear_preds, svm_linear_prob)
        if ds.param_check(additional_info, "exclude_mutation"):
            results.update(excluded_mutation.to_dict("records")[0])
        results.update(results1)


        results_frame = pd.DataFrame(results,index = [index])
        if local_results_frames.empty:
            local_results_frames = [results_frame]
        else:
            local_results_frames = [local_results_frames,results_frame]
        local_results_frames = pd.concat(local_results_frames)
        index += 1
    return local_results_frames


def draw_KM_curve_svm_experiment(file_name,additional_info={}, svm_params={}):
    train_data, train_y, test_data, test_y, test_pfs,removed = ds.transforming_Braun_dataset(additional_info)
    print("SVM Meta-parameters : " + str(svm_params))
    print("Info :                " + str(additional_info))
    _,svm_linear_preds, svm_linear_prob = svm_train_test(train_data, train_y, test_data, svm_params)

    results = km.fit(test_pfs,test_y,svm_linear_preds)
    km.plot(results)
    plt.savefig(f'{file_name}.pdf', format='pdf', bbox_inches='tight')
    return None


# In[4]:


weights= {"PS":0.1,"TF":0.1,"CF":0.1,"BP":0.1,"MT":0.3,"GE":0.1}
additional_infos = [
#    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":False,"hotspots":False, "weights":weights,
#     "contig_file":"data/contig_features_reduced2.tsv", "recompute":False},
#    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True,"hotspots":False, "weights":weights, "contig_file":"data/contig_features_reduced2.tsv"},
#    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":True , "hotspots":True, "weights":weights, "contig_file":"data/contig_features_reduced2.tsv"},
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":[]} , 
    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['unique_peptides']}
    ,
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['n_unique_HLA']}, 
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['popcov_but_sqrt4']},
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['unique_peptides','n_unique_HLA']} ,
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['unique_peptides','popcov_but_sqrt4']},
{"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":False, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":['n_unique_HLA','popcov_but_sqrt4']},  

    {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":True, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["n_unique_HLA","unique_peptides","popcov_but_sqrt4"]}

]


overall_results = pd.DataFrame()

for additional_info in additional_infos:
    results_frame = svm_experiment(additional_info=additional_info)
    overall_results_frames = [overall_results,results_frame]
    overall_results = pd.concat(overall_results_frames)


overall_results.to_csv("./data/Bruan_MutationExperiments_3s20.tsv",sep = "\t",index = 0)
overall_results


# In[5]:


#weights= {"PS":0.1,"TF":0.1,"CF":0.3,"BP":0.1,"MT":0.1,"GE":0.1}
#additional_info = {"PRIMARY_TUMOR_ONLY":False,"with_mutations":True, "random_contigs":False, "hotspots":True, "weights":weights, "contig_file":"data/contig_features_reduced2.tsv","HS_features":[]}




#overall_results = pd.DataFrame()
#results_frame = svm_experiment(additional_info=additional_info)
#overall_results_frames = [overall_results,results_frame]
#overall_results = pd.concat(overall_results_frames)


# Intersting Mutation
#for chr_to_remove in range(1,170):
#    print("excluding mutation: " + str(chr_to_remove))
#    additional_info["exclude_mutation"] = chr_to_remove
#    results_frame = svm_experiment(additional_info)
#    overall_results_frames = [overall_results,results_frame]
#    overall_results = pd.concat(overall_results_frames)


#overall_results.to_excel("./data/Bruan_MutationExperiments_3_1.xlsx")
#overall_results

#def plot_feature_aroc_curves(results_df):
#    """
#    Plot AROC curves for different feature combinations.
    
#    Parameters:
#    results_df (pandas.DataFrame): DataFrame containing the results with 'Info' and 'aroc' columns
#    """
    # Create figure and axis
#    plt.figure(figsize=(10, 6))
    
    # Colors for different feature counts
#    colors = ['blue', 'green', 'red']
#    labels = ['1 Feature', '2 Features', '3 Features']
    
    # Process each row in results
#    feature_groups = {1: [], 2: [], 3: []}
    
#    for idx, row in results_df.iterrows():
        # Extract HS_features from Info string
#        info_dict = eval(row['Info'])
#        if 'HS_features' in info_dict:
#            features = info_dict['HS_features']
#            if features:  # Only process non-empty feature lists
#                num_features = len(features)
#                if num_features in feature_groups:
#                    feature_groups[num_features].append(row['aroc'])
    
    # Plot mean AROC values for each feature group
#   x = np.arange(3)
#    means = []
#    for i, count in enumerate([1, 2, 3]):
#        if feature_groups[count]:
#            mean_aroc = np.mean(feature_groups[count])
#            means.append(mean_aroc)
#            plt.plot(x[i], mean_aroc, 'o', color=colors[i], label=labels[i], markersize=10)
    
    # Connect points with lines
#    plt.plot(x[:len(means)], means, '--', color='gray', alpha=0.5)
    
    # Customize plot
#    plt.xlabel('Number of Features')
#    plt.ylabel('AROC Score')
#    plt.title('AROC Scores by Number of Features')
#    plt.grid(True, linestyle='--', alpha=0.7)
#    plt.legend()
    
    # Set y-axis limits to focus on the relevant range
#    plt.ylim(min(means) - 0.05, max(means) + 0.05)
    
    # Set x-axis ticks and labels
#    plt.xticks(x, ['One', 'Two', 'Three'])
    
#    return plt

# Modified main execution code
#def run_aroc_analysis(additional_infos):
#    overall_results = pd.DataFrame()
    
#    for additional_info in additional_infos:
#        results_frame = svm_experiment(additional_info=additional_info)
#        overall_results_frames = [overall_results, results_frame]
#        overall_results = pd.concat(overall_results_frames)
    
    # Save results
#    overall_results.to_csv("./data/Braun_MutationExperiments_3.tsv",sep = "\t",index = 0)
    
    # Create and save plot
#    plt = plot_feature_aroc_curves(overall_results)
#    plt.savefig('./data/aroc_curves.pdf')
#    plt.close()
    
#    return overall_results


# Run the analysis
#results = run_aroc_analysis(additional_infos)

# The function will:
# 1. Run all experiments and save results to Excel
# 2. Generate and save the AROC curves plot
# 3. Return the results DataFrame

# If you want to just create the plot from existing results:
# Assuming you have already run experiments and have results in a DataFrame
#existing_results = pd.read_csv("./data/Braun_MutationExperiments_3.tsv",sep = "\t",header = 0)
#plt = plot_feature_aroc_curves(existing_results)
#plt.show()  # Display the plot
#plt.savefig('./data/aroc_curves.pdf')  # Save the plot
#plt.close()


# In[5]:


draw_KM_curve_svm_experiment('surv_scaff_10base_1',svm_params={'kernel':'poly'},additional_info={"PRIMARY_TUMOR_ONLY":False,"with_mutations":True,"hotspots":True, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv","HS_features":[]}) 


draw_KM_curve_svm_experiment('surv_scaff_10pep_1',svm_params={'kernel':'poly'},additional_info={"PRIMARY_TUMOR_ONLY":False,"with_mutations":True,"hotspots":True, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv","HS_features":['unique_peptides']})

draw_KM_curve_svm_experiment('surv_scaff_10promis_1',svm_params={'kernel':'poly'},additional_info={"PRIMARY_TUMOR_ONLY":False,"with_mutations":True,"hotspots":True, "weights":weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv","HS_features":['popcov_but_sqrt4']})

# In[10]:

#weights = {"PS": 0.1, "TF": 0.1, "CF": 0.1, "BP": 0.1, "MT": 0.1, "GE": 0.1}
draw_KM_curve_svm_experiment('surv_scaff_10HLA_1',svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["n_unique_HLA"]})
draw_KM_curve_svm_experiment('surv_scaff_10HLApep_1',svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["n_unique_HLA","unique_peptides"]})
draw_KM_curve_svm_experiment('surv_scaff_10HLApromis_1',svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["n_unique_HLA","popcov_but_sqrt4"]})

draw_KM_curve_svm_experiment('surv_scaff_10prompep_1',svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["popcov_but_sqrt4","unique_peptides"]})


# In[6]:

draw_KM_curve_svm_experiment('surv_scaff_103_feat_1',svm_params={'kernel': 'poly'},
                             additional_info={"PRIMARY_TUMOR_ONLY": False, "with_mutations": True, "hotspots": True,
                                              "weights": weights, "contig_file":"data/Braun_hg38_scaff_gp10.tsv", "HS_features":["n_unique_HLA",
"unique_peptides","popcov_but_sqrt4"]})


