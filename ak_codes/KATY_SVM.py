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

def find_contig(row):
    # Find all contigs where the mutation's start and end positions are within the range
    matching_contigs = cont2[(cont2["Start_position_2"] >= row["Start_position"]) & 
                             (cont2["End_position_2"] <= row["End_position"])]
    
    # Return the contig IDs as a list
    return "".join(set(matching_contigs["contig"].tolist()))

def contig_delta(braun_file,cont_file,cont2_file,output_file):
    #braun_file = "Braun_mutation_experiment_with_scores.csv"
    #cont_file = "contig_stats_with_tq.csv"
    #cont2_file = "Braun_mutations_hg38_with_epitope_contigs.tsv"
    #output_file = "braun_sub_sorted.csv"

    braun = pd.read_csv(braun_file, sep=",", header=0)
    
    braun_sub_1 = braun[['aroc', 'Count']]
    braun_sub_2 = braun[['aroc', 'Chromosome', 'Start_position', 'End_position', 'Count', 'popcov_but_sqrt', 'Is_in_tq']]
    aroc_base = braun_sub_1.aroc[0]
    braun_sub_2['del_perf'] = braun_sub_2['aroc'].apply(lambda x: x - aroc_base)
    braun_sub_2.sort_values(by=['del_perf'], ascending=False, inplace=True)
    braun_sub_2[braun_sub_2.Is_in_tq == True]

    cont = pd.read_csv(cont_file, sep=",", header=0)
    cont2 = pd.read_csv(cont2_file, sep="\t", header=0)
    cont2.rename(columns={"Start_position": "Start_position_2", "End_position": "End_position_2"}, inplace=True)
    cont2["Chromosome"] = cont2["Chromosome"].apply(lambda x: "chr" + str(x))

    braun_sub_2['contig'] = braun_sub_2.apply(find_contig, axis=1)

    braun_sub_sorted = braun_sub_2.merge(cont2[['gene_name', 'contig']], on='contig').drop_duplicates(['contig', 'gene_name', 'del_perf', 'Is_in_tq', 'Count', 'aroc', 'popcov_but_sqrt']).sort_values(by=['del_perf'], ascending=False)
    braun_sub_sorted = braun_sub_sorted.head(20)

    braun_sub_sorted.to_csv(output_file, index=False) 
    
    return braun_sub_sorted

def annotate_bars(ax, data, ratio_column, hue_column='Is_in_tq'):
    # Get unique hue values
    hue_values = data[hue_column].unique()
    n_hues = len(hue_values)
    
    # Extract the x-tick labels as strings
    x_tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    
    # Filter valid labels that exist in the data
    valid_labels = [label for label in x_tick_labels if label in data['contig'].values]
    
    if not valid_labels:
        print("Warning: No matching labels found in data. Skipping annotation.")
        return
        
    # Sort data by the order of the contigs in the plot
    sorted_data = data.set_index('contig').loc[valid_labels].reset_index()
    
    if sorted_data.empty:
        print("Warning: sorted_data is empty. Skipping annotation.")
        return
        
    # Now handle each hue category separately
    for idx, contig in enumerate(sorted_data['contig'].unique()):
        for hue_idx, hue_val in enumerate(hue_values):
            mask = (sorted_data['contig'] == contig) & (sorted_data[hue_column] == hue_val)
            if not mask.any():
                continue
                
            current_data = sorted_data[mask].iloc[0]
            patch_idx = idx * n_hues + hue_idx
            if patch_idx >= len(ax.patches):
                continue
                
            p = ax.patches[patch_idx]
            
            # Annotate gene name in the middle of the bar
            gene_name = current_data['gene_name']
            ax.annotate(gene_name, 
                       (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                       textcoords="offset points", 
                       xytext=(0, 0), 
                       ha='center', 
                       fontsize=12, 
                       color="white")
            
            # Annotate ratio above the bar
            count_value = current_data[ratio_column]
            ratio = count_value / 2323
            ax.annotate(f'{ratio:.2f}', 
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center', 
                       fontsize=18, 
                       color="black")

# Create figure and axes
fig, axes = plt.subplots(3, 1, figsize=(15, 20))

# Plot 1: 'contig' vs 'del_perf'
sns.barplot(x='contig', y='del_perf', data=braun_sub_sorted, 
            hue='Is_in_tq', dodge=True, palette=color_palette, ax=axes[0])
axes[0].set_title('The 10 most important contigs ranked by decreasing order of performance delta')
axes[0].set_xlabel('Contig ID', fontsize=16)
axes[0].set_ylabel('Performance delta', fontsize=16)
axes[0].tick_params(axis='x', rotation=90, labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend(frameon=False)
annotate_bars(axes[0], braun_sub_sorted, 'Count')

# Plot 2: 'contig' vs 'popcov_but_sqrt'
sns.barplot(x='contig', y='popcov_but_sqrt', data=braun_sub_sorted, 
            hue='Is_in_tq', dodge=True, palette=color_palette, ax=axes[1])
axes[1].set_title('Promiscuity scores')
axes[1].set_xlabel('Contig ID', fontsize=16)
axes[1].set_ylabel('Promiscuity', fontsize=16)
axes[1].tick_params(axis='x', rotation=90, labelsize=16)
axes[1].tick_params(axis='y', labelsize=16)
axes[1].legend(frameon=False)
annotate_bars(axes[1], braun_sub_sorted, 'Count')

# Plot 3: 'contig' vs 'Count'
sns.barplot(x='contig', y='Count', data=braun_sub_sorted, 
            hue='Is_in_tq', dodge=True, palette=color_palette, ax=axes[2])
axes[2].set_title('Number of patients')
axes[2].set_xlabel('Contig ID', fontsize=16)
axes[2].set_ylabel('Number of patients', fontsize=16)
axes[2].tick_params(axis='x', rotation=90, labelsize=16)
axes[2].tick_params(axis='y', labelsize=16)
axes[2].legend(frameon=False)
annotate_bars(axes[2], braun_sub_sorted, 'Count')

sns.despine()
plt.tight_layout()

# Save and show the plot
plt.savefig("performance_delta_contigs.pdf")
plt.show()
        

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
