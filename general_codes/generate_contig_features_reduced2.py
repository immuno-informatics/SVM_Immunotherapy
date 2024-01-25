#!/usr/bin/env python

import pandas as pd

data1 = pd.read_csv('contig_features_reduced.tsv',sep = '\t') 

data1_spec = data1.groupby('contig')['spec_count'].mean().reset_index() 

data2 = data1_spec.merge(data1,on = 'contig').drop('spec_count_y',axis = 1).drop_duplicates(['contig','Sample_ID','SUBJID','spec_count_x']).rename(columns = {'spec_count_x':'spec_count'}) 

data2[['contig','Chromosome','Start_position','End_position','Variant_Classification','Tumor_ref_count','Tumor_alt_count','Sample_ID','SUBJID','unique_peptides','max_depth','mean_depth','spec_count','n_unique_HLA']].to_csv('contig_braun_reduced2.tsv',sep = '\t',index = 0) 
