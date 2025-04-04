# SVM_Immunotherapy
Codes and data to train the SVM immunotherapy response model


To recreate the exact results and figures used in the final publication you will need to follow the code in the CARMEN pipeline up until SVM data generation script, and then continue here in the optimization_validation_and_importance_analysis folder.

First create an environment based on the svn-opti.yml definition 
```conda env create -f svn-opti.yml``` (svn-opti-exact.yml is given to provide exact commit versions used, it is not meant to be installable).

Next the Optimization.ipynb notebook together with the scripts optuna-attempt-* produces the KM curves, the importance-contigs.ipynb notebook with importance-analysis produces the performance delta panel and mutation pie charts and the aroc-curves.ipynb notebook produces the aroc curves comparison.