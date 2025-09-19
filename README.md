# SVM_Immunotherapy
Codes and data to train the SVM immunotherapy response model.

To recreate the exact results and figures used in the final publication you will need to follow the code in the CARMEN pipeline up until SVM data generation script, and then continue in the `optimization_validation_and_importance_analysis` directory.

First create an environment based on the svn-opti.yml definition:

```bash
conda env create -f svn-opti.yml
```

(svn-opti-exact.yml is given to provide exact commit versions used, it is not meant to be installable).

## Model Optimization

The script `model_optimization.py` optimizes a selected model:

```bash
python model_optimization.py <X>
```

where `<X>` is an index of one of the models saved in the `_config.py` file.

After it's finished, copy optimized parameters back to the `_config.py` file.

## Gene Importance Analysis

Run `importance_analysis.py` to generate gene importance tables, and the use `importance_analysis.ipynb` (currently set up for scaffold-level only) to generate figures.

## Running Selected

Next the Optimization.ipynb notebook together with the scripts optuna-attempt-* produces the KM curves, the importance-contigs.ipynb notebook with importance-analysis produces the performance delta panel and mutation pie charts and the aroc-curves.ipynb notebook produces the aroc curves comparison.

Description of model methodology (in Supplementary file 3) and saved classifier models as pickled files are available in the accompanying supplementary data zenodo repository [![CARMEN immunopeptidomics database](https://zenodo.org/badge/DOI/10.5281/zenodo.14859003.svg)](https://doi.org/10.5281/zenodo.14859003).
