# SVM_Immunotherapy

[![CARMEN immunopeptidomics publication analysis results](https://zenodo.org/badge/DOI/10.5281/zenodo.14859003.svg)](https://doi.org/10.5281/zenodo.14859003)

Codes and data to train the SVM immunotherapy response model.

To recreate the exact results and figures used in the final publication you will need to follow the code in the CARMEN pipeline up until SVM data generation script, and then continue in the `optimization_validation_and_importance_analysis` directory.

First create an environment based on the svn-opti.yml definition:

```bash
conda env create -f svn-opti.yml
```

(`svn-opti-exact.yml` is given to provide exact commit versions used, it is not meant to be installable).

>Remember to open each mentioned script and check its configuration, as there are several options how to run the models.

## Model Optimization

The script `model_optimization.py` optimizes a selected model:

```bash
python model_optimization.py <X>
```

where `<X>` is an index of one of the models saved in the `_config.py` file (baseline, peptide, contig, or scaffold).

After it's finished, copy optimized parameters back to the `_config.py` file.

## Gene Importance Analysis

Run `importance_analysis.py` to generate gene importance tables, and then use `importance_analysis.ipynb` (currently set up for scaffold-level only) to analyse them and generate figures.

## Running Classification Models

Use the `train_test_validation_scores.ipynb` notebook to train, test, and validate models saved in the `_config.py` file. The notebook saves pickled model files and generates a set of scores and figures (classiffication efficiencies, ROC curves, Kaplan-Meier response estimation).

## Accompanying Information

Description of model methodology (in Supplementary file 3) and saved classifier models as pickled files are available in the accompanying supplementary data Zenodo repository: https://doi.org/10.5281/zenodo.14859003.
