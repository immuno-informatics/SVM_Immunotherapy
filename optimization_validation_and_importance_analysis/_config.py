"""."""

configurations = [
    {
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "hotspots": False,
        "weights": {"PS": 1.0, "TF": 1.0, "CF": 1.0, "BP": 1.0, "MT": 1.0, "GE": 1.0},
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": [],
        "plot_label": "Baseline",
        #    Mutation vector optim:
        # "mut_vec_len": 21,
        # "clf_params": {"kernel": "rbf"},
        #    "Full" optim limited params:
        # "mut_vec_len": 887,
        # "clf_params": {
        #     "C": 50,
        #     "coef0": 3,
        #     "degree": 2,
        #     "gamma": 0.01,
        #     "kernel": "poly",
        # },
        #    Full optim test:
        "mut_vec_len": 887,
        "clf_params": {'kernel': 'rbf', 'C': 876.518, 'gamma': 0.7431, 'coef0': 26.910000000000004, 'degree': 1},
    },
    {
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "hotspots": True,
        "weights": {"PS": 1.0, "TF": 1.0, "CF": 1.0, "BP": 1.0, "MT": 1.0, "GE": 1.0},
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": ["Unique_peptides_narrow", "Promiscuity_narrow"],
        "plot_label": "Peptide level",
        #    Mutation vector optim:
        # "mut_vec_len": 149,
        # "clf_params": {"kernel": "rbf"},
        #    "Full" optim limited params:
        # "mut_vec_len": 243,
        # "clf_params": {
        #     "C": 100,
        #     "coef0": 5,
        #     "degree": 3,
        #     "gamma": 0.01,
        #     "kernel": "poly",
        # },
        #    Full optim test:
        "mut_vec_len": 243,
        "clf_params": {'kernel': 'rbf', 'C': 260.024, 'gamma': 0.5416, 'coef0': 99.7, 'degree': 3},
    },
    {
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "hotspots": True,
        "weights": {"PS": 1.0, "TF": 1.0, "CF": 1.0, "BP": 1.0, "MT": 1.0, "GE": 1.0},
        "contig_file": "../data/Michal_combined_set_14_02_2025.tsv",
        "HS_features": ["unique_peptides", "popcov_but_sqrt"],
        "plot_label": "Contig level",
        #    Mutation vector optim:
        # "mut_vec_len": 327,
        # "clf_params": {"kernel": "rbf"},
        #    "Full" optim limited params:
        # "mut_vec_len": 375,
        # "clf_params": {
        #     "C": 10,
        #     "coef0": 3,
        #     "degree": 5,
        #     "gamma": "auto",
        #     "kernel": "poly",
        # },
        #    Full optim test:
        "mut_vec_len": 375,
        "clf_params": {'kernel': 'rbf', 'C': 5.6290000000000004, 'gamma': 0.4269, 'coef0': 99.7, 'degree': 1},
    },
    {
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "hotspots": False,
        "weights": {"PS": 1.0, "TF": 1.0, "CF": 1.0, "BP": 1.0, "MT": 1.0, "GE": 1.0},
        "contig_file": "../data/Braun_hg38_epscaff10_w_score_2025.tsv",
        "HS_features": ["popcov_but_sqrt", "unique_peptides"],
        "plot_label": "Scaffold level",
        #    Mutation vector optim:
        # "mut_vec_len": 77,
        # "clf_params": {"kernel": "rbf"},
        #    "Full" optim limited params:
        # "mut_vec_len": 881,
        # "clf_params": {"C": 1, "coef0": 0, "degree": 5, "gamma": 0.1, "kernel": "poly"},
        #    Full optim test:
        "mut_vec_len": 887,
        "clf_params": {'kernel': 'rbf', 'C': 9.043, 'gamma': 0.2288, 'coef0': 84.88000000000001, 'degree': 2},
    },
]
