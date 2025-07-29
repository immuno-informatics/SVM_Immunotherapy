"""."""

configurations = [
    {
        "plot_label": "Baseline",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": [],
        #    Mutation vector optim:
        # "mut_vec_len": 21,
        # "clf_params": {"kernel": "rbf"},
        #    Grid optim limited params:
        "mut_vec_len": 887,
        "clf_params": {
            "C": 50,
            "coef0": 3,
            "degree": 2,
            "gamma": 0.01,
            "kernel": "poly",
        },
        "hotspots": False,
        "weights": {
            "PS": 1.0,
            "TF": 1.0,
            "CF": 1.0,
            "BP": 1.0,
            "MT": 1.0,
            "GE": 1.0,
            "Arm": 1.0,
        },
    },
    {
        "plot_label": "Peptide level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": ["Unique_peptides_narrow", "Promiscuity_narrow"],
        #    Mutation vector optim:
        # "mut_vec_len": 149,
        # "clf_params": {"kernel": "rbf"},
        #    Grid optim limited params:
        "mut_vec_len": 243,
        "clf_params": {
            "C": 100,
            "coef0": 5,
            "degree": 3,
            "gamma": 0.01,
            "kernel": "poly",
        },
        "hotspots": True,
        "weights": {
            "PS": 1.0,
            "TF": 1.0,
            "CF": 1.0,
            "BP": 1.0,
            "MT": 1.0,
            "GE": 1.0,
            "Arm": 1.0,
        },
    },
    {
        "plot_label": "Contig level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/Michal_combined_set_14_02_2025.tsv",
        "HS_features": ["unique_peptides", "popcov_but_sqrt"],
        #    Mutation vector optim:
        # "mut_vec_len": 327,
        # "clf_params": {"kernel": "rbf"},
        #    Grid optim limited params:
        # "mut_vec_len": 375,
        # "clf_params": {
        #     "C": 10,
        #     "coef0": 3,
        #     "degree": 5,
        #     "gamma": "auto",
        #     "kernel": "poly",
        # },
        # "hotspots": True,
        # "weights": {
        #     "PS": 1.0,
        #     "TF": 1.0,
        #     "CF": 1.0,
        #     "BP": 1.0,
        #     "MT": 1.0,
        #     "GE": 1.0,
        #     "Arm": 1.0,
        # },
        #    Test
        "mut_vec_len": 1295,
        "clf_params": {'kernel': 'rbf', 'C': 0.0021885029473738527, 'gamma': 1.1507545673493367},
        "hotspots": True,
        "weights": {'PS': 0.2, 'TF': 0.6, 'CF': 0.9, 'BP': 0.8, 'MT': 0.7, 'GE': 0.3, 'Arm': 0.9},
    },
    {
        "plot_label": "Scaffold level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/Braun_hg38_epscaff10_w_score_2025.tsv",
        "HS_features": ["unique_peptides", "popcov_but_sqrt"],
        #    Mutation vector optim:
        # "mut_vec_len": 77,
        # "clf_params": {"kernel": "rbf"},
        #    Grid optim limited params:
        "mut_vec_len": 881,
        "clf_params": {"C": 1, "coef0": 0, "degree": 5, "gamma": 0.1, "kernel": "poly"},
        "hotspots": False,
        "weights": {
            "PS": 1.0,
            "TF": 1.0,
            "CF": 1.0,
            "BP": 1.0,
            "MT": 1.0,
            "GE": 1.0,
            "Arm": 1.0,
        },
    },
]
