"""."""

configurations = [
    {
        "plot_label": "Baseline",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": [],
        #  Optimized parameters:
        #    All input data:
        # "mut_vec_len": 887,
        # "clf_params": {
        #     "kernel": "sigmoid",
        #     "C": 44.099706481862746,
        #     "gamma": 1.5592240593208087,
        #     "coef0": 4.64,
        # },
        # "hotspots": False,
        # "weights": {
        #     "MT": 1.0,
        #     "PS": 0.4,
        #     "GE": 1.0,
        #     "Arm": 0.6,
        #     "TF": 0.1,
        #     "BP": 0.5,
        #     "CF": 0.6,
        # },
        #    Only age, gender, and mutation data input:
        "mut_vec_len": 636,
        "clf_params": {
            "kernel": "rbf",
            "C": 1.7411641952699082,
            "gamma": 0.006374575477185076,
        },
        "hotspots": True,
        "weights": {"MT": 0.6, "CF": 0.2},
    },
    {
        "plot_label": "Peptide level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": ["Unique_peptides_narrow", "Promiscuity_narrow"],
        #  Optimized parameters:
        #    All input data:
        # "mut_vec_len": 497,
        # "clf_params": {
        #     "kernel": "rbf",
        #     "C": 2.2300538798718668,
        #     "gamma": 0.8133526685916405,
        # },
        # "hotspots": True,
        # "weights": {
        #     "Arm": 0.4,
        #     "TF": 0.0,
        #     "BP": 0.0,
        #     "MT": 0.7,
        #     "PS": 0.1,
        #     "CF": 0.1,
        #     "GE": 0.6,
        # },
        #    Only age, gender, and mutation data input:
        "mut_vec_len": 481,
        "clf_params": {
            "kernel": "rbf",
            "C": 0.008218035880072005,
            "gamma": 0.6866646010940057,
        },
        "hotspots": True,
        "weights": {"MT": 0.8, "CF": 0.6},
    },
    {
        "plot_label": "Contig level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/Michal_combined_set_14_02_2025.tsv",
        "HS_features": ["unique_peptides", "popcov_but_sqrt"],
        #  Optimized parameters:
        #    All input data:
        # "mut_vec_len": 450,
        # "clf_params": {
        #     "kernel": "poly",
        #     "C": 0.002215884539658647,
        #     "gamma": 0.7213727985310878,
        #     "coef0": -0.01,
        #     "degree": 6,
        # },
        # "hotspots": True,
        # "weights": {
        #     "TF": 0.0,
        #     "GE": 0.3,
        #     "PS": 0.1,
        #     "Arm": 0.2,
        #     "BP": 0.4,
        #     "CF": 0.0,
        #     "MT": 1.0,
        # },
        #    Only age, gender, and mutation data input:
        "mut_vec_len": 584,
        "clf_params": {"kernel": "linear", "C": 0.9738784539012868},
        "hotspots": False,
        "weights": {"MT": 0.7, "CF": 0.2},
    },
    {
        "plot_label": "Scaffold level",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/Braun_hg38_epscaff10_w_score_2025.tsv",
        "HS_features": ["unique_peptides", "popcov_but_sqrt"],
        #  Optimized parameters:
        #    All input data:
        # "mut_vec_len": 1337,
        # "clf_params": {
        #     "kernel": "rbf",
        #     "C": 0.07149793511267355,
        #     "gamma": 0.9762859163222898,
        # },
        # "hotspots": True,
        # "weights": {
        #     "TF": 0.1,
        #     "Arm": 1.0,
        #     "PS": 0.5,
        #     "CF": 1.0,
        #     "BP": 0.6,
        #     "GE": 0.2,
        #     "MT": 0.9,
        # },
        #    Only age, gender, and mutation data input:
        "mut_vec_len": 1710,
        "clf_params": {
            "kernel": "rbf",
            "C": 2.957484657922523,
            "gamma": 0.40331433305943104,
        },
        "hotspots": True,
        "weights": {"MT": 0.9, "CF": 0.6},
    },
]
