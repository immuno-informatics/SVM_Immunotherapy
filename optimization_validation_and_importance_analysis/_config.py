"""."""

configurations = [
    # {
    #     "plot_label": "Baseline",
    #     "PRIMARY_TUMOR_ONLY": False,
    #     "with_mutations": True,
    #     "random_contigs": False,
    #     "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
    #     "HS_features": [],
    #     #  Optimized parameters:
    #     #    All input data:
    #     # "mut_vec_len": 887,
    #     # "clf_params": {
    #     #     "kernel": "sigmoid",
    #     #     "C": 44.099706481862746,
    #     #     "gamma": 1.5592240593208087,
    #     #     "coef0": 4.64,
    #     # },
    #     # "hotspots": False,
    #     # "weights": {
    #     #     "MT": 1.0,
    #     #     "PS": 0.4,
    #     #     "GE": 1.0,
    #     #     "Arm": 0.6,
    #     #     "TF": 0.1,
    #     #     "BP": 0.5,
    #     #     "CF": 0.6,
    #     # },
    #     #    Only age, gender, and mutation data input:
    #     # "mut_vec_len": 636,
    #     # "clf_params": {
    #     #     "kernel": "rbf",
    #     #     "C": 1.7411641952699082,
    #     #     "gamma": 0.006374575477185076,
    #     # },
    #     # "hotspots": True,
    #     # "weights": {"MT": 0.6, "CF": 0.2},
    #     # "validation_contig_file": "/data/teamgdansk/katy-variants/validation-datasets/patient_mutations_alternative_scoring_narrow_broad.tsv",
    # },
    {
        "plot_label": "Baseline_hotspots_True",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": [],
        #  Optimized parameters:
        #    NEW data schema:
        "mut_vec_len": 636,
        "clf_params": {
            "kernel": "sigmoid",
            "C": 111.04467130528178,
            "gamma": 0.05111235513590068,
            "coef0": 0.71,
        },
        "hotspots": True,
        "weights": {"GE": 0.4, "BP": 0.6, "MT": 0.8, "CF": 0.5},
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
        # "mut_vec_len": 481,
        # "clf_params": {
        #     "kernel": "rbf",
        #     "C": 0.008218035880072005,
        #     "gamma": 0.6866646010940057,
        # },
        # "hotspots": True,
        # "weights": {"MT": 0.8, "CF": 0.6},
        # "validation_contig_file": "/data/teamgdansk/katy-variants/validation-datasets/patient_mutations_alternative_scoring_narrow_broad.tsv",
        #    NEW data schema:
        "mut_vec_len": 3455,
        "clf_params": {
            "kernel": "sigmoid",
            "C": 0.012143596510890463,
            "gamma": 0.00803714910274797,
            "coef0": 5.04,
        },
        "hotspots": True,
        "weights": {"GE": 0.0, "MT": 0.3, "CF": 0.0, "BP": 1.0},
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
        # "mut_vec_len": 584,
        # "clf_params": {"kernel": "linear", "C": 0.9738784539012868},
        # "hotspots": False,
        # "weights": {"MT": 0.7, "CF": 0.2},
        # "validation_contig_file": "/data/teamgdansk/katy-variants/validation-datasets/patient_mutations_contigs.tsv",
        #    NEW data schema:
        "mut_vec_len": 450,
        "clf_params": {
            "kernel": "rbf",
            "C": 0.012002523737685003,
            "gamma": 0.5653724193333343,
        },
        "hotspots": False,
        "weights": {"BP": 0.5, "GE": 0.1, "MT": 0.7, "CF": 0.5},
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
        # "mut_vec_len": 1710,
        # "clf_params": {
        #     "kernel": "rbf",
        #     "C": 2.957484657922523,
        #     "gamma": 0.40331433305943104,
        # },
        # "hotspots": True,
        # "weights": {"MT": 0.9, "CF": 0.6},
        # "validation_contig_file": "/data/teamgdansk/katy-variants/validation-datasets/patient_mutations_scaffolds10.tsv",
        #    NEW data schema:
        "mut_vec_len": 1369,
        "clf_params": {
            "kernel": "rbf",
            "C": 2.8556693848200836,
            "gamma": 1.843549982163684,
        },
        "hotspots": True,
        "weights": {"MT": 0.6, "BP": 0.1, "CF": 0.2, "GE": 0.0},
    },
    {
        "plot_label": "Baseline_hotspots_False",
        "PRIMARY_TUMOR_ONLY": False,
        "with_mutations": True,
        "random_contigs": False,
        "contig_file": "../data/braun_mutations_alternative_scoring_narrow_broad.tsv",
        "HS_features": [],
        #  Optimized parameters:
        #    NEW data schema:
        "mut_vec_len": 640,
        "clf_params": {
            "kernel": "poly",
            "C": 0.007557195141845231,
            "gamma": 0.03219807318116276,
            "coef0": 8.68,
            "degree": 6,
        },
        "hotspots": False,
        "weights": {"GE": 0.1, "CF": 0.7, "MT": 0.4, "BP": 0.7},
    },
]
