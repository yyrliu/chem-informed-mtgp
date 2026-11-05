from collections import OrderedDict
import numpy as np

from chem_informed_mtgp.fingerprint_kernel import (
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
)

# Parameter metadata and batch-level exclusions
# concentration_steps = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
concentration_steps = np.array([0.25, 0.5, 0.75, 1.0, 1.25])  # num_steps = 5
annealing_time_steps = np.arange(60, 1200 + 30, 30)  # num_steps = 39
dispense_volume_steps = np.arange(25, 150 + 25, 25)  # num_steps = 6

# fmt: off
_TRAIN_BATCHES = ["3", "4", "5", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27"] # noqa: E501
_CV_BATCHES = ["3", "4", "5", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27"] # noqa: E501
# fmt: on

NOTEBOOK_CONFIG = {
    "param_config": OrderedDict(
        [
            ("annealing_temp", {"range": (40, 150), "step": 5}),  # num_steps = 24
            (
                "annealing_time",
                {"range": (60, 1800), "step": 30, "values": annealing_time_steps},
                # {"range": (60, 1800), "step": 60}
            ),
            (
                "dispense_volume",
                {"range": (25, 300), "step": 25, "values": dispense_volume_steps},
                # {"range": (25, 300), "step": 50},
            ),
            ("spin_speed", {"range": (1000, 6000), "step": 100}),  # num_steps = 17
            ("dispense_speed", {"range": (80, 240), "step": 20}),  # num_steps = 8
            (
                "concentration",
                {"range": (0.1, 2.0), "step": 0.25, "values": concentration_steps},
                # {"range": (0.1, 2.0), "step": 0.25}
            ),
        ]
    ),
    "outliers_by_sample_name": [
        "3-21",  # Wrong PbI2 annealing time
        "4-17",  # Wrong annealing time
        "4-21",  # Wrong annealing time
        "5-17",  # Dark PL image
        "5-21",  # 5-17 pair
        "7-04",  # No CD
        "11-26",
        "11-30",  # incosistent pair
        "14-17",  # CD outlier
        "14-19",  # PL outlier
        # "14-21",  # 14-17 pair
        # "14-23",  # 14-19 pair
        # "22-13",  # Bad PL img
        # "22-14",  # Bad PL img
        "23-01",  # Wrong annealing time
    ],
    "train_batches": _TRAIN_BATCHES,
    "cv_batches": _CV_BATCHES,
    "default_target_column": "score",
    # "default_target_column": "score_norm",
    # "default_target_column": "g_log_score",
    # "default_target_column": "score_5FWHM_ratio",
    # "default_target_column": "CD_score",
    # "default_target_column": "homogeneity_score",
    # "default_target_column": "g_factor_score",
    # "default_target_column": "ldlb_score",
    # "default_target_column": "score_boxcox",
    # "default_target_column": "umap_combined_4",
    "a_site_to_smiles": {
        "MBAI": "C[C@@H](C1=CC=C(C=C1))N",
        "MeOMBAI": "C[C@@H](C1=CC=C(C=C1)OC)N",
        "ClMBAI": "C[C@@H](C1=CC=C(C=C1)Cl)N",
    },
    # "posterior_mean_thresholds_by_a_site": None,
    "posterior_mean_thresholds_by_a_site": {
        "MBAI": 23.1216,
        "MeOMBAI": 14.7235,
        "ClMBAI": 29.6402,
    },
    "condition_outlier_rsd_threshold": None,
    "kernel_type": "matern_3_2",
    "fingerprint_mode": FINGERPRINT_MODE_INDEXED_TANIMOTO,
    "manual_entries": {
        7: [3, 4, 7, 8, 11, 12, 15, 16, 17, 19, 21, 23, 25, 29],
        11: [3, 7, 19, 23],
        12: [42, 46],
    },
}
