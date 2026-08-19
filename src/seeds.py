"""Central seed registry; every random operation draws its seed from ``SEEDS``."""

from __future__ import annotations

import os
import random

import numpy as np


SEEDS: dict[str, int] = {
    "global": 42,
    "train_test_split": 4201,
    "hyperparameter_cv": 4202,
    "randomized_search": 4203,
    "calibration_cv": 4204,
    "threshold_oof_cv": 4205,
    "model_LR": 4210,
    "model_EBM": 4211,
    "model_XGBoost": 4212,
    "model_CatBoost": 4213,
    "bootstrap": 4220,
    "explanation_sampling": 4221,
    "learning_curve": 4230,
}

QUANTILE_METHOD = "linear"
SEARCH_TIE_EPSILON = 1e-6
BOOTSTRAP_CI_METHOD = "percentile"
BOOTSTRAP_CI_LEVEL = 0.95
REPRODUCIBILITY_RTOL = 1e-9
REPRODUCIBILITY_ATOL = 1e-12

ES_PATIENCE = 50
ES_MAX_ROUNDS = 2000


def model_seed(name: str) -> int:
    """Return the frozen seed of one model family."""
    key = f"model_{name}"
    if key not in SEEDS:
        raise KeyError(f"No registered seed for model {name!r}")
    return SEEDS[key]


def seed_everything() -> None:
    """Seed the global RNGs at an execution entry point."""
    random.seed(SEEDS["global"])
    np.random.seed(SEEDS["global"])


def environment_determinism() -> dict[str, object]:
    """Snapshot the determinism-relevant environment for the run manifest."""
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "seeds": dict(SEEDS),
        "quantile_method": QUANTILE_METHOD,
        "search_tie_epsilon": SEARCH_TIE_EPSILON,
        "bootstrap_ci_method": BOOTSTRAP_CI_METHOD,
        "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
        "reproducibility_rtol": REPRODUCIBILITY_RTOL,
        "reproducibility_atol": REPRODUCIBILITY_ATOL,
        "es_patience": ES_PATIENCE,
        "es_max_rounds": ES_MAX_ROUNDS,
    }
